"""Forward-chaining model selection and fixed-iteration refits on raw replay.

Every fold refits calibration on its chronological prefix. Final test dates are
never used to select methods, seeds, checkpoints or baseline parameters.
"""

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import hashlib
import json
import os
import tempfile
import time
import numpy as np
import pandas as pd
from .agents import MLP, Replay, bellman, emit_nstep
from .ppo import Actor, gae, log_probability
from .policies import Fixed, PID, BudgetLinear
from .adaptive import MarketFilter
from .replay import ReplayData, ReplayEnv, make_replay_day, rollout_replay
from .control import TrafficDQN, TrafficPPO, ResidualPPO, CostController, fit_control

SCENARIOS = {
    "standard": {},
    "competition_1.5": {"competition": 1.5},
    "evening_traffic_3": {"traffic": 3.0},
    "budget_0.15": {"budget_fraction": 0.15},
    "budget_0.60": {"budget_fraction": 0.6},
    "quality_0.5": {"quality": 0.5},
    "coverage_0.5": {"coverage": 0.5},
}


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)
        )
        + "\n"
    )
    temporary.replace(path)


def source_hashes():
    return {
        p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path("bidpace").glob("*.py")) + [Path("run.py")]
    }


def save_arrays(path, arrays):
    """One complete atomic archive write, without worker read/modify/write races."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".weights-", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            np.savez_compressed(output, **arrays)
            output.flush()
            os.fsync(output.fileno())
        Path(temporary).replace(path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


class Ensemble:
    """Average Q-values or latent actor means before choosing a common action."""

    def __init__(self, policies, family):
        self.policies, self.family = list(policies), family
        self.requires_traffic = any(getattr(p, "requires_traffic", False) for p in self.policies)

    def reset(self, day):
        for policy in self.policies:
            policy.reset(day)

    def act(self, obs):
        predictions = np.stack([p.model(p.encode(obs)) for p in self.policies])
        if self.family == "dqn":
            action = int(predictions.mean(0).argmax())
            values = [p.apply(action) for p in self.policies]
        else:
            latent = float(predictions.mean())
            values = [p.apply(latent) for p in self.policies]
        if not np.allclose(values, values[0], atol=1e-10):
            raise AssertionError("Ensemble controller states diverged")
        return float(values[0])


def policy_spec(data, variant):
    """Describe one supported learner and its training-fitted controller."""
    if variant not in ("traffic_dqn", "traffic_ppo", "residual_ppo"):
        raise ValueError(f"Unsupported RL variant: {variant}")
    spec = dict(
        family="dqn" if variant == "traffic_dqn" else "ppo",
        campaigns=sorted(data.profiles),
        variant=variant,
    )
    if variant == "residual_ppo":
        spec["controller"] = {k: v.tolist() for k, v in fit_control(data).items()}
        spec["residual_scale"] = 0.5
    return spec


def blank_policy(spec):
    variant = spec["variant"]
    if variant == "traffic_dqn":
        return TrafficDQN(None)
    if variant == "traffic_ppo":
        return TrafficPPO(None)
    if variant == "residual_ppo":
        return ResidualPPO(None, spec)
    raise ValueError(f"Unsupported RL variant: {variant}")


def create_policy(tensors, spec):
    model = MLP(
        input_dim=tensors["w0"].shape[0],
        output_dim=tensors["w2"].shape[1],
        hidden=(tensors["w0"].shape[1], tensors["w1"].shape[1]),
    )
    if set(tensors) != set(model.p) or any(not np.isfinite(v).all() for v in tensors.values()):
        raise ValueError("Invalid model tensors")
    model.p = {k: v.copy() for k, v in tensors.items()}
    policy = blank_policy(spec)
    policy.model = model
    return policy


def training_controls(cfg, variant, rng):
    """Randomize budget and competition using a policy-independent RNG stream."""
    u_budget, u_price = rng.random(2)
    domain = cfg["domain_randomization"]
    budget = domain["budget"][0] + u_budget * np.diff(domain["budget"])[0]
    competition = domain["competition"][0] + u_price * np.diff(domain["competition"])[0]
    return float(budget), float(competition)


def fit_rl(data, cfg, variant, seed, stop=None):
    """Fixed-step RL training; no validation checkpoint selection in this loop."""
    spec = policy_spec(data, variant)
    settings = cfg["dqn"] if spec["family"] == "dqn" else cfg["ppo"]
    maximum = settings["episodes"]
    stop = maximum if stop is None else int(stop)
    if not 0 <= stop <= maximum:
        raise ValueError("Invalid refit stopping iteration")
    checkpoints = set(settings["checkpoints"]) | {stop}
    policy = blank_policy(spec)
    output_dim = len(policy.actions) if spec["family"] == "dqn" else 1
    model_class = MLP if spec["family"] == "dqn" else Actor
    model = model_class(seed, input_dim=policy.input_dim, output_dim=output_dim)
    policy.model = model
    if spec["family"] == "ppo":
        model.p["w2"].fill(0)
        model.p["b2"].fill(0)
    auxiliary = MLP(
        seed + 1,
        input_dim=policy.input_dim,
        output_dim=output_dim if spec["family"] == "dqn" else 1,
    )
    if spec["family"] == "dqn":
        auxiliary.copy_from(model)
    replay = (
        Replay(settings.get("replay_capacity", 80000), input_dim=policy.input_dim)
        if spec["family"] == "dqn"
        else None
    )
    snapshots = {0: {k: v.copy() for k, v in model.p.items()}}
    # Keep curriculum, exploration and coverage on separate seeded streams.
    curriculum = np.random.default_rng(seed + 19000)
    rng = np.random.default_rng(seed + 49000)
    exposure = np.random.default_rng(seed + 89000)
    batch = []
    logs = []
    recent = []
    losses = []
    steps = updates = 0
    started = time.monotonic()
    for episode in range(1, stop + 1):
        if getattr(policy, "requires_traffic", False):
            adv = spec["campaigns"][int(curriculum.integers(len(spec["campaigns"])))]
            candidates = [t for t in data.templates["train"] if t.advertiser == adv]
            template = candidates[int(curriculum.integers(len(candidates)))]
        else:
            template = data.templates["train"][
                int(curriculum.integers(len(data.templates["train"])))
            ]
        budget, competition = training_controls(cfg, variant, curriculum)
        coverage = (
            float(exposure.uniform(*cfg["domain_randomization"]["coverage"]))
            if getattr(policy, "requires_traffic", False)
            else 1.0
        )
        day = make_replay_day(
            template,
            budget_fraction=budget,
            competition=competition,
            coverage=coverage,
            seed=int(exposure.integers(2**31)),
        )
        env = ReplayEnv(day, features=getattr(policy, "requires_traffic", False))
        policy.reset(day)
        obs = policy.encode(env.obs())
        queue = deque()
        states = []
        latents = []
        logps = []
        rewards = []
        values = []
        dones = []
        if spec["family"] == "ppo":
            batch_index = (episode - 1) // settings["batch_episodes"]
            num_batches = int(np.ceil(maximum / settings["batch_episodes"]))
            exploration = settings["std_final"] + (
                settings["std_initial"] - settings["std_final"]
            ) * (1 - batch_index / max(num_batches - 1, 1))
        else:
            exploration = max(
                settings["epsilon_final"],
                1 - (1 - settings["epsilon_final"]) * episode / (maximum * 0.75),
            )
        while not env.done:
            if spec["family"] == "ppo":
                mean = float(model(obs)[0])
                latent = float(rng.normal(mean, exploration))
                action = policy.apply(latent)
                states.append(obs.copy())
                latents.append(latent)
                logps.append(float(log_probability(latent, mean, exploration)))
                values.append(float(auxiliary(obs)[0]))
            else:
                index = (
                    int(rng.integers(output_dim))
                    if rng.random() < exploration
                    else int(model(obs).argmax())
                )
                action = policy.apply(index)
            raw_next, reward, done = env.step(action)
            # A shared training-only scale preserves reward comparability across campaigns.
            nxt = policy.encode(raw_next)
            normalized = reward / data.reward_scale
            steps += 1
            if spec["family"] == "ppo":
                rewards.append(normalized)
                dones.append(done)
            else:
                queue.append((obs.copy(), index, normalized, nxt.copy(), done))
                emit_nstep(queue, replay, settings["nstep"], cfg["gamma"], flush=done)
                if replay.size >= settings["warmup"] and steps % settings["update_every"] == 0:
                    x, acts, r, ns, discount = replay.sample(rng, settings["batch_size"])
                    targets = bellman(model, auxiliary, r, ns, discount, double=True)
                    losses.append(model.update(x, acts, targets, settings["learning_rate"]))
                    updates += 1
                    if updates % settings["target_every_updates"] == 0:
                        auxiliary.copy_from(model)
            obs = nxt
        recent.append(env.clicks)
        if spec["family"] == "ppo":
            advantage, returns = gae(
                rewards, values + [0.0], dones, cfg["gamma"], settings["lambda"]
            )
            batch.extend(zip(states, latents, logps, advantage, returns))
            if episode % settings["batch_episodes"] == 0 or episode == stop:
                x = np.stack([b[0] for b in batch])
                u, old, adv, ret = [np.asarray([b[j] for b in batch]) for j in range(1, 5)]
                adv = (adv - adv.mean()) / max(adv.std(), 1e-8)
                stop_epoch = False
                for _ in range(settings["epochs"]):
                    order = rng.permutation(len(x))
                    for offset in range(0, len(x), settings["minibatch"]):
                        ids = order[offset : offset + settings["minibatch"]]
                        loss, g, kl = model.policy_gradients(
                            x[ids], u[ids], old[ids], adv[ids], exploration, settings["clip"]
                        )
                        if kl > settings["target_kl"]:
                            stop_epoch = True
                            break
                        model.apply_gradients(g, settings["actor_lr"])
                        auxiliary.update(
                            x[ids], np.zeros(len(ids), dtype=int), ret[ids], settings["critic_lr"]
                        )
                        updates += 1
                        losses.append(loss)
                    if stop_epoch:
                        break
                batch.clear()
        interval = settings.get("batch_episodes", 100)
        if episode % interval == 0 or episode == stop:
            logs.append(
                dict(
                    episode=episode,
                    clicks=float(np.mean(recent)),
                    loss=float(np.mean(losses)) if losses else 0.0,
                    exploration=exploration,
                )
            )
            recent.clear()
            losses.clear()
        if episode in checkpoints:
            snapshots[episode] = {k: v.copy() for k, v in model.p.items()}
            print(f"{variant} seed={seed} trained={episode}", flush=True)
    # Store inference tensors at fixed iterations; optimizer state is not exported.
    return dict(
        snapshots=snapshots,
        spec=spec,
        logs=logs,
        steps=steps,
        updates=updates,
        seconds=time.monotonic() - started,
    )


def fit_worker(task):
    cfg, variant, seed, train_dates, validation_dates, stop = task
    data = ReplayData(cfg["data_file"], train_dates, validation_dates)
    return variant, seed, fit_rl(data, cfg, variant, seed, stop)


def validation_days(data, cfg):
    return [
        (template.date, template.advertiser, i, make_replay_day(template, **case))
        for template in data.templates["validation"]
        for i, case in enumerate(cfg["validation_conditions"])
    ]


def value(policy, days):
    return float(np.mean([rollout_replay(day, policy)["clicks"] for _, _, _, day in days]))


def market_tensors(data, quantiles=101):
    return fit_control(data, quantiles)


class Baseline:
    def __init__(self, spec, market=None):
        self.spec, self.market = spec, market
        self.requires_traffic = spec["kind"] in ("mpc", "traffic_linear")

    def reset(self, day):
        s = self.spec
        kind = s["kind"]
        adv = str(day.advertiser)
        raw_alpha = s.get("alphas", {}).get(adv, s.get("alpha", 1.0))
        alpha = 1.0 if kind == "budget_linear" else float(raw_alpha)
        self.constant = kind == "constant"
        if kind == "mpc":
            self.controller = CostController(self.market, s["rate"], s["reserve"])
            self.controller.reset(day)
        elif kind == "budget_linear":
            self.controller = BudgetLinear(
                s["fractions"], s["alphas"].get(adv, s.get("fallback", []))
            )
            self.controller.reset(day)
        elif kind == "pid":
            self.controller = PID(base=alpha * s.get("factor", 1.0), **s["parameters"])
            self.controller.reset(day)
        elif kind in ("market_linear", "traffic_linear"):
            self.filter = MarketFilter(self.market, rate=s["rate"])
            self.filter.reset(day)
            self.filter.budget = day.budget / day.profile.price
            self.filter.auctions = day.profile.mean_daily_impressions
            self.alpha = alpha * s.get("factor", 1.0)
        else:
            if kind == "privileged_linear":
                alpha *= day.price_multiplier
            self.controller = Fixed(alpha, self.constant)

    def act(self, obs):
        if self.spec["kind"] in ("market_linear", "traffic_linear"):
            self.filter.observe(obs)
            traffic = (
                max(float(obs[8]) * 4.0, 0.05) if self.spec["kind"] == "traffic_linear" else 1.0
            )
            alpha = float(
                np.clip(
                    self.alpha * self.filter.price / traffic ** self.spec.get("power", 0.0),
                    0.0,
                    3.0,
                )
            )
            self.filter.record(obs, alpha)
            return alpha
        return float(np.clip(self.controller.act(obs), 0.0, 3.0))


def tune_baselines(folds, cfg):
    """Use exactly the same chronological validation conditions as the agents."""
    grid = np.unique(
        np.r_[np.geomspace(0.08, 3.0, 20), [0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0]]
    )
    rows = []
    grid_scores = []
    markets = [market_tensors(data) for data in folds]
    all_days = [validation_days(data, cfg) for data in folds]

    def score_spec(name, spec):
        scores = []
        for fold, (data, days, market) in enumerate(zip(folds, all_days, markets)):
            score = value(Baseline(spec, market), days)
            scores.append(score)
            rows.append(
                dict(policy=name, fold=fold, score=score, spec=json.dumps(spec, sort_keys=True))
            )
        return float(np.mean(scores))

    for kind in ["constant", "linear", "privileged_linear"]:
        for alpha in grid:
            spec = dict(kind=kind, alpha=float(alpha))
            total = score_spec(kind, spec)
            if kind == "linear":
                for fold, days in enumerate(all_days):
                    policy = Baseline(spec)
                    for date, adv, case, day in days:
                        grid_scores.append(
                            dict(
                                fold=fold,
                                advertiser=adv,
                                case=case,
                                alpha=float(alpha),
                                clicks=rollout_replay(day, policy)["clicks"],
                            )
                        )
        print("Tuned cross-date", kind, flush=True)
    table = pd.DataFrame(rows)

    def best(name):
        choices = table[table.policy == name].groupby("spec", sort=False).score.mean()
        return json.loads(choices.idxmax())

    specs = {kind: best(kind) for kind in ["constant", "linear", "privileged_linear"]}
    scores = pd.DataFrame(grid_scores)
    campaign = {}
    for adv, g in scores.groupby("advertiser"):
        campaign[str(adv)] = float(g.groupby("alpha").clicks.mean().idxmax())
    specs["campaign_linear"] = dict(kind="linear", alphas=campaign, alpha=specs["linear"]["alpha"])
    score_spec("campaign_linear", specs["campaign_linear"])
    # Budget anchors are fitted across dates, with competition conditions kept.
    fractions = sorted({case["budget_fraction"] for case in cfg["validation_conditions"]})
    anchors = {str(a): [] for a in scores.advertiser.unique()}
    for fraction in fractions:
        cases = [
            i
            for i, c in enumerate(cfg["validation_conditions"])
            if c["budget_fraction"] == fraction
        ]
        for adv, g in scores[scores.case.isin(cases)].groupby("advertiser"):
            anchors[str(adv)].append(float(g.groupby("alpha").clicks.mean().idxmax()))
    fallback = []
    for fraction in fractions:
        cases = [
            i
            for i, c in enumerate(cfg["validation_conditions"])
            if c["budget_fraction"] == fraction
        ]
        fallback.append(
            float(scores[scores.case.isin(cases)].groupby("alpha").clicks.mean().idxmax())
        )
    specs["budget_linear"] = dict(
        kind="budget_linear", fractions=fractions, alphas=anchors, fallback=fallback
    )
    score_spec("budget_linear", specs["budget_linear"])
    pid_candidates = []
    import itertools

    for factor, kp, ki, kd in itertools.product(
        (0.7, 1.0, 1.3), (1.0, 3.0, 6.0), (0.0, 0.2), (0.0, 1.0)
    ):
        spec = dict(
            kind="pid",
            alphas=campaign,
            alpha=specs["linear"]["alpha"],
            factor=factor,
            parameters=dict(kp=kp, ki=ki, kd=kd),
        )
        pid_candidates.append((score_spec("pid", spec), spec))
    specs["pid"] = max(pid_candidates, key=lambda x: x[0])[1]
    quality_candidates = []
    for _, parent in sorted(pid_candidates, key=lambda x: x[0], reverse=True)[:6]:
        for power, target in itertools.product((-0.5, 0.0, 0.5), ("volume", "quality")):
            spec = dict(
                parent, parameters=dict(parent["parameters"], quality_power=power, target=target)
            )
            quality_candidates.append((score_spec("quality_pid", spec), spec))
    specs["quality_pid"] = max(quality_candidates, key=lambda x: x[0])[1]
    market_candidates = []
    for factor, rate in itertools.product((0.7, 1.0, 1.3), (0.1, 0.25, 0.5)):
        spec = dict(
            kind="market_linear",
            alphas=campaign,
            alpha=specs["linear"]["alpha"],
            factor=factor,
            rate=rate,
        )
        market_candidates.append((score_spec("market_linear", spec), spec))
    specs["market_linear"] = max(market_candidates, key=lambda x: x[0])[1]
    if cfg.get("traffic_baselines", False):
        choices = []
        for rate, reserve in itertools.product((0.0, 0.25, 0.5, 1.0), (0.8, 1.0, 1.2)):
            spec = dict(kind="mpc", rate=rate, reserve=reserve)
            choices.append((score_spec("mpc", spec), spec))
        specs["mpc"] = max(choices, key=lambda x: x[0])[1]
        choices = []
        for factor, rate, power in itertools.product((0.8, 1.0, 1.2), (0.25, 0.5), (0.3, 0.7, 1.0)):
            spec = dict(
                kind="traffic_linear",
                alphas=campaign,
                alpha=specs["linear"]["alpha"],
                factor=factor,
                rate=rate,
                power=power,
            )
            choices.append((score_spec("traffic_linear", spec), spec))
        specs["traffic_linear"] = max(choices, key=lambda x: x[0])[1]
    specs["unscaled"] = dict(kind="linear", alpha=1.0)
    score_spec("unscaled", specs["unscaled"])
    return specs, rows


def folds_for(cfg):
    development = cfg["development_dates"]
    if development != sorted(set(development)) or cfg["test_dates"] != sorted(
        set(cfg["test_dates"])
    ):
        raise ValueError("Dates must be unique and chronological")
    if not cfg["cv_validation_dates"] or not set(cfg["cv_validation_dates"]).issubset(development):
        raise ValueError("Validation dates must belong to development, never the reserved test")
    folds = [
        (tuple(d for d in development if d < date), (date,)) for date in cfg["cv_validation_dates"]
    ]
    if any(not train or max(train) >= min(val) for train, val in folds):
        raise ValueError("Invalid forward folds")
    if max(development) >= min(cfg["test_dates"]):
        raise ValueError("Test dates overlap development")
    if (
        cfg["domain_randomization"]["budget"][0] <= 0.15
        or cfg["domain_randomization"]["budget"][1] >= 0.6
    ):
        raise ValueError("Keep declared stress budget endpoints unseen in training controls")
    if cfg["domain_randomization"]["competition"][1] >= 1.5:
        raise ValueError("Keep competition x1.5 out of training controls")
    return folds


def select_iterations(cv, cfg, state):
    selected = {}
    penalty = float(cfg.get("selection_penalty", 0.0))
    reference = {}
    if penalty:
        search = pd.DataFrame(state["baseline_search"])
        name = "mpc"
        if cfg.get("selection_reference") == "best_baseline":
            scores = {}
            for candidate, spec in state["baselines"].items():
                if candidate == "privileged_linear":
                    continue
                key = json.dumps(spec, sort_keys=True)
                scores[candidate] = search[
                    (search.policy == candidate) & (search.spec == key)
                ].score.mean()
            name = max(scores, key=scores.get)
        key = json.dumps(state["baselines"][name], sort_keys=True)
        reference = (
            search[(search.policy == name) & (search.spec == key)].set_index("fold").score.to_dict()
        )
    for variant in cfg["variants"]:
        candidates = []
        for episode, group in cv[(cv.variant == variant) & (cv.seed == -1)].groupby("episode"):
            score = group.score.mean()
            delta = np.asarray([r.score - reference.get(r.fold, 0.0) for r in group.itertuples()])
            objective = float(delta.mean() - penalty * delta.std())
            candidates.append(dict(episode=int(episode), score=float(score), objective=objective))
        selected[variant] = max(candidates, key=lambda r: r["objective"])
    return selected


def final_variants(cfg):
    return list(cfg["variants"])


def prefix(fold, variant, seed, episode):
    return f"f{fold}__{variant}__s{seed}__e{episode}__"


def tensors_at(arrays, key):
    tensors = {k: arrays[key + k] for k in ["w0", "b0", "w1", "b1", "w2", "b2"]}
    if any(not np.isfinite(v).all() for v in tensors.values()):
        raise ValueError("Nonfinite checkpoint")
    return tensors


def train(cfg, out, resume=False):
    if not Path(cfg["data_file"]).is_file():
        raise FileNotFoundError(
            f"Missing replay input: {Path(cfg['data_file']).resolve()}. The source ZIP excludes data. Download with python -m bidpace.fetch and prepare with python -m bidpace.replay, or copy the provided processed replay input to this path."
        )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "protocol.json"
    folds = folds_for(cfg)
    frozen = dict(
        config=cfg,
        source_sha256=source_hashes(),
        data_sha256=hashlib.sha256(Path(cfg["data_file"]).read_bytes()).hexdigest(),
        protocol="Forward-chaining CV; fixed-step refit on all development dates; five-seed ensembles; reserved-date evaluation",
        primary_objective="mean recorded binary clicks over chronological validation dates and declared conditions",
        data_scope=cfg["data_scope"],
    )
    if path.exists():
        state = json.loads(path.read_text())
        if not resume:
            raise FileExistsError("Output exists; choose a new --out or explicitly use --resume")
        if state["frozen"] != frozen:
            raise ValueError("Cannot resume with changed code, configuration or data")
    else:
        state = dict(frozen=frozen, completed_folds=[], phase="training")
        dump(path, state)
    started = time.monotonic()
    if "baselines" not in state:
        data_folds = [ReplayData(cfg["data_file"], tr, val) for tr, val in folds]
        specs, search = tune_baselines(data_folds, cfg)
        state.update(baselines=specs, baseline_search=search)
        dump(path, state)
        del data_folds
    arrays = {}
    if (out / "cv_models.npz").exists():
        with np.load(out / "cv_models.npz", allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
    records = state.get("cv_records", [])
    training = state.get("training_records", [])
    for fold, (tr, val) in enumerate(folds):
        if fold in state["completed_folds"]:
            continue
        data = ReplayData(cfg["data_file"], tr, val)
        days = validation_days(data, cfg)
        tasks = [
            (cfg, variant, seed, tr, val, None)
            for variant in cfg["variants"]
            for seed in cfg["seeds"]
        ]
        with ProcessPoolExecutor(max_workers=min(cfg["workers"], len(tasks))) as pool:
            results = list(pool.map(fit_worker, tasks))
        by_variant = {}
        for variant, seed, result in results:
            by_variant.setdefault(variant, {})[seed] = result
            for row in result["logs"]:
                training.append(dict(phase="cv", fold=fold, variant=variant, seed=seed, **row))
            for episode, tensors in result["snapshots"].items():
                arrays.update(
                    {prefix(fold, variant, seed, episode) + k: v for k, v in tensors.items()}
                )
                score = value(create_policy(tensors, result["spec"]), days)
                records.append(
                    dict(fold=fold, variant=variant, seed=seed, episode=episode, score=score)
                )
        for variant, results in by_variant.items():
            episodes = sorted(next(iter(results.values()))["snapshots"])
            for episode in episodes:
                policies = [
                    create_policy(results[s]["snapshots"][episode], results[s]["spec"])
                    for s in cfg["seeds"]
                ]
                score = value(Ensemble(policies, results[cfg["seeds"][0]]["spec"]["family"]), days)
                records.append(
                    dict(fold=fold, variant=variant, seed=-1, episode=episode, score=score)
                )
        save_arrays(out / "cv_models.npz", arrays)
        state.update(cv_records=records, training_records=training)
        state["completed_folds"].append(fold)
        dump(path, state)
        pd.DataFrame(records).to_csv(out / "cv.csv", index=False)
        print("Completed forward fold", fold, "train", tr, "validation", val, flush=True)
    cv = pd.DataFrame(records)
    ensemble = cv[cv.seed == -1]
    selected = select_iterations(cv, cfg, state)
    primary = max(cfg["variants"], key=lambda v: selected[v]["objective"])
    state["selection"] = dict(
        primary=primary,
        iterations=selected,
        aggregation="mean Q-values / mean latent actions across all configured seeds",
        criterion="mean paired validation gain minus penalty times across-date SD",
        penalty=cfg.get("selection_penalty", 0.0),
    )
    dump(path, state)
    final_arrays = {}
    specs = {}
    final_records = []
    if (out / "models.npz").exists() and state.get("phase") in ("trained", "evaluated"):
        return
    variants = final_variants(cfg)
    tasks = []
    for variant in variants:
        stop = selected[variant]["episode"]
        tasks.extend(
            (cfg, variant, seed, tuple(cfg["development_dates"]), (), stop) for seed in cfg["seeds"]
        )
    with ProcessPoolExecutor(max_workers=min(cfg["workers"], len(tasks))) as pool:
        for variant, seed, result in pool.map(fit_worker, tasks):
            stop = selected[variant]["episode"]
            tensors = result["snapshots"][stop]
            final_arrays.update({f"{variant}__s{seed}__" + k: v for k, v in tensors.items()})
            specs[variant] = result["spec"]
            final_records.append(
                dict(
                    variant=variant,
                    seed=seed,
                    episode=stop,
                    steps=result["steps"],
                    updates=result["updates"],
                    seconds=result["seconds"],
                )
            )
            training.extend(
                dict(phase="refit", fold=-1, variant=variant, seed=seed, **r)
                for r in result["logs"]
            )
    expected = len(variants) * len(cfg["seeds"]) * 6
    if len(final_arrays) != expected:
        raise AssertionError("Missing final model entries")
    save_arrays(out / "models.npz", final_arrays)
    state.update(
        policy_specs=specs,
        final_records=final_records,
        training_records=training,
        phase="trained",
        training_seconds=time.monotonic() - started,
    )
    dump(path, state)
    pd.DataFrame(training).to_csv(out / "training.csv", index=False)


def load_state(cfg, out):
    out = Path(out)
    if not (out / "protocol.json").is_file():
        raise FileNotFoundError(
            "No saved replay experiment. Run python run.py all successfully first, with the same --config and --out."
        )
    state = json.loads((out / "protocol.json").read_text())
    frozen = state["frozen"]
    if cfg != frozen["config"] or source_hashes() != frozen["source_sha256"]:
        raise ValueError("Code/config differs from frozen protocol")
    if hashlib.sha256(Path(cfg["data_file"]).read_bytes()).hexdigest() != frozen["data_sha256"]:
        raise ValueError("Replay data changed")
    return state


def final_policies(cfg, state, out):
    with np.load(Path(out) / "models.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    policies = []
    for variant, spec in state["policy_specs"].items():
        members = []
        for seed in cfg["seeds"]:
            p = create_policy(tensors_at(arrays, f"{variant}__s{seed}__"), spec)
            members.append(p)
            policies.append((variant, seed, p))
        policies.append(
            (
                variant,
                -1,
                Ensemble(
                    [
                        create_policy(tensors_at(arrays, f"{variant}__s{s}__"), spec)
                        for s in cfg["seeds"]
                    ],
                    spec["family"],
                ),
            )
        )
    return policies


def upper_bound(day):
    prices = np.concatenate([h.prices for h in day.hours])
    clicks = np.concatenate([h.clicks for h in day.hours])
    costs = np.sort(prices[clicks > 0])
    cumulative = np.cumsum(costs)
    k = int(np.searchsorted(cumulative, day.budget, side="right"))
    total = float(k)
    if k < len(costs) and costs[k] > 0:
        total += max(0.0, day.budget - (cumulative[k - 1] if k else 0.0)) / costs[k]
    return total


def evaluate(cfg, out):
    out = Path(out)
    state = load_state(cfg, out)
    if state.get("phase") not in ("trained", "evaluated"):
        raise ValueError("Training/refit incomplete")
    data = ReplayData(cfg["data_file"], cfg["development_dates"], (), cfg["test_dates"])
    market = market_tensors(data)
    policies = final_policies(cfg, state, out)
    policies.extend((name, -1, Baseline(spec, market)) for name, spec in state["baselines"].items())
    rows = []
    headroom = []
    for scenario, kw in SCENARIOS.items():
        for i, template in enumerate(data.templates["test"]):
            settings = dict(budget_fraction=cfg["evaluation_budget_fraction"])
            settings.update(kw)
            day = make_replay_day(template, seed=3000000 + i * 10000, **settings)
            bound = upper_bound(day)
            for name, seed, policy in policies:
                result = rollout_replay(day, policy)
                rows.append(
                    dict(
                        scenario=scenario,
                        date=day.date,
                        advertiser=day.advertiser,
                        policy=name,
                        seed=seed,
                        privileged=name == "privileged_linear",
                        upper_bound=bound,
                        **result,
                    )
                )
            if scenario == "standard":
                alphas = np.unique(
                    np.r_[
                        np.geomspace(0.04, 5.0, 60),
                        [1.0],
                        state["baselines"]["linear"]["alpha"],
                        *state["baselines"]["campaign_linear"]["alphas"].values(),
                    ]
                )
                candidates = [
                    (
                        rollout_replay(day, Baseline(dict(kind="linear", alpha=float(a))))[
                            "clicks"
                        ],
                        float(a),
                    )
                    for a in alphas
                ]
                best = max(candidates)
                headroom.append(
                    dict(
                        date=day.date,
                        advertiser=day.advertiser,
                        best_fixed_alpha=best[1],
                        hindsight_clicks=best[0],
                        upper_bound=bound,
                    )
                )
        print("Evaluated reserved dates:", scenario, flush=True)
    # Traces remain structured in JSON; numeric metrics have one row per real day.
    dump(
        out / "traces.json",
        [
            {k: r[k] for k in ["scenario", "date", "advertiser", "policy", "seed", "trace"]}
            for r in rows
            if r["scenario"] == "standard"
        ],
    )
    pd.DataFrame([{k: v for k, v in r.items() if k != "trace"} for r in rows]).to_csv(
        out / "evaluation.csv", index=False
    )
    pd.DataFrame(headroom).to_csv(out / "headroom.csv", index=False)
    state.update(
        phase="evaluated",
        test_audit=data.audit(),
        evaluation_rows=len(rows),
        model_sha256=hashlib.sha256((out / "models.npz").read_bytes()).hexdigest(),
        cv_model_sha256=hashlib.sha256((out / "cv_models.npz").read_bytes()).hexdigest(),
    )
    dump(out / "protocol.json", state)


def analyze(cfg, out):
    out = Path(out)
    state = load_state(cfg, out)
    raw = pd.read_csv(out / "evaluation.csv")
    ensemble = raw[raw.seed == -1]
    summary = (
        ensemble.groupby(["scenario", "policy", "privileged"])
        .agg(
            clicks=("clicks", "mean"),
            spent=("spent", "mean"),
            utilization=("utilization", "mean"),
            violations=("violations", "sum"),
        )
        .reset_index()
    )
    seed_summary = (
        raw[raw.seed != -1]
        .groupby(["scenario", "policy", "seed"])
        .agg(clicks=("clicks", "mean"), utilization=("utilization", "mean"))
        .reset_index()
    )
    seed_summary.to_csv(out / "seed_summary.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    primary = state["selection"]["primary"]
    # Reference is selected on CV; privileged baselines cannot be the deployment reference.
    search = pd.DataFrame(state["baseline_search"])
    allowed = [n for n in state["baselines"] if n != "privileged_linear"]
    scores = {
        n: float(
            search[
                (search.policy == n)
                & (search.spec == json.dumps(state["baselines"][n], sort_keys=True))
            ].score.mean()
        )
        for n in allowed
    }
    reference = max(scores, key=scores.get)
    comparisons = []
    rng = np.random.default_rng(48531)
    for scenario, g in ensemble.groupby("scenario"):
        p = g[g.policy == primary].set_index(["date", "advertiser"]).clicks
        for name in g.policy.unique():
            if name == primary:
                continue
            b = g[g.policy == name].set_index(["date", "advertiser"]).clicks.reindex(p.index)
            diff = p - b
            clusters = diff.groupby(level="date").agg(["sum", "count"])
            ids = rng.integers(len(clusters), size=(5000, len(clusters)))
            draws = clusters["sum"].to_numpy()[ids].sum(1) / clusters["count"].to_numpy()[ids].sum(
                1
            )
            lo, hi = np.quantile(draws, [0.025, 0.975])
            comparisons.append(
                dict(
                    scenario=scenario,
                    primary=primary,
                    comparator=name,
                    difference=float(diff.mean()),
                    relative_percent=100 * float((p.mean() - b.mean()) / max(b.mean(), 1e-8)),
                    ci_low=float(lo),
                    ci_high=float(hi),
                    date_clusters=len(clusters),
                )
            )
    dump(
        out / "analysis.json",
        dict(
            primary=primary,
            reference=reference,
            comparisons=comparisons,
            limitations="Conditional logged-winning-inventory replay. Only three reserved historical dates. Date-cluster bootstrap for deterministic ensembles; five member seeds are not independent test dates. Synthetic stress transformations are not new real data.",
        ),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 180}
    )
    names = (
        [n for n in state["baselines"] if n != "privileged_linear"]
        + list(state["policy_specs"])
        + ["privileged_linear"]
    )
    standard = summary[summary.scenario == "standard"].set_index("policy")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        names,
        [standard.loc[n, "clicks"] for n in names],
        color=[
            "#2e6589" if n == primary else "#c49a59" if n.startswith("privileged") else "#8b939c"
            for n in names
        ],
    )
    ax.set_xlabel("Recorded binary clicks per advertiser-day; five-seed neural ensembles")
    fig.tight_layout()
    fig.savefig(out / "comparison.png")
    plt.close(fig)
    cv = pd.DataFrame(state["cv_records"])
    tr = pd.DataFrame(state["training_records"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for variant in cfg["variants"]:
        curve = (
            cv[(cv.variant == variant) & (cv.seed == -1)]
            .groupby("episode")
            .score.agg(["mean", "std"])
        )
        axes[0].plot(curve.index, curve["mean"], label=variant)
        axes[0].fill_between(
            curve.index, curve["mean"] - curve["std"], curve["mean"] + curve["std"], alpha=0.12
        )
        curve = tr[(tr.variant == variant) & (tr.phase == "refit")].groupby("episode").clicks.mean()
        axes[1].plot(curve.index, curve, label=variant)
    axes[0].set(xlabel="Training episode", ylabel="Forward-CV recorded clicks")
    axes[1].set(xlabel="Refit episode", ylabel="Training clicks, batch mean")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "learning.png")
    plt.close(fig)
    shown = list(dict.fromkeys([primary, reference, "pid", "quality_pid"]))
    scenarios = list(SCENARIOS)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(scenarios))
    w = 0.8 / len(shown)
    for j, name in enumerate(shown):
        vals = summary[summary.policy == name].set_index("scenario").loc[scenarios, "clicks"]
        ax.bar(x + (j - (len(shown) - 1) / 2) * w, vals, width=w, label=name)
    ax.set_xticks(
        x,
        [
            "Standard",
            "Prices x1.5",
            "Evening x3",
            "Budget .15",
            "Budget .60",
            "Clicks x.5",
            "Coverage .5",
        ],
    )
    ax.set_ylabel("Binary clicks / advertiser-day")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "stress.png")
    plt.close(fig)
    traces = json.loads((out / "traces.json").read_text())
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for name in list(dict.fromkeys([primary, reference, "linear", "pid"])):
        curves = [r["trace"] for r in traces if r["policy"] == name and r["seed"] == -1]
        ax.plot(range(25), np.mean(curves, axis=0), label=name)
    ax.set(xlabel="Hour", ylabel="Mean fraction of budget spent", ylim=(0, 1.04))
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "spending.png")
    plt.close(fig)
    print(summary.to_string(index=False))
    print("Validation-selected primary:", primary, "reference:", reference)


def verify(cfg, out):
    out = Path(out)
    state = load_state(cfg, out)
    if state.get("phase") != "evaluated":
        raise FileNotFoundError(
            "Evaluation incomplete. Run python run.py all successfully before verify."
        )
    assert hashlib.sha256((out / "models.npz").read_bytes()).hexdigest() == state["model_sha256"]
    assert (
        hashlib.sha256((out / "cv_models.npz").read_bytes()).hexdigest() == state["cv_model_sha256"]
    )
    raw = pd.read_csv(out / "evaluation.csv")
    summary = pd.read_csv(out / "summary.csv")
    assert not raw.duplicated(["scenario", "date", "advertiser", "policy", "seed"]).any()
    assert set(raw.date) == set(cfg["test_dates"])
    assert np.isfinite(raw.select_dtypes("number")).all().all()
    assert (raw.spent <= raw.budget + 1e-7).all() and raw.violations.sum() == 0
    assert (raw.clicks <= raw.upper_bound + 1e-7).all()
    test_data = ReplayData(cfg["data_file"], cfg["development_dates"], (), cfg["test_dates"])
    days = {(t.date, t.advertiser) for t in test_data.templates["test"]}
    keys = {(name, -1) for name in state["baselines"]}
    keys.update((name, seed) for name in final_variants(cfg) for seed in cfg["seeds"] + [-1])
    expected = {
        (case, date, adv, name, seed)
        for case in SCENARIOS
        for date, adv in days
        for name, seed in keys
    }
    actual = set(
        raw[["scenario", "date", "advertiser", "policy", "seed"]].itertuples(index=False, name=None)
    )
    assert actual == expected
    cv = pd.DataFrame(state["cv_records"])
    selected = state["selection"]["iterations"]
    reproduced = select_iterations(cv, cfg, state)
    assert selected == reproduced
    assert state["selection"]["primary"] == max(
        cfg["variants"], key=lambda v: reproduced[v]["objective"]
    )
    with np.load(out / "cv_models.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    checked = 0
    for fold, (train_dates, validation_dates) in enumerate(folds_for(cfg)):
        data = ReplayData(cfg["data_file"], train_dates, validation_dates)
        days = validation_days(data, cfg)
        for variant in cfg["variants"]:
            episode = selected[variant]["episode"]
            spec = policy_spec(data, variant)
            members = []
            for seed in cfg["seeds"]:
                policy = create_policy(
                    tensors_at(arrays, prefix(fold, variant, seed, episode)), spec
                )
                recorded = cv[
                    (cv.fold == fold)
                    & (cv.variant == variant)
                    & (cv.seed == seed)
                    & (cv.episode == episode)
                ].score.iloc[0]
                assert np.isclose(value(policy, days), recorded)
                members.append(policy)
                checked += 1
            recorded = cv[
                (cv.fold == fold)
                & (cv.variant == variant)
                & (cv.seed == -1)
                & (cv.episode == episode)
            ].score.iloc[0]
            assert np.isclose(value(Ensemble(members, spec["family"]), days), recorded)
    data = ReplayData(cfg["data_file"], cfg["development_dates"], (), cfg["test_dates"])
    policies = final_policies(cfg, state, out)
    # Independently reproduce every neural agent on every nominal real day.
    for template in data.templates["test"]:
        day = make_replay_day(template, budget_fraction=cfg["evaluation_budget_fraction"])
        for name, seed, policy in policies:
            actual = rollout_replay(day, policy)
            saved = raw[
                (raw.scenario == "standard")
                & (raw.date == day.date)
                & (raw.advertiser == day.advertiser)
                & (raw.policy == name)
                & (raw.seed == seed)
            ].iloc[0]
            assert actual["clicks"] == saved.clicks and np.isclose(actual["spent"], saved.spent)
    for _, row in summary.iterrows():
        g = raw[(raw.scenario == row.scenario) & (raw.policy == row.policy) & (raw.seed == -1)]
        assert np.isclose(row.clicks, g.clicks.mean())
    print(
        f"PASS: {len(raw):,} evaluation rows; {checked} selected fold agents; all final nominal neural rollouts; chronological cutoffs; data/source/model hashes; zero budget violations."
    )


def export(cfg, out, path):
    state = load_state(cfg, out)
    variant = state["selection"]["primary"]
    path = Path(path)
    if path.exists():
        raise FileExistsError("Agent path exists")
    with np.load(Path(out) / "models.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k.startswith(variant + "__")}
    arrays["metadata"] = np.asarray(
        json.dumps(
            dict(
                variant=variant,
                seeds=cfg["seeds"],
                spec=state["policy_specs"][variant],
                data_sha256=state["frozen"]["data_sha256"],
                development_dates=cfg["development_dates"],
                test_dates=cfg["test_dates"],
                budget_fraction=cfg["evaluation_budget_fraction"],
            )
        )
    )
    save_arrays(path, arrays)
    print("Exported validation-selected ensemble:", variant)


def demo(cfg, out, agent=None):
    if agent:
        with np.load(agent, allow_pickle=False) as z:
            metadata = json.loads(str(z["metadata"]))
            arrays = {k: z[k] for k in z.files if k != "metadata"}
        assert (
            hashlib.sha256(Path(cfg["data_file"]).read_bytes()).hexdigest()
            == metadata["data_sha256"]
        )
        data = ReplayData(
            cfg["data_file"], metadata["development_dates"], (), metadata["test_dates"]
        )
        variant = metadata["variant"]
        spec = metadata["spec"]
        policy = Ensemble(
            [
                create_policy(tensors_at(arrays, f"{variant}__s{s}__"), spec)
                for s in metadata["seeds"]
            ],
            spec["family"],
        )
    else:
        state = load_state(cfg, out)
        data = ReplayData(cfg["data_file"], cfg["development_dates"], (), cfg["test_dates"])
        policy = next(
            p
            for n, s, p in final_policies(cfg, state, out)
            if n == state["selection"]["primary"] and s == -1
        )
    day = make_replay_day(
        data.templates["test"][0], budget_fraction=cfg["evaluation_budget_fraction"]
    )
    print(json.dumps(rollout_replay(day, policy), indent=2))
