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
from .policies import Fixed, PID, BudgetLinear
from .adaptive import MarketFilter
from .replay import ReplayData, ReplayEnv, make_replay_day, rollout_replay
from .control import TrafficDQN, CampaignDQN, CostController, fit_control

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
    """Fingerprint current shared code; imported studies retain their own hashes."""
    return {
        p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path("bidpace").glob("*.py"))
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
    """Average seed Q-values before choosing a common bid multiplier."""

    def __init__(self, policies, family):
        if family != "dqn":
            raise ValueError(f"Unsupported ensemble family: {family}")
        self.policies, self.family = list(policies), family
        self.requires_traffic = any(getattr(p, "requires_traffic", False) for p in self.policies)

    def reset(self, day):
        for policy in self.policies:
            policy.reset(day)

    def act(self, obs):
        predictions = np.stack([p.model(p.encode(obs)) for p in self.policies])
        action = int(predictions.mean(0).argmax())
        values = [p.apply(action) for p in self.policies]
        if not np.allclose(values, values[0], atol=1e-10):
            raise AssertionError("Ensemble controller states diverged")
        return float(values[0])


def policy_spec(data, variant):
    """Describe one supported learner and its training-fitted controller."""
    if variant not in ("traffic_dqn", "campaign_dqn"):
        raise ValueError(f"Unsupported RL variant: {variant}")
    spec = dict(
        family="dqn",
        campaigns=sorted(data.profiles),
        variant=variant,
    )
    return spec


def blank_policy(spec):
    variant = spec["variant"]
    if variant == "traffic_dqn":
        return TrafficDQN(None)
    if variant == "campaign_dqn":
        return CampaignDQN(None, spec["campaigns"])
    raise ValueError(f"Unsupported RL variant: {variant}")


def create_policy(tensors, spec):
    """Reconstruct a DQN from inference tensors and its training campaign map."""
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
    """Fixed-step Double DQN; stopping is selected by the outer forward folds."""
    spec = policy_spec(data, variant)
    settings = cfg["dqn"]
    maximum = settings["episodes"]
    stop = maximum if stop is None else int(stop)
    if not 0 <= stop <= maximum:
        raise ValueError("Invalid refit stopping iteration")
    checkpoints = set(settings["checkpoints"]) | {stop}
    policy = blank_policy(spec)
    output_dim = len(policy.actions)
    model = MLP(seed, input_dim=policy.input_dim, output_dim=output_dim)
    target = MLP(seed + 1, input_dim=policy.input_dim, output_dim=output_dim)
    target.copy_from(model)
    replay = Replay(settings.get("replay_capacity", 80000), input_dim=policy.input_dim)
    policy.model = model
    snapshots = {0: {k: v.copy() for k, v in model.p.items()}}
    # Keep campaign sampling, exploration and coverage on independent seeded streams.
    curriculum = np.random.default_rng(seed + 19000)
    rng = np.random.default_rng(seed + 49000)
    exposure = np.random.default_rng(seed + 89000)
    logs = []
    recent = []
    losses = []
    steps = updates = 0
    started = time.monotonic()
    for episode in range(1, stop + 1):
        adv = spec["campaigns"][int(curriculum.integers(len(spec["campaigns"])))]
        candidates = [t for t in data.templates["train"] if t.advertiser == adv]
        template = candidates[int(curriculum.integers(len(candidates)))]
        budget, competition = training_controls(cfg, variant, curriculum)
        coverage = float(exposure.uniform(*cfg["domain_randomization"]["coverage"]))
        day = make_replay_day(
            template, budget_fraction=budget, competition=competition,
            coverage=coverage, seed=int(exposure.integers(2**31)),
        )
        env = ReplayEnv(day, features=True)
        policy.reset(day)
        obs = policy.encode(env.obs())
        queue = deque()
        exploration = max(
            settings["epsilon_final"],
            1 - (1 - settings["epsilon_final"]) * episode / (maximum * 0.75),
        )
        while not env.done:
            index = (int(rng.integers(output_dim)) if rng.random() < exploration
                     else int(model(obs).argmax()))
            action = policy.apply(index)
            raw_next, reward, done = env.step(action)
            nxt = policy.encode(raw_next)
            # One training-fitted scale preserves reward comparisons across campaigns.
            normalized = reward / data.reward_scale
            steps += 1
            queue.append((obs.copy(), index, normalized, nxt.copy(), done))
            emit_nstep(queue, replay, settings["nstep"], cfg["gamma"], flush=done)
            if replay.size >= settings["warmup"] and steps % settings["update_every"] == 0:
                x, acts, r, ns, discount = replay.sample(rng, settings["batch_size"])
                targets = bellman(model, target, r, ns, discount, double=True)
                losses.append(model.update(x, acts, targets, settings["learning_rate"]))
                updates += 1
                if updates % settings["target_every_updates"] == 0:
                    target.copy_from(model)
            obs = nxt
        recent.append(env.clicks)
        if episode % 100 == 0 or episode == stop:
            logs.append(dict(
                episode=episode, clicks=float(np.mean(recent)),
                loss=float(np.mean(losses)) if losses else 0.0, exploration=exploration,
            ))
            recent.clear()
            losses.clear()
        if episode in checkpoints:
            snapshots[episode] = {k: v.copy() for k, v in model.p.items()}
            print(f"{variant} seed={seed} trained={episode}", flush=True)
    # Only inference tensors are exported; an interrupted seed restarts from its seed.
    return dict(snapshots=snapshots, spec=spec, logs=logs, steps=steps,
                updates=updates, seconds=time.monotonic() - started)


def fit_worker(task):
    cfg, variant, seed, train_dates, validation_dates, stop = task
    data = ReplayData(cfg["data_file"], train_dates, validation_dates)
    return variant, seed, fit_rl(data, cfg, variant, seed, stop)


def cached_fit_worker(job):
    """Save a completed seed before returning it to parent-side validation.

    An interrupted seed restarts, but completed seeds survive a failed fold.
    The cache key includes the frozen code/config/data and the complete task.
    """
    task, directory, frozen = job
    identity = hashlib.sha256(json.dumps([frozen, task], sort_keys=True).encode()).hexdigest()
    path = Path(directory) / (identity + ".npz")
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            metadata = json.loads(str(z["metadata"]))
            if metadata["identity"] != identity:
                raise ValueError("Worker cache identity mismatch")
            result = metadata["result"]
            result["snapshots"] = {
                int(episode): {k.split("__", 1)[1]: z[k].copy()
                               for k in z.files if k.startswith(f"e{episode}__")}
                for episode in metadata["episodes"]
            }
        print(f"Reused completed seed: {task[1]} seed={task[2]}", flush=True)
        return task[1], task[2], result
    variant, seed, result = fit_worker(task)
    arrays = {}
    for episode, tensors in result["snapshots"].items():
        arrays.update({f"e{episode}__{k}": v
                       for k, v in tensors.items()})
    arrays["metadata"] = np.asarray(json.dumps(dict(
        identity=identity, episodes=list(result["snapshots"]),
        result={k: v for k, v in result.items() if k != "snapshots"},
    )))
    save_arrays(path, arrays)
    return variant, seed, result


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


def tune_baselines(folds, cfg, nominal_only=False):
    """Use exactly the same chronological validation conditions as the agents."""
    grid = np.unique(
        np.r_[np.geomspace(0.08, 3.0, 20), [0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0]]
    )
    rows = []
    grid_scores = []
    markets = [market_tensors(data) for data in folds]
    all_days = [validation_days(data, cfg) for data in folds]
    nominal_cases = [i for i, case in enumerate(cfg["validation_conditions"])
                     if case == {"budget_fraction": cfg["evaluation_budget_fraction"]}]
    if nominal_only and len(nominal_cases) != 1:
        raise ValueError("Nominal baseline tuning requires exactly one nominal condition")

    def score_spec(name, spec):
        scores = []
        for fold, (data, days, market) in enumerate(zip(folds, all_days, markets)):
            if nominal_only:
                days = [row for row in days if row[2] in nominal_cases]
            score = value(Baseline(spec, market), days)
            scores.append(score)
            rows.append(
                dict(policy=name, fold=fold, score=score, spec=json.dumps(spec, sort_keys=True))
            )
        return float(np.mean(scores))

    for kind in ["constant", "linear", "privileged_linear"]:
        for alpha in grid:
            spec = dict(kind=kind, alpha=float(alpha))
            score_spec(kind, spec)
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
        if nominal_only:
            g = g[g.case.isin(nominal_cases)]
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
            and (not nominal_only or c == {"budget_fraction": fraction})
        ]
        for adv, g in scores[scores.case.isin(cases)].groupby("advertiser"):
            anchors[str(adv)].append(float(g.groupby("alpha").clicks.mean().idxmax()))
    fallback = []
    for fraction in fractions:
        cases = [
            i
            for i, c in enumerate(cfg["validation_conditions"])
            if c["budget_fraction"] == fraction
            and (not nominal_only or c == {"budget_fraction": fraction})
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


def prefix(fold, variant, seed, episode):
    return f"f{fold}__{variant}__s{seed}__e{episode}__"


def tensors_at(arrays, key):
    tensors = {k[len(key):]: v for k, v in arrays.items() if k.startswith(key)}
    if not tensors:
        raise ValueError(f"Missing checkpoint: {key}")
    if any(not np.isfinite(v).all() for v in tensors.values()):
        raise ValueError("Nonfinite checkpoint")
    return tensors


def train(cfg, out, resume=False):
    """Tune on chronological folds, then refit each seed at its selected iteration."""
    if not Path(cfg['data_file']).is_file():
        raise FileNotFoundError(
            f"Missing replay input: {Path(cfg['data_file']).resolve()}. "
            "Prepare data with python -m bidpace.benchmark prepare."
        )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'protocol.json'
    folds = folds_for(cfg)
    frozen = dict(
        config=cfg, source_sha256=source_hashes(),
        data_sha256=hashlib.sha256(Path(cfg["data_file"]).read_bytes()).hexdigest(),
        protocol="Forward-chaining CV; fixed-step refit; configured-seed ensembles",
        primary_objective="mean recorded clicks over validation dates and declared conditions",
        data_scope=cfg["data_scope"],
    )
    if path.exists():
        state = json.loads(path.read_text())
        if not resume:
            raise FileExistsError('Output exists; choose a new --out or explicitly use --resume')
        if state['frozen'] != frozen:
            raise ValueError('Cannot resume with changed code, configuration or data')
    else:
        state = dict(frozen=frozen, completed_folds=[], phase='training')
        dump(path, state)
    started = time.monotonic()
    if 'baselines' not in state:
        data_folds = [ReplayData(cfg['data_file'], tr, val) for tr, val in folds]
        specs, search = tune_baselines(data_folds, cfg)
        state.update(baselines=specs, baseline_search=search)
        dump(path, state)
        del data_folds
    arrays = {}
    if (out / 'cv_models.npz').exists():
        with np.load(out / 'cv_models.npz', allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
    records = state.get('cv_records', [])
    training = state.get('training_records', [])
    for fold, (tr, val) in enumerate(folds):
        if fold in state['completed_folds']:
            continue
        data = ReplayData(cfg['data_file'], tr, val)
        days = validation_days(data, cfg)
        tasks = [(cfg, variant, seed, tr, val, None) for variant in cfg['variants'] for seed in cfg['seeds']]
        with ProcessPoolExecutor(max_workers=min(cfg['workers'], len(tasks))) as pool:
            results = list(pool.map(cached_fit_worker, [(t, out / 'worker_cache', frozen) for t in tasks]))
        by_variant = {}
        for variant, seed, result in results:
            by_variant.setdefault(variant, {})[seed] = result
            for row in result['logs']:
                training.append(dict(phase='cv', fold=fold, variant=variant, seed=seed, **row))
            for episode, tensors in result['snapshots'].items():
                arrays.update({prefix(fold, variant, seed, episode) + k: v for k, v in tensors.items()})
                score = value(create_policy(tensors, result['spec']), days)
                records.append(dict(fold=fold, variant=variant, seed=seed, episode=episode, score=score))
        for variant, results in by_variant.items():
            episodes = sorted(next(iter(results.values()))['snapshots'])
            for episode in episodes:
                policies = [create_policy(results[s]['snapshots'][episode], results[s]['spec']) for s in cfg['seeds']]
                score = value(Ensemble(policies, results[cfg['seeds'][0]]['spec']['family']), days)
                records.append(dict(fold=fold, variant=variant, seed=-1, episode=episode, score=score))
        save_arrays(out / 'cv_models.npz', arrays)
        state.update(cv_records=records, training_records=training)
        state['completed_folds'].append(fold)
        dump(path, state)
        pd.DataFrame(records).to_csv(out / 'cv.csv', index=False)
        print('Completed forward fold', fold, 'train', tr, 'validation', val, flush=True)
    cv = pd.DataFrame(records)
    selected = select_iterations(cv, cfg, state)
    primary = max(cfg['variants'], key=lambda v: selected[v]['objective'])
    state["selection"] = dict(
        primary=primary, iterations=selected,
        aggregation="mean Q-values across all configured seeds",
        criterion="mean paired validation gain minus penalty times across-date SD",
        penalty=cfg.get("selection_penalty", 0.0),
    )
    dump(path, state)
    final_arrays = {}
    specs = {}
    final_records = []
    if (out / 'models.npz').exists() and state.get('phase') in ('trained', 'evaluated'):
        return
    variants = list(cfg['variants'])
    tasks = []
    for variant in variants:
        stop = selected[variant]['episode']
        tasks.extend(((cfg, variant, seed, tuple(cfg['development_dates']), (), stop) for seed in cfg['seeds']))
    with ProcessPoolExecutor(max_workers=min(cfg['workers'], len(tasks))) as pool:
        for variant, seed, result in pool.map(cached_fit_worker, [(t, out / 'worker_cache', frozen) for t in tasks]):
            stop = selected[variant]['episode']
            tensors = result['snapshots'][stop]
            final_arrays.update({f'{variant}__s{seed}__' + k: v for k, v in tensors.items()})
            specs[variant] = result['spec']
            final_records.append(dict(
                variant=variant, seed=seed, episode=stop,
                steps=result["steps"], updates=result["updates"], seconds=result["seconds"],
            ))
            training.extend((dict(phase='refit', fold=-1, variant=variant, seed=seed, **r) for r in result['logs']))
    expected = {(variant, seed) for variant in variants for seed in cfg['seeds']}
    if {(r['variant'], r['seed']) for r in final_records} != expected:
        raise AssertionError('Missing final model entries')
    for variant, seed in expected:
        create_policy(tensors_at(final_arrays, f'{variant}__s{seed}__'), specs[variant])
    save_arrays(out / 'models.npz', final_arrays)
    state.update(
        policy_specs=specs, final_records=final_records, training_records=training,
        phase="trained", training_seconds=time.monotonic() - started,
        model_sha256=hashlib.sha256((out / "models.npz").read_bytes()).hexdigest(),
        cv_model_sha256=hashlib.sha256((out / "cv_models.npz").read_bytes()).hexdigest(),
    )
    dump(path, state)
    pd.DataFrame(training).to_csv(out / 'training.csv', index=False)
