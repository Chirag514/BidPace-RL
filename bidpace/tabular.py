"""Training-hour sampled Bellman RL with exact ordered affordability transitions.

The empirical MDP samples a training day independently at each hour. It retains
within-hour arrival order and cost/reward dependence, but not cross-hour latent
day correlations. Budget interpolation and causal market rescaling are approximate.
"""

import numpy as np
from .adaptive import MarketFilter
from .control import fit_control

REWARDS = ("ctr", "shrunk", "clicks")
ACTIONS = np.unique(np.r_[0.0, np.geomspace(0.04, 3.0, 39),
                          [0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0]])
GRID = np.unique(np.r_[0.0, np.geomspace(1e-6, 0.002, 25),
                       np.linspace(0.002, 1.0, 500), np.geomspace(1.0, 4.0, 60)])


def ordered_outcomes(prices, rewards, thresholds, budgets):
    """Exact replay outcomes for every budget/action, with arbitrary reward columns.

    Prefix sums handle affordable prefixes. Cached positive-price arrival lists
    locate affordable suffix purchases by binary search; free suffix rewards are
    added separately. This avoids rescanning long tails for every budget state.
    Zero budget is terminal, matching ReplayEnv. A negative threshold means no bid.
    """
    prices = np.asarray(prices)
    rewards = np.asarray(rewards, dtype=float)
    if rewards.ndim == 1:
        rewards = rewards[:, None]
    budgets = np.asarray(budgets, dtype=float)
    spent = np.zeros((len(budgets), len(thresholds)))
    earned = np.zeros((*spent.shape, rewards.shape[1]))
    cheap = {}
    free = rewards * (prices == 0)[:, None]
    free_suffix = np.vstack([np.cumsum(free[::-1], axis=0)[::-1],
                             np.zeros(rewards.shape[1])])
    for action, threshold in enumerate(thresholds):
        if threshold < 0:
            continue
        ids = np.flatnonzero(prices <= threshold)
        if not len(ids):
            continue
        costs = np.r_[0.0, np.cumsum(prices[ids], dtype=float)]
        returns = np.vstack([np.zeros(rewards.shape[1]), np.cumsum(rewards[ids], axis=0)])
        counts = np.searchsorted(costs[1:], budgets, side="right")
        spent[:, action] = costs[counts]
        earned[:, action] = returns[counts]
        for b in np.flatnonzero((counts < len(ids)) & (budgets > 0)):
            remaining = budgets[b] - costs[counts[b]]
            start = int(ids[counts[b]]) + 1
            earned[b, action] += free_suffix[start]
            # Remaining is smaller than the unaffordable (bid-eligible) price,
            # so every subsequent affordable price also satisfies the bid.
            while remaining >= 1:
                limit = int(remaining)  # input replay prices are integral
                if limit not in cheap:
                    cheap[limit] = np.flatnonzero((prices > 0) & (prices <= limit))
                candidates = cheap[limit]
                k = int(np.searchsorted(candidates, start))
                if k == len(candidates):
                    break
                index = candidates[k]
                cost = float(prices[index])
                remaining -= cost
                spent[b, action] += cost
                earned[b, action] += rewards[index]
                start = int(index) + 1
    spent[budgets <= 0] = 0
    earned[budgets <= 0] = 0
    return spent, earned


def reward_rates(templates, profile):
    """Training-only hierarchical CTR estimates in fixed relative-price bands."""
    edges = np.r_[np.geomspace(0.05, 4.0, 20), np.inf]
    counts = np.zeros((24, len(edges)))
    clicked = np.zeros_like(counts)
    for template in templates:
        for h, hour in enumerate(template.hours):
            bins = np.searchsorted(edges, hour.prices / profile.price, side="left")
            counts[h] += np.bincount(bins, minlength=len(edges))
            clicked[h] += np.bincount(bins, weights=hour.clicks, minlength=len(edges))
    pooled = (clicked.sum(0) + 10000 * profile.mean_ctr) / (counts.sum(0) + 10000)
    prior = pooled[None, :] * (profile.ctr / profile.mean_ctr)[:, None]
    rates = (clicked + 10000 * prior) / (counts + 10000)
    return edges, np.clip(rates, 0, 1)


def fit(data, grid=GRID, actions=ACTIONS):
    """Finite-horizon fitted Q iteration, averaging *after* sampled Bellman backup."""
    campaigns = sorted(data.profiles)
    tables = np.zeros((len(REWARDS), len(campaigns), 24, len(grid), len(actions)),
                      dtype=np.float32)
    for j, adv in enumerate(campaigns):
        profile = data.profiles[adv]
        templates = [t for t in data.templates["train"] if t.advertiser == adv]
        edges, rates = reward_rates(templates, profile)
        future = np.zeros((len(REWARDS), len(grid)))
        for h in reversed(range(24)):
            q = np.zeros((len(REWARDS), len(grid), len(actions)))
            # Unobserved launch hours use training observed hours as a prior;
            # an empty hour on an otherwise observed date remains a zero sample.
            samples = [t.hours[h] for t in templates]
            if not any(len(hr.prices) for hr in samples):
                samples = [hr for t in templates for hr in t.hours if len(hr.prices)]
            thresholds = actions * profile.price * profile.ctr[h] / profile.mean_ctr
            thresholds = np.where(actions > 0, thresholds, -1)
            for hour in samples:
                bins = np.searchsorted(edges, hour.prices / profile.price, side="left")
                rewards = np.column_stack([np.full(len(bins), profile.ctr[h]),
                                           rates[h, bins], hour.clicks])
                spent, earned = ordered_outcomes(hour.prices, rewards, thresholds,
                                                 grid * profile.reference_cost)
                left = np.maximum(0, grid[:, None] - spent / profile.reference_cost)
                for r in range(len(REWARDS)):
                    continuation = np.interp(left.ravel(), grid, future[r]).reshape(left.shape)
                    q[r] += earned[:, :, r] + continuation
            q /= len(samples)
            q[:, 0] = 0
            tables[:, j, h] = q
            future = q.max(axis=2)
        print("Fitted sampled Bellman campaign", adv, flush=True)
    control = fit_control(data)
    spec = {k: v.tolist() for k, v in control.items()
            if k in ("campaigns", "market_quantiles", "market_prices")}
    return dict(q=tables, grid=np.asarray(grid), actions=np.asarray(actions)), spec


class TabularPolicy:
    requires_traffic = True

    def __init__(self, tensors, spec, reward="shrunk", reserve=1.0, rate=0.5):
        self.tensors = tensors
        self.spec = spec
        self.reward = REWARDS.index(reward)
        self.reserve = float(reserve)
        self.filter = MarketFilter(spec, rate=rate)

    def reset(self, day):
        self.profile = day.profile
        self.budget = day.budget
        self.j = self.spec["campaigns"].index(day.advertiser)
        self.filter.reset(day)
        self.filter.budget = day.budget / day.profile.price
        self.filter.auctions = day.profile.mean_daily_impressions

    def act(self, obs):
        self.filter.observe(obs)
        hour = min(int(round(float(obs[1]) * 24)), 23)
        traffic = max(float(obs[8]) * 4, 0.05)
        price = self.filter.price
        budget = (float(obs[0]) * self.budget / self.profile.reference_cost
                  / price / traffic / self.reserve)
        grid = self.tensors["grid"]
        curve = self.tensors["q"][self.reward, self.j, hour]
        hi = int(np.clip(np.searchsorted(grid, budget, side="right"), 1, len(grid) - 1))
        weight = float(np.clip((budget - grid[hi - 1]) / (grid[hi] - grid[hi - 1]), 0, 1))
        values = (1 - weight) * curve[hi - 1] + weight * curve[hi]
        # Evaluate only actions executable after causal market rescaling.
        feasible = self.tensors["actions"] * price <= 3.0 + 1e-8
        action = int(np.where(feasible, values, -np.inf).argmax())
        alpha = float(np.clip(self.tensors["actions"][action] * price, 0, 3))
        self.filter.record(obs, alpha)
        return alpha
