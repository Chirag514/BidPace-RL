"""Training-only cost curves and causal traffic-aware bidding controllers."""

import numpy as np
from .adaptive import MarketFilter
from .agents import DQNPolicy
from .ppo import PPOPolicy


def fit_control(data, quantiles=101):
    campaigns = sorted(data.profiles)
    grid = np.r_[0.0, np.geomspace(0.02, 3.0, 79)]
    levels = np.linspace(0, 1, quantiles)
    prices = np.empty((len(campaigns), 24, quantiles))
    cost = np.zeros((len(campaigns), 24, len(grid)))
    for j, adv in enumerate(campaigns):
        templates = [t for t in data.templates["train"] if t.advertiser == adv]
        profile = data.profiles[adv]
        fallback = np.concatenate([h.prices for t in templates for h in t.hours])
        for hour in range(24):
            sample = np.sort(np.concatenate([t.hours[hour].prices for t in templates]))
            prices[j, hour] = (
                np.quantile(sample if len(sample) else fallback, levels) / profile.price
            )
            cumulative = np.r_[0.0, np.cumsum(sample, dtype=np.float64)]
            thresholds = grid * profile.price * profile.ctr[hour] / profile.mean_ctr
            count = np.searchsorted(sample, thresholds, side="right")
            cost[j, hour] = cumulative[count] / len(templates) / profile.reference_cost
            cost[j, hour, 0] = 0.0
    return dict(
        campaigns=np.asarray(campaigns),
        market_quantiles=levels,
        market_prices=prices,
        cost_grid=grid,
        cost_curve=cost,
    )


class CostController:
    """Replan a price threshold from training curves and completed-hour feedback.

    The controller sees the same causal traffic observations as the learners.
    It never reads the day's future inventory, prices, responses or multiplier.
    """

    def __init__(self, tensors, rate=0.5, reserve=1.0):
        self.tensors = {k: np.asarray(v) for k, v in tensors.items()}
        self.filter = MarketFilter(self.tensors, rate=rate)
        self.reserve = float(reserve)

    def reset(self, day):
        self.profile = day.profile
        self.budget = day.budget
        self.filter.reset(day)
        self.filter.budget = day.budget / day.profile.price
        self.filter.auctions = day.profile.mean_daily_impressions
        self.j = self.tensors["campaigns"].tolist().index(day.advertiser)
        self.hour = -1

    def forecast(self, obs):
        if len(obs) < 12:
            raise ValueError("Cost controller requires causal traffic features")
        hour = min(int(round(float(obs[1]) * 24)), 23)
        if hour != self.hour:
            self.filter.observe(obs)
            self.hour = hour
        price = self.filter.price
        traffic = max(float(obs[8]) * 4.0, 0.05)
        grid = self.tensors["cost_grid"]
        remaining = sum(
            np.interp(grid / price, grid, curve)
            for curve in self.tensors["cost_curve"][self.j, hour:]
        )
        expected = remaining * price * traffic
        target = float(obs[0]) * self.budget / self.profile.reference_cost / self.reserve
        return float(np.clip(np.interp(target, expected, grid), 0.0, 3.0))

    def record(self, obs, alpha):
        self.filter.record(obs, float(alpha))

    def act(self, obs):
        alpha = self.forecast(obs)
        self.record(obs, alpha)
        return alpha


class TrafficDQN(DQNPolicy):
    requires_traffic = True

    def __init__(self, model):
        super().__init__(model)
        self.input_dim += 5


class TrafficPPO(PPOPolicy):
    requires_traffic = True

    def __init__(self, model):
        super().__init__(model, history=True, relative=True)
        self.input_dim += 5


class ResidualPPO(PPOPolicy):
    """Learn bounded corrections around a training-calibrated cost controller."""

    requires_traffic = True

    def __init__(self, model, spec):
        super().__init__(model, history=True, relative=False)
        self.input_dim += 6  # Five public features and the current controller bid.
        self.controller = CostController(spec["controller"])
        self.residual_scale = float(spec.get("residual_scale", 0.5))

    def reset(self, day):
        super().reset(day)
        self.controller.reset(day)

    def encode(self, obs):
        self.base = self.controller.forecast(obs)
        self.current_obs = np.asarray(obs).copy()
        return np.r_[super().encode(obs), self.base / 3.0].astype(np.float32)

    def apply(self, latent):
        self.alpha = float(
            np.clip(self.base * np.exp(self.residual_scale * np.tanh(latent)), 0.0, 3.0)
        )
        # The market filter must record the executed correction, not its proposal.
        self.controller.record(self.current_obs, self.alpha)
        return self.alpha
