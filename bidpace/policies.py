"""Fixed, budget-dependent linear and PID bidding baselines."""

import numpy as np


class Fixed:
    """Use a fixed bid multiplier or a literal constant bid price."""

    def __init__(self, alpha=1.0, constant=False):
        self.alpha, self.constant = (float(alpha), constant)

    def reset(self, day):
        pass

    def act(self, obs):
        return self.alpha


class BudgetLinear(Fixed):
    """Validation-tuned bid schedule indexed by the known initial budget."""

    def __init__(self, fractions, alphas):
        self.fractions = np.asarray(fractions, dtype=float)
        self.alphas = np.asarray(alphas, dtype=float)
        self.constant = False
        if len(self.fractions) != len(self.alphas) or len(self.fractions) < 2:
            raise ValueError("At least two budget/alpha pairs are required")
        if (
            not (np.diff(self.fractions) > 0).all()
            or (self.fractions <= 0).any()
            or (self.alphas <= 0).any()
        ):
            raise ValueError("Budget anchors must increase and multipliers must be positive")

    def reset(self, day):
        fraction = day.budget / day.profile.reference_cost
        self.alpha = float(
            np.exp(np.interp(np.log(fraction), np.log(self.fractions), np.log(self.alphas)))
        )


class PID:
    """Adjust bids from cumulative spending error against a training profile."""

    def __init__(self, base=0.5, kp=4.0, ki=0.1, kd=0.0, quality_power=0.0, target="volume"):
        self.base, self.kp, self.ki, self.kd = (base, kp, ki, kd)
        self.quality_power, self.target = (quality_power, target)

    def reset(self, day):
        p = day.profile
        share = p.volume.copy() if self.target == "volume" else p.volume * p.ctr / p.mean_ctr
        self.curve = np.r_[0.0, np.cumsum(share / share.sum())]
        self.integral = self.previous = 0.0

    def act(self, obs):
        h = min(24, int(round(float(obs[1]) * 24)))
        error = self.curve[h] - (1 - float(obs[0]))
        self.integral = float(np.clip(self.integral + error, -2.0, 2.0))
        derivative = error - self.previous
        self.previous = error
        loggain = np.clip(
            self.kp * error + self.ki * self.integral + self.kd * derivative, -5.0, 5.0
        )
        return float(
            np.clip(
                self.base * np.exp(loggain) * max(float(obs[4]) * 4, 0.05) ** self.quality_power,
                0.001,
                5.0,
            )
        )
