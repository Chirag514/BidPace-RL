"""Estimate price scale from training quantiles and completed-hour feedback."""

import numpy as np


class MarketFilter:
    """Estimate persistent price and traffic scales from past winning feedback."""

    def __init__(self, tensors, auctions=12000, rate=0.25):
        self.levels = np.asarray(tensors["market_quantiles"])
        self.prices = np.asarray(tensors["market_prices"])
        self.campaigns = np.asarray(tensors["campaigns"]).tolist()
        self.auctions, self.rate = int(auctions), float(rate)
        if not 0 <= self.rate <= 1 or self.auctions < 1:
            raise ValueError("Invalid market-filter settings")

    def reset(self, day):
        self.campaign = self.campaigns.index(day.advertiser)
        self.profile, self.budget = day.profile, day.budget
        self.price = self.traffic = 1.0
        self.previous = None

    def observe(self, obs):
        """Process the completed hour, never the current/future hidden hour."""
        if self.previous is None:
            return
        hour, bid, left = self.previous
        if int(round(float(obs[1]) * 24)) != hour + 1:
            raise ValueError("Market feedback must arrive once per completed hour")
        self.previous = None
        win = float(obs[3])
        spend = max(0.0, left - float(obs[0]) * self.budget)
        # Affordability clipping changes the win CDF. Ignore censored feedback.
        if bid <= 0 or not 0.02 < win < 0.98 or spend >= left * 0.95:
            return
        quantile = float(np.interp(win, self.levels, self.prices[self.campaign, hour]))
        if quantile > 1e-8:
            estimate = np.clip(bid / quantile, 0.25, 4.0)
            self.price = float(
                np.exp((1 - self.rate) * np.log(self.price) + self.rate * np.log(estimate))
            )
        # cost / average paid price / win fraction recovers inventory size.
        # The average paid-price observation must be uncensored for this step.
        paid = float(obs[7]) * 4.0
        forecast = self.auctions * self.profile.volume[hour]
        if paid > 1e-8 and forecast > 0 and float(obs[7]) < 0.999:
            inventory = spend / paid / win
            estimate = np.clip(inventory / forecast, 0.25, 4.0)
            self.traffic = float(
                np.exp((1 - self.rate) * np.log(self.traffic) + self.rate * np.log(estimate))
            )

    def record(self, obs, alpha):
        hour = min(int(round(float(obs[1]) * 24)), 23)
        bid = alpha * self.profile.ctr[hour] / self.profile.mean_ctr
        self.previous = (hour, bid, float(obs[0]) * self.budget)
