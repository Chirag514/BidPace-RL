"""Chronological replay of logged winning impressions, with dated click labels.

Only prices, binary responses and campaign/time groups are exported. Lost-auction
prices remain unobserved. This is conditional logged-inventory replay, not a
counterfactual estimate of the full advertising market.
"""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import bz2
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import numpy as np


@dataclass
class Profile:
    advertiser: int
    ctr: np.ndarray
    volume: np.ndarray
    price: float
    mean_ctr: float
    mean_daily_impressions: float
    reference_cost: float


def open_text(path):
    return (
        bz2.open(path, "rt", encoding="utf8")
        if str(path).endswith(".bz2")
        else Path(path).open(encoding="utf8")
    )


def bid_counts(path):
    """Count all bid requests without pretending their prices are observed."""
    counts = Counter()
    with open_text(path) as source:
        for line in source:
            if not line.strip():
                continue
            if line.count("\t") != 20:
                raise ValueError(f"{Path(path).name}: expected 21 bid columns")
            timestamp = line.split("\t", 2)[1]
            advertiser = int(line.rsplit("\t", 2)[1])
            counts[int(timestamp[:8]), int(timestamp[8:10]), advertiser] += 1
    return counts


def prepare(raw, output, workers=4):
    """Join raw records; same-price repeats keep the earliest impression time.

    Price-conflicting IDs are quarantined within their impression date. Clicks
    attach to the latest preceding matching impression; label dates are retained
    so each training fold can enforce its own information cutoff.
    """
    raw, output = Path(raw), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    database = output / "join.sqlite"
    if database.exists() or (output / "replay.npz").exists():
        raise FileExistsError("Choose a fresh preparation output directory")
    started = time.monotonic()
    files = sorted(raw.rglob("*"))
    files = [p for p in files if p.is_file() and not p.name.endswith((".part", ".tmp"))]
    bids = [p for p in files if p.name.startswith("bid.")]
    logs = [p for p in files if p.name.startswith(("imp.", "clk.", "leaderboard.test.data."))]
    if not logs:
        raise ValueError("No supported impression/click/test logs found")
    con = sqlite3.connect(database)
    con.executescript(
        """PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;
        CREATE TABLE imps(date INTEGER,adv INTEGER,bid TEXT,ts TEXT,hour INTEGER,
                          price INTEGER,attached INTEGER,seen INTEGER DEFAULT 1,
                          PRIMARY KEY(date,adv,bid)) WITHOUT ROWID;
        CREATE TABLE conflicts(date INTEGER,adv INTEGER,bid TEXT,
                               PRIMARY KEY(date,adv,bid)) WITHOUT ROWID;
        CREATE TABLE clicks(adv INTEGER,bid TEXT,ts TEXT,PRIMARY KEY(adv,bid,ts)) WITHOUT ROWID;
        CREATE TRIGGER conflict BEFORE UPDATE ON imps WHEN OLD.price<>NEW.price
        BEGIN INSERT INTO conflicts VALUES(NEW.date,NEW.adv,NEW.bid)
          ON CONFLICT(date,adv,bid) DO NOTHING; END;
    """
    )
    counts, requests, sources = Counter(), Counter(), []
    imp_sql = """INSERT INTO imps(date,adv,bid,ts,hour,price,attached) VALUES(?,?,?,?,?,?,?)
        ON CONFLICT(date,adv,bid) DO UPDATE SET ts=MIN(ts,excluded.ts),
        hour=MIN(hour,excluded.hour),price=excluded.price,
        attached=MAX(attached,excluded.attached),seen=seen+1"""
    click_sql = "INSERT INTO clicks VALUES(?,?,?) ON CONFLICT(adv,bid,ts) DO NOTHING"
    try:
        # Separate compressed files can be decompressed independently.
        if bids:
            print(f"Counting {len(bids)} bid logs", flush=True)
            with ProcessPoolExecutor(max_workers=min(workers, len(bids))) as pool:
                for path, counter in zip(bids, pool.map(bid_counts, bids)):
                    requests.update(counter)
                    sources.append(
                        dict(file=str(path.relative_to(raw)), bid_requests=sum(counter.values()))
                    )
                    print("Counted", path.name, sum(counter.values()), flush=True)
        for path in logs:
            print("Joining", path.name, flush=True)
            imp_batch, click_batch = [], []
            source_count = Counter()
            with open_text(path) as source:
                for line in source:
                    if not line.strip():
                        continue
                    fields = line.rstrip("\r\n").split("\t")
                    if len(fields) not in (24, 26):
                        raise ValueError(f"{path.name}: expected 24 or 26 columns")
                    ts = fields[1]
                    if len(ts) != 17 or not ts.isdigit():
                        raise ValueError("Expected millisecond timestamp")
                    date, hour = int(ts[:8]), int(ts[8:10])
                    if not 0 <= hour < 24:
                        raise ValueError("Invalid hour")
                    advertiser, kind = int(fields[22]), int(fields[2])
                    if kind == 1:
                        price = float(fields[20])
                        if not np.isfinite(price) or price < 0 or price != int(price):
                            raise ValueError("Replay requires nonnegative integral paying prices")
                        attached = int(len(fields) == 26 and int(fields[24]) > 0)
                        imp_batch.append(
                            (date, advertiser, fields[0], ts, hour, int(price), attached)
                        )
                        source_count["impression_events"] += 1
                    elif kind == 2:
                        click_batch.append((advertiser, fields[0], ts))
                        source_count["click_events"] += 1
                    else:
                        raise ValueError("Unexpected event type in impression/click input")
                    if len(imp_batch) + len(click_batch) >= 50000:
                        con.executemany(imp_sql, imp_batch)
                        imp_batch.clear()
                        con.executemany(click_sql, click_batch)
                        click_batch.clear()
                        con.commit()
            con.executemany(imp_sql, imp_batch)
            con.executemany(click_sql, click_batch)
            con.commit()
            counts.update(source_count)
            sources.append(dict(file=str(path.relative_to(raw)), **source_count))
        print("Assigning dated click labels", flush=True)
        con.executescript(
            """CREATE INDEX imp_bid_time ON imps(adv,bid,ts);
            CREATE TABLE assigned AS
            WITH ranked AS (
              SELECT i.date,i.adv,i.bid,c.ts,
              ROW_NUMBER() OVER(PARTITION BY c.adv,c.bid,c.ts ORDER BY i.ts DESC) AS rank
              FROM clicks c JOIN imps i ON i.adv=c.adv AND i.bid=c.bid AND i.ts<=c.ts)
            SELECT r.date,r.adv,r.bid,MIN(r.ts) AS click_ts FROM ranked r
            WHERE rank=1 AND NOT EXISTS(SELECT 1 FROM conflicts x
              WHERE x.date=r.date AND x.adv=r.adv AND x.bid=r.bid)
            GROUP BY r.date,r.adv,r.bid;
            CREATE UNIQUE INDEX assigned_key ON assigned(date,adv,bid);
        """
        )
        raw_unique = int(con.execute("SELECT COUNT(*) FROM imps").fetchone()[0])
        conflicts = int(con.execute("SELECT COUNT(*) FROM conflicts").fetchone()[0])
        excluded = int(
            con.execute(
                "SELECT COALESCE(SUM(seen),0) FROM imps JOIN conflicts USING(date,adv,bid)"
            ).fetchone()[0]
        )
        valid = raw_unique - conflicts
        prices = np.empty(valid, dtype=np.uint32)
        label_dates = np.zeros(valid, dtype=np.int32)
        attached = np.zeros(valid, dtype=np.uint8)
        keys, starts, ends = [], [], []
        position = 0
        previous = None
        cursor = con.execute(
            """SELECT i.date,i.adv,i.hour,i.price,a.click_ts,i.attached
            FROM imps i LEFT JOIN assigned a USING(date,adv,bid)
            WHERE NOT EXISTS(SELECT 1 FROM conflicts x
              WHERE x.date=i.date AND x.adv=i.adv AND x.bid=i.bid)
            ORDER BY i.date,i.adv,i.hour,i.ts,i.bid"""
        )
        print("Exporting ordered replay arrays", valid, "impressions", flush=True)
        while rows := cursor.fetchmany(100000):
            for date, adv, hour, price, click_ts, test_click in rows:
                key = (date, adv, hour)
                if key != previous:
                    if previous is not None:
                        ends.append(position)
                    keys.append(key)
                    starts.append(position)
                    previous = key
                prices[position] = price
                if click_ts:
                    label_dates[position] = int(click_ts[:8])
                attached[position] = test_click
                position += 1
        if previous is not None:
            ends.append(position)
        assert position == valid
        np.savez_compressed(
            output / "replay.npz",
            prices=prices,
            label_dates=label_dates,
            attached=attached,
            keys=np.asarray(keys, dtype=np.int32),
            starts=np.asarray(starts, dtype=np.int64),
            ends=np.asarray(ends, dtype=np.int64),
        )
        report = dict(
            scope="selected public priced winning inventory",
            counts=dict(counts),
            bid_requests=sum(requests.values()),
            raw_unique_impression_ids=raw_unique,
            exported_impressions=valid,
            repeated_impression_records=counts["impression_events"] - raw_unique,
            conflicting_price_ids=conflicts,
            excluded_conflicting_events=excluded,
            unique_click_events=int(con.execute("SELECT COUNT(*) FROM clicks").fetchone()[0]),
            matched_clicked_impressions=int(
                con.execute("SELECT COUNT(*) FROM assigned").fetchone()[0]
            ),
            sources=sources,
            label_policy="earliest click assigned to latest preceding impression; fold-specific label-date cutoff",
            duplicate_policy="date/advertiser/bid ID; same-price repeats keep earliest timestamp; conflicting prices quarantined",
            attached_policy="provider binary attached click labels allowed only in reserved test dates",
            unobserved="lost-auction prices and click counterfactuals unavailable",
            sha256=hashlib.sha256((output / "replay.npz").read_bytes()).hexdigest(),
            seconds=time.monotonic() - started,
        )
        (output / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    finally:
        con.close()
        if (output / "preparation.json").exists():
            for path in [database, Path(str(database) + "-wal"), Path(str(database) + "-shm")]:
                if path.exists():
                    path.unlink()


@dataclass
class ReplayHour:
    prices: np.ndarray
    clicks: np.ndarray
    bid_quality: float


@dataclass
class ReplayTemplate:
    date: int
    advertiser: int
    split: str
    profile: Profile
    hours: tuple


@dataclass
class ReplayDay:
    hours: tuple
    profile: Profile
    budget: float
    date: int
    advertiser: int
    price_multiplier: float


class ReplayData:
    """Fit each chronological fold using only its dated training labels."""

    def __init__(self, path, train_dates, validation_dates=(), test_dates=()):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as z:
            self.arrays = {k: z[k] for k in z.files}
        a = self.arrays
        if not (len(a["prices"]) == len(a["label_dates"]) == len(a["attached"])):
            raise ValueError("Replay arrays disagree")
        if np.any(a["prices"] < 0) or np.any(a["ends"] < a["starts"]):
            raise ValueError("Invalid replay values")
        self.dates = {
            "train": tuple(sorted(train_dates)),
            "validation": tuple(sorted(validation_dates)),
            "test": tuple(sorted(test_dates)),
        }
        reserved = {20130613, 20130614, 20130615}
        if reserved.intersection(self.dates["train"] + self.dates["validation"]):
            raise ValueError("Public Season 2 leaderboard dates are reserved for evaluation")
        if not self.dates["train"]:
            raise ValueError("Training dates required")
        groups = [v for v in self.dates.values() if v]
        if any(max(left) >= min(right) for left, right in zip(groups, groups[1:])):
            raise ValueError("Partitions must be disjoint and chronological")
        self.lookup = {
            tuple(key): slice(int(start), int(end))
            for key, start, end in zip(a["keys"], a["starts"], a["ends"])
        }
        if len(self.lookup) != len(a["keys"]):
            raise ValueError("Duplicate replay groups")
        self.profiles = {}
        train_cut = max(self.dates["train"])
        campaigns = sorted({adv for date, adv, h in self.lookup if date in self.dates["train"]})
        for adv in campaigns:
            impressions = np.zeros(24)
            clicks = np.zeros(24)
            costs = 0.0
            for date in self.dates["train"]:
                for h in range(24):
                    sl = self.lookup.get((date, adv, h), slice(0, 0))
                    impressions[h] += len(a["prices"][sl])
                    costs += float(a["prices"][sl].sum(dtype=np.float64))
                    labels = a["label_dates"][sl]
                    clicks[h] += np.count_nonzero((labels > 0) & (labels <= train_cut))
                    if a["attached"][sl].any():
                        raise ValueError("Attached test labels cannot be used for fitting")
            if impressions.sum() == 0 or costs <= 0:
                raise ValueError("Training campaign lacks inventory")
            mean_ctr = float(clicks.sum() / impressions.sum())
            if mean_ctr == 0:
                mean_ctr = 0.5 / (impressions.sum() + 1)
            ctr = (clicks + 10000 * mean_ctr) / (impressions + 10000)
            volume = impressions / impressions.sum()
            active_dates = {
                date
                for date, campaign, h in self.lookup
                if campaign == adv and date in self.dates["train"]
            }
            mean_daily = float(impressions.sum() / len(active_dates))
            mean_price = costs / impressions.sum()
            self.profiles[int(adv)] = Profile(
                int(adv),
                ctr,
                volume,
                mean_price,
                float(np.dot(ctr, volume)),
                mean_daily,
                costs / len(active_dates),
            )
        self.reward_scale = float(
            np.mean([p.mean_daily_impressions * p.mean_ctr for p in self.profiles.values()])
        )
        self.templates = {s: [] for s in self.dates}
        available = {(date, adv) for date, adv, h in self.lookup}
        for split, dates in self.dates.items():
            if not dates:
                continue
            cutoff = max(dates)
            for date in dates:
                for adv, p in self.profiles.items():
                    if (date, adv) not in available:
                        continue
                    hours = []
                    for h in range(24):
                        sl = self.lookup.get((date, adv, h), slice(0, 0))
                        labels = a["label_dates"][sl]
                        clicked = (labels > 0) & (labels <= cutoff)
                        if split == "test":
                            clicked |= a["attached"][sl] > 0
                        elif a["attached"][sl].any():
                            raise ValueError("Reserved labels entered development data")
                        prices = a["prices"][sl]
                        clicked = clicked.astype(np.uint8)
                        prices.flags.writeable = False
                        clicked.flags.writeable = False
                        hours.append(
                            ReplayHour(prices, clicked, float(p.ctr[h] / p.mean_ctr * p.price))
                        )
                    self.templates[split].append(ReplayTemplate(date, adv, split, p, tuple(hours)))

    def audit(self):
        return dict(
            sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
            dates=self.dates,
            reward="recorded binary clicks on logged winning inventory",
            profiles={
                str(k): dict(
                    price=p.price,
                    ctr=p.ctr.tolist(),
                    volume=p.volume.tolist(),
                    reference_cost=p.reference_cost,
                )
                for k, p in self.profiles.items()
            },
        )


def make_replay_day(
    template, budget_fraction=0.3, competition=1.0, traffic=1.0, quality=1.0, seed=0, coverage=1.0
):
    if min(budget_fraction, competition, traffic, quality, coverage) <= 0:
        raise ValueError("Positive scenario parameters required")
    if coverage > 1:
        raise ValueError("Coverage retains a fraction of recorded opportunities")
    rng = np.random.default_rng(seed)
    hours = []
    total_cost = total_inventory = 0
    for h, hr in enumerate(template.hours):
        prices = (
            hr.prices
            if competition == 1.0
            else np.rint(hr.prices.astype(np.float64) * competition).astype(np.uint32)
        )
        clicks = hr.clicks
        if coverage != 1.0:
            keep = rng.random(len(prices)) < coverage
            prices = prices[keep]
            clicks = clicks[keep]
        if traffic != 1.0 and 18 <= h < 22:
            # Explicit synthetic traffic stress; duplicates are not new real data.
            copies = int(traffic)
            fractional = int(round((traffic - copies) * len(prices)))
            prices = np.concatenate([np.tile(prices, copies), prices[:fractional]])
            clicks = np.concatenate([np.tile(clicks, copies), clicks[:fractional]])
        if quality != 1.0:
            if quality > 1:
                raise ValueError("Recorded-label quality stress can only thin clicks")
            clicks = clicks * (rng.random(len(clicks)) < quality)
        prices.flags.writeable = False
        clicks.flags.writeable = False
        total_cost += prices.sum()
        total_inventory += len(prices)
        hours.append(ReplayHour(prices, clicks, hr.bid_quality))
    # Full-day mean price is a look-ahead diagnostic, never a deployable feature.
    hidden_price = float(total_cost / max(total_inventory, 1) / template.profile.price)
    return ReplayDay(
        tuple(hours),
        template.profile,
        budget_fraction * template.profile.reference_cost,
        template.date,
        template.advertiser,
        hidden_price,
    )


def affordable_indices(prices, ids, budget):
    """Arrival-order affordability with a vectorized prefix and exact suffix."""
    candidate = prices[ids]
    cumulative = np.cumsum(candidate, dtype=np.float64)
    k = int(np.searchsorted(cumulative, budget, side="right"))
    if k == len(ids):
        return ids
    spent = float(cumulative[k - 1]) if k else 0.0
    remaining = budget - spent
    # Near a floating threshold, use the literal rule to preserve tie behavior.
    tolerance = np.finfo(float).eps * max(1.0, float(cumulative[-1])) * (len(ids) + 1)
    if candidate.dtype.kind not in "ui" and (
        (k and abs(remaining) <= tolerance) or abs(cumulative[k] - budget) <= tolerance
    ):
        k = 0
        remaining = budget
    chosen = []
    # The remaining budget only decreases, so initially unaffordable prices can
    # never become affordable. Most long tails therefore need no Python loop.
    tail = ids[k:]
    for index in tail[prices[tail] <= remaining]:
        if prices[index] <= remaining:
            chosen.append(index)
            remaining -= float(prices[index])
    return np.r_[ids[:k], np.asarray(chosen, dtype=np.int64)]


class ReplayEnv:
    """24 hourly decisions; real ordered prices/clicks and hard daily budget."""

    def __init__(self, day, features=False):
        self.day = day
        self.h = 0
        self.left = float(day.budget)
        self.spent = 0.0
        self.clicks = self.wins = 0
        self.last_spend = self.last_win = self.last_price = 0.0
        self.done = False
        self.trace = [0.0]
        self.features = features
        self.seen = 0
        self.last_seen = 0
        self.last_alpha = 1.0

    def obs(self):
        p = self.day.profile
        h = min(self.h, 23)
        basic = [
            self.left / self.day.budget,
            self.h / 24,
            min(self.last_spend / (self.day.budget / 24), 4) / 4,
            self.last_win,
            min(p.ctr[h] / p.mean_ctr, 4) / 4,
            min(p.volume[h] * 24, 4) / 4,
            self.day.budget / p.reference_cost,
            min(self.last_price / p.price, 4) / 4,
        ]
        if self.features:
            prior = 0.05 * p.mean_daily_impressions
            expected = p.mean_daily_impressions * p.volume[: self.h].sum()
            scale = float(np.clip((self.seen + prior) / (expected + prior), 0.05, 4.0))
            remaining = p.mean_daily_impressions * p.volume[self.h :].sum() * scale
            per_opportunity = self.left / max(remaining * p.price, 1.0)
            last_expected = p.mean_daily_impressions * p.volume[max(self.h - 1, 0)]
            last_ratio = self.last_seen / max(last_expected, 1.0) if self.h else 1.0
            basic.extend(
                [
                    scale / 4,
                    min(per_opportunity, 2.0) / 2,
                    min(last_ratio, 4.0) / 4,
                    self.last_alpha / 3.0,
                    np.log1p(p.mean_daily_impressions * p.mean_ctr) / 10.0,
                ]
            )
        return np.asarray(basic, dtype=np.float32)

    def step(self, alpha, constant=False):
        if self.done:
            raise RuntimeError("Episode already ended")
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("Invalid bid")
        hr = self.day.hours[self.h]
        bid = alpha * self.day.profile.price if constant else alpha * hr.bid_quality
        ids = np.flatnonzero((hr.prices <= bid) & (alpha > 0))
        ids = affordable_indices(hr.prices, ids, self.left)
        cost = min(float(hr.prices[ids].sum()), self.left)
        clicks = int(hr.clicks[ids].sum())
        self.left = max(0.0, self.left - cost)
        self.spent += cost
        self.clicks += clicks
        self.wins += len(ids)
        self.last_spend = cost
        self.last_win = len(ids) / max(len(hr.prices), 1)
        self.last_price = cost / max(len(ids), 1)
        self.seen += len(hr.prices)
        self.last_seen = len(hr.prices)
        self.last_alpha = float(alpha)
        self.h += 1
        self.done = self.h == 24 or self.left <= 1e-10
        self.trace.append(self.spent)
        return self.obs(), float(clicks), self.done


def rollout_replay(day, policy):
    env = ReplayEnv(day, features=getattr(policy, "requires_traffic", False))
    policy.reset(day)
    while not env.done:
        action = policy.act(env.obs())
        env.step(action, constant=getattr(policy, "constant", False))
    trace = np.pad(env.trace, (0, 25 - len(env.trace)), mode="edge") / day.budget
    return dict(
        clicks=env.clicks,
        spent=env.spent,
        budget=day.budget,
        utilization=env.spent / day.budget,
        wins=env.wins,
        violations=int(env.spent > day.budget + 1e-7),
        trace=trace.tolist(),
    )


def combine(paths, output, exclude_dates=()):
    """Combine disjoint numeric replay groups, retaining dated response labels."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "replay.npz"
    if target.exists():
        raise FileExistsError("Choose a fresh combined output directory")
    parts = []
    groups = []
    excluded = set(exclude_dates)
    sources = []
    for path in map(Path, paths):
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        owner = len(parts)
        parts.append(arrays)
        for key, start, end in zip(arrays["keys"], arrays["starts"], arrays["ends"]):
            if int(key[0]) not in excluded:
                groups.append((tuple(map(int, key)), owner, int(start), int(end)))
        sources.append(
            dict(file=path.as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        )
    groups.sort()
    if len({g[0] for g in groups}) != len(groups):
        raise ValueError("Input replay groups overlap")
    total = sum(end - start for _, _, start, end in groups)
    result = {
        name: np.empty(total, dtype=parts[0][name].dtype)
        for name in ["prices", "label_dates", "attached"]
    }
    keys = []
    starts = []
    ends = []
    position = 0
    for key, owner, start, end in groups:
        keys.append(key)
        starts.append(position)
        next_position = position + end - start
        for name in result:
            result[name][position:next_position] = parts[owner][name][start:end]
        position = next_position
        ends.append(position)
    np.savez_compressed(
        target,
        **result,
        keys=np.asarray(keys, dtype=np.int32),
        starts=np.asarray(starts, dtype=np.int64),
        ends=np.asarray(ends, dtype=np.int64),
    )
    audit = dict(
        scope="combined disjoint logged winning inventory",
        sources=sources,
        excluded_dates=sorted(excluded),
        impressions=total,
        sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    )
    (output / "preparation.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw")
    parser.add_argument("--out", required=True)
    parser.add_argument("--combine", nargs="+")
    parser.add_argument("--exclude-dates", nargs="*", type=int, default=[])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if bool(args.raw) == bool(args.combine):
        parser.error("Choose --raw or --combine")
    report = (
        combine(args.combine, args.out, args.exclude_dates)
        if args.combine
        else prepare(args.raw, args.out, args.workers)
    )
    print(json.dumps({k: v for k, v in report.items() if k != "sources"}, indent=2))


if __name__ == "__main__":
    main()
