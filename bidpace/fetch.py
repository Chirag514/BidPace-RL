"""Download public iPinYou logs with provider MD5 verification.

Completed verified downloads are reused. Partial transfers retain a .part suffix
and are ignored by replay preparation.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request

API = "https://www.kaggle.com/api/v1/datasets/"
DATASET = "lastsummer/ipinyou"
PREFIX = "ipinyou.contest.dataset/"


def selected(name, season=2, scope="training", include_test=False):
    """Select log roles explicitly; tests stay separate from training semantics.

    ``training`` preserves the original bid/imp/clk download scope. ``priced``
    omits bid-request logs: losing prices are unobserved and those logs are not
    training examples for this conditional winning-inventory replay.
    """
    if season not in (1, 2, 3) or scope not in ("training", "priced"):
        raise ValueError("Unsupported season or download scope")
    relative = name.removeprefix(PREFIX)
    ordinal = {1: "1st", 2: "2nd", 3: "3rd"}[season]
    tests = {1: "20130318_20", 2: "20130613_15", 3: "20131021_28"}
    if include_test and relative == f"testing{ordinal}/leaderboard.test.data.{tests[season]}.txt.bz2":
        return True
    dates = {1: r"201303(11|12|13|14|15|16|17)",
             2: r"201306(06|07|08|09|10|11|12)",
             3: r"201310(19|20|21|22|23|24|25|26|27)"}[season]
    roles = "imp|clk" if scope == "priced" else "bid|imp|clk"
    return bool(re.fullmatch(rf"training{ordinal}/({roles})\.{dates}\.txt\.bz2", relative))


def md5_file(path):
    digest = hashlib.md5()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(name, root, expected_size=None, expected_md5=None):
    relative = Path(name.removeprefix(PREFIX))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("unsafe provider path")
    target = Path(root) / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and (expected_size is None or target.stat().st_size == expected_size):
        if expected_md5 and md5_file(target) == expected_md5:
            return target
    temporary = target.with_suffix(target.suffix + ".part")
    url = API + "download/" + DATASET + "/" + urllib.parse.quote(name, safe="")
    with urllib.request.urlopen(url, timeout=60) as source, temporary.open("wb") as output:
        while chunk := source.read(4 * 1024 * 1024):
            output.write(chunk)
    if expected_size is not None and temporary.stat().st_size != expected_size:
        raise ValueError(f"size mismatch: {relative}")
    if expected_md5 and md5_file(temporary) != expected_md5:
        raise ValueError(f"MD5 mismatch: {relative}")
    temporary.replace(target)
    return target


def fetch(root, workers=4, season=2, scope="training", include_test=False):
    root = Path(root)
    metadata = []
    token = None
    while True:
        url = API + "list/" + DATASET
        if token:
            url += "?pageToken=" + urllib.parse.quote(token, safe="")
        with urllib.request.urlopen(url, timeout=60) as response:
            page = json.load(response)
        metadata.extend(page["datasetFiles"])
        token = page.get("nextPageToken")
        if not token:
            break
    names = [item["name"] for item in metadata]
    if len(names) != len(set(names)):
        raise ValueError("provider listing contains duplicate names")
    chosen = [item for item in metadata if selected(item["name"], season, scope, include_test)]
    expected_count = (9 if season == 3 else 7) * (2 if scope == "priced" else 3) + int(include_test)
    if len(chosen) != expected_count:
        raise ValueError(
            f"expected {expected_count} Season {season} files, got {len(chosen)}; inspect changed provider listing"
        )
    for name in ["README", "files.md5", "known.data.bugs.txt"]:
        download_file(PREFIX + name, root)
    checks = dict(re.findall(r"MD5 \((.+)\) = ([0-9a-f]{32})", (root / "files.md5").read_text()))

    def get(item):
        relative = item["name"].removeprefix(PREFIX)
        expected = checks[relative]
        path = download_file(item["name"], root, int(item["totalBytes"]), expected)
        print("Verified", relative, flush=True)
        return dict(file=relative, size=path.stat().st_size, provider_md5=expected)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        verified = list(executor.map(get, chosen))
    report = dict(dataset=DATASET, scope=f"public Season {season} {scope} logs", include_test=include_test, files=verified)
    (root / "download_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--season", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--scope", choices=["training", "priced"], default="training")
    parser.add_argument("--include-test", action="store_true", help="Also download the public leaderboard; never use its labels for fitting")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    try:
        fetch(args.out, args.workers, args.season, args.scope, args.include_test)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Download failed: {error}. Keep completed files and rerun the command.\n")


if __name__ == "__main__":
    main()
