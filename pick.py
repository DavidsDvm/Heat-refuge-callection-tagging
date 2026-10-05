"""Pick the images to annotate from the usable pool (project.md 4.2 and 4.6).

1. KartaView is capped at KARTAVIEW_SHARE of each city. Nearly all of it is shot from the
   traffic lane through a windshield, so where it dominates a city a model could learn
   viewpoint or source instead of the scene.
2. The rest is drawn by weighted sampling without replacement, each image weighted by
   1 / frequency of its rarest feature (composition tag or scene type). Rare compositions
   such as shade structures, water, heavy shade or plazas end up over-represented compared
   with the pool. The weights come only from the automatic composition tags, never from a
   heat-refuge judgement.

Each image's random draw comes from a hash of its id, so removing images from the pool
(e.g. after review) keeps everything else selected and only adds replacements.

Output: data/selected.csv, the candidate metadata (license, attribution, stratum, ...) and
the tags of every selected image, and data/final_images/ kept in sync with it.

Usage:
  python pick.py             # 1000 images
  python pick.py --n 1200
"""
import argparse
import csv
import hashlib
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from tag import FIELDS, summarize

DATA = Path(__file__).parent / "data"
FINAL = DATA / "final_images"
KARTAVIEW_SHARE = .40


def features(row):
    return {t for t in row["tags"].split(";") if t} | {row["scene"]}


def weighted_sample(rows, k, weight, seed):
    """k rows without replacement, each drawn with chance proportional to weight (Efraimidis-Spirakis)."""
    u = lambda r: int(hashlib.md5(f"{seed}:{r['image_id']}".encode()).hexdigest(), 16) / 16 ** 32
    return sorted(rows, key=lambda r: u(r) ** (1 / weight(r)), reverse=True)[:k]


def select(rows, n, seed):
    freq = Counter(f for r in rows for f in features(r))
    weight = lambda r: 1 / min(freq[f] for f in features(r))
    by_city = defaultdict(list)
    for r in rows:
        by_city[r["city"]].append(r)
    pool = []
    for city_rows in by_city.values():
        kart = [r for r in city_rows if r["source_platform"] == "kartaview"]
        other = [r for r in city_rows if r["source_platform"] != "kartaview"]
        # kart / (kart + other) <= share  <=>  kart <= share / (1 - share) * other.
        # ponytail: the cap holds for the pool; the weighted draw below can shift a city's
        # share by a few points. Re-cap after drawing if it ever needs to be exact.
        pool += other + weighted_sample(kart, int(KARTAVIEW_SHARE / (1 - KARTAVIEW_SHARE) * len(other)), weight, seed)
    return weighted_sample(pool, n, weight, seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1000, help="number of images to select")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    candidates = {r["image_id"]: r for r in csv.DictReader((DATA / "candidates.csv").open())}
    usable = [{**candidates[r["image_id"]], **r} for r in csv.DictReader((DATA / "tags.csv").open())
              if r["usable"] == "1" and (DATA / candidates[r["image_id"]]["file"]).exists()]
    chosen = sorted(select(usable, args.n, args.seed), key=lambda r: r["image_id"])

    fields = list(next(iter(candidates.values()))) + FIELDS[1:]
    with (DATA / "selected.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(chosen)
    FINAL.mkdir(exist_ok=True)
    keep = {Path(r["file"]).name for r in chosen}
    removed = [p for p in FINAL.glob("*.jpg") if p.name not in keep]
    for path in removed:
        path.unlink()
    added = [r for r in chosen if not (FINAL / Path(r["file"]).name).exists()]
    for r in added:
        shutil.copy2(DATA / r["file"], FINAL)
    print(f"{len(usable)} usable images -> {len(chosen)} selected into data/selected.csv")
    print(f"  data/final_images: {len(added)} added, {len(removed)} removed")
    if len(chosen) < args.n:
        print(f"  fewer than --n {args.n} remain after the KartaView cap: collect more images")
    print("  by city:", dict(Counter(r["city"] for r in chosen).most_common()))
    print("  by stratum:", dict(Counter(r["stratum"] for r in chosen)))
    summarize(chosen, 10)


if __name__ == "__main__":
    main()
