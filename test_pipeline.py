"""Self-check for the pure logic in collect.py, tag.py and pick.py. Run: python test_pipeline.py"""
import math
import random

import numpy as np

from collect import is_open_license, random_point, season
from pick import select
from tag import PROMPTS, describe, pixel_stats, train_junk_filter

# Licences: open ones in, NC/ND and unknown out.
for name in ["CC BY-SA 4.0", "CC-BY-SA-4.0", "CC BY 2.0", "CC0", "etalab-2.0", "Public domain"]:
    assert is_open_license(name), name
for name in ["CC BY-NC-SA 2.0", "CC BY-ND 4.0", "GFDL", "", "All rights reserved"]:
    assert not is_open_license(name), name

# Random points stay inside the radius.
rng = random.Random(0)
for _ in range(1000):
    lat, lon = random_point(rng, 48.85, 2.35, 5)
    dy, dx = (lat - 48.85) * 111.32, (lon - 2.35) * 111.32 * math.cos(math.radians(48.85))
    assert math.hypot(dx, dy) <= 5.001

# Pixel stats: blue sky over a grey road whose bottom 2 of 5 rows are in shadow.
mask = np.zeros((10, 10), int)
mask[5:] = 1  # 0 = sky, 1 = road
rgb = np.zeros((10, 10, 3), np.float32)
rgb[:5] = [.3, .5, .9]
rgb[5:] = .6
rgb[8:] = .2
sky_blue, ground_shadow, gray = pixel_stats(rgb, mask, [0], [1])
assert sky_blue > .5 and abs(ground_shadow - .4) < 1e-6 and not gray, (sky_blue, ground_shadow, gray)
assert pixel_stats(rgb.mean(axis=2, keepdims=True).repeat(3, axis=2), mask, [0], [1])[2]  # black and white

# Foliage season: hemispheres mirrored, tropics exempt, year-only dates unknown.
assert season("2023-07-14", 48.8) == "leaf_on" and season("2023-01", 48.8) == "leaf_off"
assert season("2023-01-02", -33.9) == "leaf_on" and season("2023-10", -33.9) == "shoulder"
assert season("2023-01", 1.35) == "tropical" and season("2012", 40.4) == "unknown"


def clip_logits(*prompts):
    return [10.0 if p in prompts else 0.0 for _, _, p in PROMPTS]


day_street = clip_logits("a street-level photo of a city street", "a photo taken during the day")
row = describe({"tree": .2, "sky": .2, "road": .3, "sidewalk": .1, "car": .05}, sky_blue, .2, False, day_street)
assert row["tags"] == "tree;open_sky;street;sidewalk;vehicle;sunny;partial_shadow", row["tags"]
assert row["usable"] == 1 and row["reject_reason"] == "" and row["scene"] == "street"
# Overcast: dark ground patches are not shadows. No sky visible: lighting unknown.
assert describe({"road": .5}, 0.0, .5, False, day_street)["tags"] == "street;overcast"
assert describe({"road": .5}, math.nan, .5, False, day_street)["tags"] == "street;mostly_shaded"
# Night, black-and-white and non-streetscapes are rejected, with the reason recorded.
night = clip_logits("a street-level photo of a city street", "a photo taken at night")
statue = clip_logits("a photo of a statue", "a photo taken during the day")
assert describe({"road": .5}, math.nan, math.nan, False, night)["reject_reason"] == "night"
assert describe({"road": .5}, math.nan, math.nan, True, day_street)["reject_reason"] == "grayscale"
assert describe({}, math.nan, math.nan, False, statue)["reject_reason"] == "not_streetscape"

# pick.py: KartaView capped at 40% of a city; a rare-tag image is picked far more often than chance.
row = lambda i, src, tags: {"image_id": i, "city": "X", "source_platform": src, "tags": tags, "scene": "street"}
rows = ([row(f"k{i}", "kartaview", "street") for i in range(30)] + [row(f"w{i}", "wikimedia", "street") for i in range(10)]
        + [row("rare", "wikimedia", "street;shade_structure")])
assert sum(r["source_platform"] == "kartaview" for r in select(rows, 100, 0)) == 7  # .4/.6 * 11
hits = sum(any(r["image_id"] == "rare" for r in select(rows, 5, s)) for s in range(200))
assert hits > 180, hits  # uniform picking would give about 5/18 * 200 = 56
# Removing an unpicked row from the pool leaves the rest of the selection unchanged.
picked = {r["image_id"] for r in select(rows, 5, 0)}
dropped = next(r for r in rows if r["image_id"] not in picked and r["source_platform"] == "wikimedia")
assert {r["image_id"] for r in select([r for r in rows if r is not dropped], 5, 0)} == picked
# Junk filter: junk shifted along one embedding axis is told apart from good images.
gen = np.random.default_rng(0)
X = gen.normal(size=(40, 8)).astype(np.float32)
X[:20, 0] += 3
X /= np.linalg.norm(X, axis=1, keepdims=True)
w, b = train_junk_filter(X, np.array([1] * 20 + [0] * 20, dtype=np.float32))
p_junk = 1 / (1 + np.exp(-(X @ w + b)))
assert (p_junk[:20] > .5).mean() >= .9 and (p_junk[20:] < .5).mean() >= .9, p_junk
print("ok")
