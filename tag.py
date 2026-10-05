"""Initial automatic tagging of the collected images (project.md 4.2).

The tags describe dataset composition only. They are NOT heat-refuge labels.

Two pretrained open-weight models:
  * UperNet-ConvNeXt trained on ADE20K (MIT licence) labels every pixel, giving the
    share of the image covered by trees, sky, road, buildings, ... The same mask makes
    two pixel measurements cheap: sky colour (sunny vs overcast) and the share of the
    ground in cast shadow.
  * CLIP ViT-L/14 (MIT licence) zero-shot classifies the scene type, which flags images
    that are not usable streetscapes (interiors, close-ups, statues, highways), and day
    vs night. On a hand-checked sample CLIP was close to chance for sunny/overcast and
    shadows, which is why those come from pixels instead.
  * Junk filter: a logistic regression on the CLIP image embeddings, retrained on every
    run from the human review in data/review.csv, gives usable_p. CLIP prompts miss
    most junk (skylines, crowds, close-ups that still look "urban"); on 801 reviewed
    images the filter reached AUC 0.92; zero-shot Qwen3-VL 2B/4B reached 0.77/0.79.

Images are rejected (usable = 0) for: not_streetscape or highway_or_rural scenes, night,
black-and-white photos, verdict "reject" in data/review.csv, usable_p below USABLE_P_MIN
(unless the review says "keep"), and leafless-tree season beyond a control set of
KEEP_LEAF_OFF.

data/review.csv has columns image_id, verdict (reject / keep), note. Every row is also
a training example for the junk filter, so reviewing more images improves it.

Output (resumable: re-running only tags new images):
  data/tags.csv              frac_* coverage, sky_blue, ground_shadow, scene, time, season,
                             tags, usable_p, reject_reason, usable
  data/clip_embeddings.npz   CLIP embedding of every tagged image, pruned ones included:
                             the junk filter trains on them, so don't delete it
and a composition summary on stdout (tag frequencies, most common tag combinations).

Usage:
  python tag.py              # tag new images, then print the summary
  python tag.py --prune      # also delete the image files of rejected images
  python tag.py --top 30     # longer list of combinations
"""
import argparse
import csv
import hashlib
import math
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from collect import season

DATA = Path(__file__).parent / "data"
REVIEW = DATA / "review.csv"
EMBEDDINGS = DATA / "clip_embeddings.npz"
SEG_MODEL = "openmmlab/upernet-convnext-small"
CLIP_MODEL = "openai/clip-vit-large-patch14"

# Tag -> ADE20K classes whose pixel shares are summed.
GROUPS = {
    "tree": ["tree", "palm"],
    "vegetation": ["grass", "plant", "flower", "field"],  # low greenery; tree + vegetation = green view index
    "open_sky": ["sky"],
    "street": ["road"],
    "sidewalk": ["sidewalk", "path"],
    "building": ["building", "house", "skyscraper", "wall"],
    "vehicle": ["car", "bus", "truck", "van", "minibike", "bicycle"],
    "urban_furniture": ["bench", "chair", "seat", "table", "streetlight", "ashcan", "booth"],
    "shade_structure": ["awning", "canopy", "tent"],
    "water": ["water", "fountain", "swimming pool", "river", "lake", "sea"],
}
GROUND = ["road", "sidewalk", "path", "earth", "floor"]
# ponytail: all thresholds below were picked by eye on a 30-image sample. Tune them
# against the raw columns in tags.csv; that needs no model re-run.
THRESHOLDS = {"tree": .05, "vegetation": .03, "open_sky": .10, "street": .05, "sidewalk": .03, "building": .10,
              "vehicle": .01, "urban_furniture": .003, "shade_structure": .01, "water": .01}
SUNNY_SKY_BLUE = .06      # mean (blue - red) of sky pixels: clear sky ~.1-.6, grey overcast ~0
PARTIAL_SHADOW = .10      # share of ground in cast shadow
MOSTLY_SHADED = .40
NIGHT_P = .70             # CLIP day/night; daytime photos scored <= .5 on the sample, night >= .79
# Junk filter threshold. 5-fold cross-validation on 801 reviewed images: rejecting below .7
# catches ~80% of the junk and loses ~12% of good images, affordable with 3x more images
# than the annotation set needs. Lower it if good images run short.
USABLE_P_MIN = .70
REVIEW_P = .85            # usable images below this junk-filter score are the borderline ones
KEEP_LEAF_OFF = 20        # leafless-tree images kept as a control set; the rest are rejected

# Attribute -> label -> prompts. Softmax runs over all prompts of one attribute and the
# probabilities are summed per label, so a label can be described several ways.
CLIP_ATTRIBUTES = {
    "scene": {
        "street": ["a street-level photo of a city street", "a photo of an urban avenue taken from a car"],
        "plaza": ["a photo of a public square or plaza"],
        "park": ["a photo of an urban park"],
        "highway_or_rural": ["a photo of a multi-lane freeway", "a photo of a country road surrounded by fields"],
        "not_streetscape": ["a photo of an indoor space", "a close-up photo of an object or a sign",
                            "a photo of a statue", "an aerial photo of a city", "a black image"],
    },
    "time": {
        "day": ["a photo taken during the day"],
        "night": ["a photo taken at night"],
    },
}
PROMPTS = [(attr, label, p) for attr, labels in CLIP_ATTRIBUTES.items() for label, ps in labels.items() for p in ps]
UNUSABLE_SCENES = {"highway_or_rural", "not_streetscape"}
FIELDS = ["image_id", *(f"frac_{g}" for g in GROUPS), "sky_blue", "ground_shadow",
          *(k for a in CLIP_ATTRIBUTES for k in (a, f"{a}_p")), "season", "tags", "usable_p", "reject_reason", "usable"]


def pixel_stats(rgb, mask, sky_ids, ground_ids):
    """rgb: HxWx3 floats in [0, 1] aligned with the class mask. Returns (sky_blue, ground_shadow,
    grayscale); the first two are nan when the region covers too little of the image to measure."""
    sky, ground = np.isin(mask, sky_ids), np.isin(mask, ground_ids)
    sky_blue = float((rgb[sky, 2] - rgb[sky, 0]).mean()) if sky.mean() >= .02 else math.nan
    # Shadowed ground is much darker than sunlit ground of the same material.
    lum = rgb[ground].max(axis=1)
    ground_shadow = float((lum < .5 * np.percentile(lum, 90)).mean()) if ground.mean() >= .03 else math.nan
    # Black-and-white photos have identical channels; grey overcast colour photos score >= .005.
    grayscale = bool(np.abs(rgb - rgb.mean(axis=2, keepdims=True)).mean() < .002)
    return sky_blue, ground_shadow, grayscale


def describe(class_frac, sky_blue, ground_shadow, grayscale, prompt_logits):
    """class_frac: {ADE20K class: pixel share}; prompt_logits: CLIP logits aligned with PROMPTS."""
    row = {f"frac_{g}": round(sum(class_frac.get(c, 0.0) for c in classes), 4) for g, classes in GROUPS.items()}
    row["sky_blue"], row["ground_shadow"] = round(sky_blue, 3), round(ground_shadow, 3)
    for attr in CLIP_ATTRIBUTES:
        idx = [i for i, (a, _, _) in enumerate(PROMPTS) if a == attr]
        top = max(prompt_logits[i] for i in idx)
        exps = {i: math.exp(prompt_logits[i] - top) for i in idx}
        scores = Counter()
        for i in idx:
            scores[PROMPTS[i][1]] += exps[i] / sum(exps.values())
        row[attr], p = scores.most_common(1)[0]
        row[f"{attr}_p"] = round(p, 3)

    if row["time"] == "night" and row["time_p"] >= NIGHT_P:
        lighting = "night"
    elif math.isnan(sky_blue):
        lighting = ""  # no sky visible: unknown
    else:
        lighting = "sunny" if sky_blue >= SUNNY_SKY_BLUE else "overcast"
    shadow = ""
    if lighting not in ("overcast", "night") and not math.isnan(ground_shadow) and ground_shadow >= PARTIAL_SHADOW:
        shadow = "mostly_shaded" if ground_shadow >= MOSTLY_SHADED else "partial_shadow"
    tags = [g for g, t in THRESHOLDS.items() if row[f"frac_{g}"] >= t] + [t for t in (lighting, shadow) if t]
    row["tags"] = ";".join(tags)
    if row["scene"] in UNUSABLE_SCENES:
        reason = row["scene"]
    elif lighting == "night":
        reason = "night"
    elif grayscale:
        reason = "grayscale"
    else:
        reason = ""
    row["reject_reason"], row["usable"] = reason, int(not reason)
    return row


def train_junk_filter(X, y):
    """Logistic regression on CLIP image embeddings X (n, d), y = 1 for junk.
    Returns (w, b) with P(junk) = sigmoid(X @ w + b)."""
    import torch

    X, y = torch.from_numpy(X), torch.from_numpy(y)
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=200)

    def loss():
        opt.zero_grad()
        # x10: embeddings are unit-length, so this keeps the logits in a well-conditioned range
        value = torch.nn.functional.binary_cross_entropy_with_logits(10 * X @ w + b, y) + 1e-3 * (w ** 2).sum()
        value.backward()
        return value

    opt.step(loss)
    return 10 * w.detach().numpy(), b.item()


def write_tags(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, top):
    usable = [r for r in rows if str(r["usable"]) == "1"]
    print(f"\n{len(rows)} images tagged, {len(usable)} usable")
    print("  rejected:        ", dict(Counter(r["reject_reason"] for r in rows if str(r["usable"]) == "0")))
    print("  usable by source:", dict(Counter(r["image_id"].split("_")[0] for r in usable)))
    print("  usable by season:", dict(Counter(r["season"] for r in usable)))
    scored = [r for r in usable if r.get("usable_p") not in (None, "")]
    if scored:
        print(f"  review first:     {sum(float(r['usable_p']) < REVIEW_P for r in scored)} usable images with"
              f" usable_p < {REVIEW_P} (sort tags.csv by usable_p)")
    if not usable:
        return
    print("\nTag frequency among usable images:")
    for tag, n in Counter(t for r in usable for t in r["tags"].split(";") if t).most_common():
        print(f"  {tag:18} {n:5}  {n / len(usable):6.1%}")
    print(f"\nTop {top} tag combinations among usable images:")
    for combo, n in Counter(r["tags"] for r in usable).most_common(top):
        print(f"  {n:5}  {n / len(usable):6.1%}  {combo.replace(';', ' + ')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--top", type=int, default=15, help="number of tag combinations to list")
    ap.add_argument("--prune", action="store_true", help="delete the image files of rejected images")
    args = ap.parse_args()

    out_path = DATA / "tags.csv"
    candidates = list(csv.DictReader((DATA / "candidates.csv").open()))
    files = {r["image_id"]: DATA / r["file"] for r in candidates}
    emb = dict(np.load(EMBEDDINGS)) if EMBEDDINGS.exists() else {}
    done = list(csv.DictReader(out_path.open())) if out_path.exists() else []
    # Images still on disk without an embedding (e.g. after an interrupted run) are re-tagged.
    done = [r for r in done if r["image_id"] in emb or not files[r["image_id"]].exists()]
    write_tags(out_path, done)  # current columns, so the rows appended below line up
    seen = {r["image_id"] for r in done}
    todo = [r for r in candidates if r["image_id"] not in seen and files[r["image_id"]].exists()]

    if todo:
        import torch  # imported here so select.py and the self-check load without it
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation, CLIPModel, CLIPProcessor

        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        # 576 px instead of the default 512: the stride-32 feature map (18x18) must divide by the
        # pyramid pool sizes (1, 2, 3, 6), otherwise adaptive pooling fails on Apple MPS.
        size = 576
        seg_proc = AutoImageProcessor.from_pretrained(SEG_MODEL, size={"height": size, "width": size})
        seg = AutoModelForSemanticSegmentation.from_pretrained(SEG_MODEL).to(device).eval()
        names = [seg.config.id2label[i].strip() for i in range(seg.config.num_labels)]
        sky_ids, ground_ids = [names.index("sky")], [names.index(c) for c in GROUND]
        clip_proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
        clip = CLIPModel.from_pretrained(CLIP_MODEL).to(device).eval()
        text = clip_proc(text=[p for _, _, p in PROMPTS], return_tensors="pt", padding=True).to(device)

        with out_path.open("a", newline="", buffering=1) as f:
            writer = csv.DictWriter(f, FIELDS)
            for start in range(0, len(todo), args.batch):
                batch = todo[start:start + args.batch]
                images = [Image.open(DATA / r["file"]).convert("RGB") for r in batch]
                with torch.no_grad():
                    masks = seg(**seg_proc(images=images, return_tensors="pt").to(device)).logits.argmax(1).cpu()
                    out = clip(**text, pixel_values=clip_proc(images=images, return_tensors="pt").pixel_values.to(device))
                    logits, embeds = out.logits_per_image.cpu(), out.image_embeds.float().cpu().numpy()
                for r, img, mask, lg, e in zip(batch, images, masks, logits, embeds):
                    emb[r["image_id"]] = e.astype(np.float16)
                    counts = torch.bincount(mask.flatten(), minlength=len(names)) / mask.numel()
                    rgb = np.asarray(img.resize((size, size)), dtype=np.float32) / 255
                    row = describe(dict(zip(names, counts.tolist())),
                                   *pixel_stats(rgb, mask.numpy(), sky_ids, ground_ids), lg.tolist())
                    row["image_id"], row["season"] = r["image_id"], season(r["capture_date"], float(r["lat"]))
                    writer.writerow(row)
                    done.append(row)
                print(f"tagged {start + len(batch)}/{len(todo)}", flush=True)
                if start // args.batch % 50 == 49 or start + args.batch >= len(todo):
                    np.savez(EMBEDDINGS, **emb)

    # Human review decisions override the automatic verdict and train the junk filter.
    review = {r["image_id"]: r["verdict"] for r in csv.DictReader(REVIEW.open())} if REVIEW.exists() else {}
    for r in done:
        if review.get(r["image_id"]) == "reject":
            r["reject_reason"], r["usable"] = "manual", 0
    labelled = [k for k in review if k in emb]
    if {review[k] for k in labelled} >= {"reject", "keep"}:
        w, b = train_junk_filter(np.stack([emb[k] for k in labelled]).astype(np.float32),
                                 np.array([review[k] == "reject" for k in labelled], dtype=np.float32))
        for r in done:
            if r["image_id"] not in emb:
                continue
            r["usable_p"] = round(float(1 / (1 + np.exp(emb[r["image_id"]].astype(np.float32) @ w + b))), 3)
            if str(r["usable"]) == "1" and r["usable_p"] < USABLE_P_MIN and review.get(r["image_id"]) != "keep":
                r["reject_reason"], r["usable"] = "junk_filter", 0
    # Keep a fixed pseudo-random KEEP_LEAF_OFF of the usable leaf-off images (ordered by a hash
    # of the id, so every run keeps the same ones) and reject the rest.
    leaf_off = sorted((r for r in done if r["season"] == "leaf_off" and str(r["usable"]) == "1"),
                      key=lambda r: hashlib.md5(r["image_id"].encode()).hexdigest())
    for r in leaf_off[KEEP_LEAF_OFF:]:
        r["reject_reason"], r["usable"] = "leaf_off", 0
    write_tags(out_path, done)

    if args.prune:
        pruned = [files[r["image_id"]] for r in done if str(r["usable"]) == "0" and files[r["image_id"]].exists()]
        for path in pruned:
            path.unlink()
        print(f"Pruned {len(pruned)} rejected images (each can be re-downloaded from image_url in candidates.csv).")
    summarize(done, args.top)


if __name__ == "__main__":
    main()
