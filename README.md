# Heat-refuge streetscapes: image collection and initial tagging

Implements section 4.2 of `project.md`: build a large, visually diverse pool of
openly licensed street-level images, then tag them automatically to see what the pool
is made of. The tags are **not** heat-refuge labels; those come from the human annotators.

## Setup

```bash
uv pip install -r requirements.txt     # or: pip install -r requirements.txt
cp .env.example .env                   # then fill in CONTACT_EMAIL (needed for Wikimedia)
python test_pipeline.py                # prints "ok"
```

No API keys are needed. Panoramax, KartaView and Wikimedia Commons are open for reading.
The two models (~2 GB) download from Hugging Face on the first `tag.py` run.

## 1. Collect: `python collect.py`

For each (source, city) pair it draws random points within 5 km of the city centre,
asks the source for photos near each point and keeps one at random. No keyword search
is involved, so nothing favours places that look like good or bad refuges.
It only keeps summer photos (leafy months, or any month in the tropics), skips 360°
panoramas and photos taken before 2000, takes at most one image per capture sequence
(dashcam frames are near-duplicates) and accepts only CC0 / public domain / CC BY /
CC BY-SA / Etalab images.

- `data/images/<image_id>.jpg`: max 1280 px, EXIF and GPS stripped
- `data/candidates.csv`: source, city, coordinates, capture date, **license, license URL, attribution**, source page

Defaults: 33 cities (13 of them hot-summer cities added after the first audit) × 3 sources
× 60 images on disk. Images pruned by `tag.py` free their slot, so re-running refills it.
A source with no coverage in a city gives up after `--patience` misses. KartaView is slow
(15–25 s per query), so run it separately: `python collect.py --sources kartaview --workers 8`.
Runs are resumable: interrupt at any time and re-run.

### Rare scene types: `--stratum`

Random points mostly land on car streets, so shade structures (1%), water (3%) and plazas
are scarce. Two OpenStreetMap strata sample at specific places instead (`STRATA` in
`collect.py`):

| `--stratum` | Points at | Pilot effect (4 cities, vs random) |
|---|---|---|
| `pedestrian` | pedestrian streets, squares, parks | fewer cars (35% vs 47%), more plazas; rare tags unchanged |
| `features` | fountains, street-tree rows, canopies, shelters | shade structures 6% vs 2%, water 15% vs 6%, parks 21% vs 14% |

```bash
python collect.py --stratum features   --sources panoramax wikimedia --per-city 15
python collect.py --stratum pedestrian --sources panoramax wikimedia --per-city 15
```

They choose by place type or object, never by how good a refuge a place looks, but they
do shift the pool towards those elements. Keep them a minority and report the strata
separately: every image records its `stratum` (random / pedestrian / features).

## 2. Tag: `python tag.py`

| Column | From | Meaning |
|---|---|---|
| `frac_tree`, `frac_open_sky`, `frac_street`, … | UperNet-ConvNeXt, ADE20K segmentation (MIT) | share of pixels covered |
| `sky_blue` | pixels of the sky mask | blue minus red; ≥ 0.06 means sunny, below means overcast, empty means no sky visible |
| `ground_shadow` | pixels of the road/sidewalk mask | share of the ground in cast shadow |
| `scene`, `time` | CLIP ViT-L/14 zero-shot (MIT) | street / plaza / park / highway_or_rural / not_streetscape; day / night |
| `usable_p` | junk filter: logistic regression on CLIP embeddings, trained on `data/review.csv` | probability the image is a usable streetscape; below 0.7 it is rejected |
| `season` | capture month + latitude | leaf_on (May–Sep north / Nov–Mar south), leaf_off, shoulder (Apr/Oct), tropical, unknown |
| `tags` | thresholds on the above | e.g. `tree;sidewalk;building;sunny;partial_shadow` |
| `reject_reason`, `usable` | | not_streetscape, highway_or_rural, night, grayscale (black-and-white photo), manual, junk_filter, leaf_off (all but a fixed control set of 20) |

It then prints the tag frequencies and the most common tag combinations among usable
images. Those are the over-represented groups to thin out, or to balance by collecting
more of the rarer ones.

## 3. Junk filter, review and prune

CLIP prompts alone let through a lot of junk that still looks "urban": skylines across
water, crowds, close-ups, station platforms. The junk filter is trained on human
decisions in `data/review.csv` (`image_id,verdict,note`, verdict `reject` or `keep`) and
retrained on every `tag.py` run. Compared on 801 reviewed images (5-fold cross-validation):

| Junk detector | AUC | At its threshold |
|---|---|---|
| CLIP zero-shot scene prompts (before) | – | caught none of the junk that reached the selection |
| Qwen3-VL-2B / 4B asked directly (zero-shot) | 0.77 / 0.79 | 19% caught, or 87% caught but 38% of good images lost |
| **Logistic regression on CLIP-L embeddings** (used) | **0.92** | **~80% of junk caught, ~12% of good images lost** |
| same on SigLIP 2 / CLIP-H embeddings | 0.93 / 0.92 | no real gain over CLIP-L |

Losing ~12% of good images is affordable because the pool is 2–3× the annotation set.
`USABLE_P_MIN` in `tag.py` sets the trade-off.

To review: open the usable images with the lowest `usable_p` first (the summary says how
many are below 0.85), add a row per decision to `data/review.csv`, then:

```bash
python tag.py --prune     # retrains the filter, applies the review, deletes rejected image files
python pick.py            # replaces removed images in the selection, syncs data/final_images
```

`keep` rows protect an image from the filter, and every row makes the filter better.
Rows stay in `candidates.csv` and `tags.csv`, so rejected images aren't collected again and
the reasons stay on record. Any image can be re-downloaded from its `image_url`.
`data/clip_embeddings.npz` holds the embeddings of every tagged image, pruned ones
included; the filter trains on them, so don't delete it.

## 4. Pick the annotation set: `python pick.py --n 1000`

Writes `data/selected.csv` (metadata, license, attribution, stratum and tags per image) and
copies the selected images into `data/final_images/`, removing any no longer selected.
Each image's draw is fixed by a hash of its id, so after a review only the rejected images
are swapped out:
1. KartaView is capped at 40% of each city. Its dashcam viewpoint from the traffic lane
   would otherwise let a model learn source instead of scene.
2. The rest is drawn by weighted sampling, each image weighted by 1 / frequency of its
   rarest tag or scene type, so rare compositions are over-represented in the set.

Thresholds were set by eye on a 30-image sample (`THRESHOLDS` and the constants below it
in `tag.py`). Change them and re-derive the tags from the raw columns; no model re-run needed.

## Known limitations

- Sunny/overcast is unknown when no sky is visible, and backlit shots with blown-out
  white skies read as overcast.
- `ground_shadow` catches large shaded areas but misses thin shadows.
- Panoramax coverage is mostly France; elsewhere the mix leans on KartaView and Wikimedia.
- Wikimedia geotagged photos lean towards landmarks and include many non-street photos
  (45% of the first collection was rejected).
- Leafless trees make tree-lined streets look like poor heat refuges, so collection is
  summer-only and only 20 leaf-off images are kept as a control set.
