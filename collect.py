"""Collect candidate street-level images for the heat-refuge dataset (project.md 4.2).

Neutral sampling: for every (source, city) pair we draw uniformly random points within
RADIUS_KM of the city centre, ask the source for photos near that point and keep one
at random. There is no keyword search, so nothing selects for "good" or "bad" heat
refuges. At most one image per capture sequence avoids near-duplicate dashcam frames.
Only photos taken in the leafy season (or in the tropics) are kept, see SEASONS.

With --stratum the points are instead OpenStreetMap features (see STRATA), each image
recording its stratum:
  pedestrian  pedestrian streets, squares and parks, which random points (mostly landing
              on car streets) rarely reach
  features    fountains, street-tree rows, canopies and shelters: the rare visual
              elements (water, shade structures, heavy shade) the tags show are scarce
Both select by objective place type or object, never by how good a refuge a place looks;
keep their share small and report them as separate strata.

Output (resumable: re-running fills each (source, city, stratum) up to --per-city images
on disk, so images pruned by tag.py get replaced):
  data/images/<image_id>.jpg   resized to MAX_SIDE px, EXIF (incl. GPS) stripped
  data/candidates.csv          one row per image, with license, attribution and stratum

Usage:
  python collect.py
  python collect.py --per-city 20 --sources panoramax wikimedia --cities Paris Madrid
  python collect.py --stratum features --sources panoramax wikimedia --per-city 15
"""
import argparse
import csv
import functools
import html
import json
import math
import os
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from PIL import Image, ImageOps

ROOT = Path(__file__).parent
DATA = ROOT / "data"
CSV_PATH = DATA / "candidates.csv"
RADIUS_KM = 5
MAX_SIDE = 1280
FIELDS = ["image_id", "source_platform", "source_id", "group_id", "city", "lat", "lon", "capture_date",
          "license", "license_url", "attribution", "page_url", "image_url", "file", "width", "height", "stratum"]
# Leafless trees make shaded streets look like poor heat refuges, so only these are collected.
SEASONS = {"leaf_on", "tropical"}
# Overpass QL selectors for the OpenStreetMap strata.
STRATA = {
    "pedestrian": ['way["highway"="pedestrian"]', 'way["place"="square"]', 'way["leisure"="park"]'],
    "features": ['node["amenity"="fountain"]', 'way["natural"="tree_row"]', 'way["building"="roof"]',
                 'node["amenity"="shelter"]', 'way["amenity"="shelter"]'],
}

# Centre coordinates. Spread over climates and continents; coverage differs per source,
# and a source that has nothing in a city simply gives up there (see --patience).
CITIES = {
    "Paris": (48.8566, 2.3522),
    "Lyon": (45.7640, 4.8357),
    "Marseille": (43.2965, 5.3698),
    "Bordeaux": (44.8378, -0.5792),
    "Madrid": (40.4168, -3.7038),
    "Barcelona": (41.3874, 2.1686),
    "Rome": (41.9028, 12.4964),
    "Athens": (37.9838, 23.7275),
    "Berlin": (52.5200, 13.4050),
    "Bucharest": (44.4268, 26.1025),
    "New York": (40.7128, -74.0060),
    "Detroit": (42.3314, -83.0458),
    "Phoenix": (33.4484, -112.0740),
    "Mexico City": (19.4326, -99.1332),
    "Sao Paulo": (-23.5505, -46.6333),
    "Delhi": (28.6139, 77.2090),
    "Singapore": (1.3521, 103.8198),
    "Tokyo": (35.6762, 139.6503),
    "Sydney": (-33.8688, 151.2093),
    "Cairo": (30.0444, 31.2357),
    # Hot-summer cities added after the first audit. Outside France almost only Wikimedia
    # has summer imagery here; Dubai, Riyadh and Las Vegas had too little to be worth it.
    "Seville": (37.3891, -5.9845),
    "Valencia": (39.4699, -0.3763),
    "Montpellier": (43.6108, 3.8767),
    "Istanbul": (41.0082, 28.9784),
    "Tel Aviv": (32.0853, 34.7818),
    "Los Angeles": (34.0522, -118.2437),
    "Houston": (29.7604, -95.3698),
    "Austin": (30.2672, -97.7431),
    "Rio de Janeiro": (-22.9068, -43.1729),
    "Buenos Aires": (-34.6037, -58.3816),
    "Mumbai": (19.0760, 72.8777),
    "Bangkok": (13.7563, 100.5018),
    "Perth": (-31.9505, 115.8605),
}

USER_AGENT = "heat-refuge-dataset/0.1"
LOCK = threading.Lock()
PLACES_LOCK = threading.Lock()  # one Overpass query at a time: its rate limit is two per IP


def load_env(path=ROOT / ".env"):
    if path.exists():
        for line in path.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and not key.lstrip().startswith("#"):
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def get(url, binary=False, tries=3, timeout=60):
    """GET with backoff on network errors, 429 and 5xx. Other HTTP errors raise at once."""
    for attempt in range(tries):
        try:
            with urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=timeout) as resp:
                body = resp.read()
            return body if binary else json.loads(body)
        except HTTPError as e:
            if (e.code < 500 and e.code != 429) or attempt == tries - 1:
                raise
        except OSError:
            if attempt == tries - 1:
                raise
        time.sleep(5 * 2 ** attempt)


def is_open_license(name):
    """CC0, public domain, CC BY, CC BY-SA or Etalab (CC BY compatible). Rejects NC/ND."""
    name = name.lower()
    return bool(re.match(r"(cc0|cc[- ]by|public domain|pd\b|etalab)", name)) and not re.search(r"\bn[cd]\b", name)


def random_point(rng, lat, lon, radius_km):
    """Uniform point in a disc around (lat, lon); flat-earth approximation is fine at city scale."""
    r = radius_km * math.sqrt(rng.random()) / 111.32
    theta = rng.uniform(0, 2 * math.pi)
    return lat + r * math.sin(theta), lon + r * math.cos(theta) / math.cos(math.radians(lat))


def season(capture_date, lat):
    """Foliage season at capture time: leaf_on (May-Sep in the north), leaf_off (Nov-Mar),
    shoulder (Apr, Oct), tropical, or unknown when the date has no month."""
    # ponytail: latitude only; ignores evergreen and dry-season climates outside the tropics.
    if abs(lat) < 23.5:
        return "tropical"
    month = re.match(r"\d{4}-(\d{2})", capture_date)
    if not month:
        return "unknown"
    m = int(month.group(1))
    if lat < 0:
        m = (m + 5) % 12 + 1  # southern hemisphere: shift six months
    return "leaf_on" if 5 <= m <= 9 else "shoulder" if m in (4, 10) else "leaf_off"


@functools.lru_cache
def place_points(lat, lon, stratum):
    """Locations of the stratum's OpenStreetMap features within RADIUS_KM (way centres, node positions)."""
    around = f"(around:{RADIUS_KM * 1000},{lat},{lon});"
    query = f"[out:json][timeout:180];({''.join(sel + around for sel in STRATA[stratum])});out skel center qt;"
    data = get("https://overpass-api.de/api/interpreter?" + urlencode({"data": query}), timeout=200)
    return [(e["center"]["lat"], e["center"]["lon"]) if "center" in e else (e["lat"], e["lon"])
            for e in data["elements"] if "center" in e or "lat" in e]


def strip_html(text):
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", text)).split())


def search_panoramax(lat, lon):
    d = 0.001  # ~110 m half-side
    dl = d / math.cos(math.radians(lat))
    rows = []
    for f in get(f"https://api.panoramax.xyz/api/search?bbox={lon-dl},{lat-d},{lon+dl},{lat+d}&limit=50")["features"]:
        p = f["properties"]
        # ponytail: 360° panoramas are skipped rather than reprojected to a perspective
        # view; add an equirectangular crop if flat images run short.
        if ((p.get("pers:interior_orientation") or {}).get("field_of_view") == 360
                or (p.get("exif") or {}).get("Xmp.GPano.ProjectionType") == "equirectangular"):
            continue
        links = {link["rel"]: link["href"] for link in f["links"]}
        rows.append({
            "source_id": f["id"], "group_id": f["collection"],
            "lat": f["geometry"]["coordinates"][1], "lon": f["geometry"]["coordinates"][0],
            "capture_date": p.get("datetime", "")[:10],
            "license": p.get("license", ""), "license_url": links.get("license", ""),
            "attribution": ", ".join(pr["name"] for pr in f.get("providers", [])),
            "page_url": links.get("self", ""), "image_url": f["assets"]["sd"]["href"],
        })
    return rows


def search_kartaview(lat, lon):
    # The API times out on anything much larger than a 50 m radius.
    data = get(f"https://api.openstreetcam.org/2.0/photo/?lat={lat}&lng={lon}&radius=50&itemsPerPage=50")
    return [{
        "source_id": p["id"], "group_id": p["sequenceId"],
        "lat": float(p["lat"]), "lon": float(p["lng"]), "capture_date": p["shotDate"][:10],
        # All KartaView imagery is CC BY-SA 4.0 with this credit line (kartaview.org/terms).
        "license": "CC-BY-SA-4.0", "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "attribution": "© Grab and KartaView Contributors",
        "page_url": f"https://api.openstreetcam.org/2.0/photo/{p['id']}", "image_url": p["imageLthUrl"],
    } for p in (data.get("result") or {}).get("data", [])  # "result" is missing when nothing is near
        if p["projection"] == "PLANE" and p["status"] == "active" and p["visibility"] == "public"]


def search_wikimedia(lat, lon):
    query = urlencode({
        "action": "query", "format": "json", "formatversion": 2,
        "generator": "geosearch", "ggscoord": f"{lat}|{lon}", "ggsradius": 200, "ggslimit": 50, "ggsnamespace": 6,
        "prop": "imageinfo|coordinates|categories", "colimit": 50, "iiprop": "url|extmetadata|size|mime",
        "iiurlwidth": MAX_SIDE, "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Artist|DateTimeOriginal",
        # Only this category is returned, so a page with "categories" is a 360° panorama. These are
        # exactly 2:1 like ordinary 18:9 phone photos, so the aspect check below can't catch them.
        "clcategories": "Category:360° panoramas", "cllimit": "max",
    })
    rows = []
    for page in get(f"https://commons.wikimedia.org/w/api.php?{query}").get("query", {}).get("pages", []):
        if "imageinfo" not in page or "coordinates" not in page or "categories" in page:
            continue
        info = page["imageinfo"][0]
        w, h = info["width"], info["height"]
        if info["mime"] != "image/jpeg" or min(w, h) < 600 or max(w, h) > 2 * min(w, h):  # also drops wide panoramas
            continue
        meta = {k: strip_html(v["value"]) for k, v in info.get("extmetadata", {}).items()}
        date = re.search(r"\d{4}(-\d{2}){0,2}", meta.get("DateTimeOriginal", ""))
        if date and int(date.group()[:4]) < 2000:  # historic photos show a different city
            continue
        rows.append({
            "source_id": str(page["pageid"]), "group_id": str(page["pageid"]),
            "lat": page["coordinates"][0]["lat"], "lon": page["coordinates"][0]["lon"],
            "capture_date": date.group() if date else "",
            "license": meta.get("LicenseShortName", ""), "license_url": meta.get("LicenseUrl", ""),
            "attribution": meta.get("Artist", ""),
            "page_url": info["descriptionurl"], "image_url": info["thumburl"],
        })
    return rows


SEARCH = {"panoramax": search_panoramax, "kartaview": search_kartaview, "wikimedia": search_wikimedia}


def save_image(url, path):
    img = ImageOps.exif_transpose(Image.open(BytesIO(get(url, binary=True)))).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    img.save(path, quality=90)  # no exif= argument, so metadata (incl. GPS) is dropped
    return img.size


def collect(source, city, args, taken, writer, have):
    stratum = args.stratum
    rng = random.Random(f"{args.seed}:{source}:{city}" + ("" if stratum == "random" else f":{stratum}"))
    lat0, lon0 = CITIES[city]
    if stratum != "random":
        try:
            with PLACES_LOCK:
                places = place_points(lat0, lon0, stratum)
        except Exception as e:
            places = []
            print(f"[{source}/{city}] OpenStreetMap query failed: {type(e).__name__}: {e}", file=sys.stderr)
        if not places:
            print(f"{source:10} {city:12} skipped: no OpenStreetMap {stratum} features", flush=True)
            return
    misses = 0
    while have < args.per_city and misses < args.patience:
        lat, lon = random_point(rng, lat0, lon0, RADIUS_KM) if stratum == "random" else rng.choice(places)
        try:
            rows = [r for r in SEARCH[source](lat, lon)
                    if is_open_license(r["license"]) and season(r["capture_date"], r["lat"]) in SEASONS]
        except Exception as e:
            print(f"[{source}/{city}] search failed: {type(e).__name__}: {e}", file=sys.stderr)
            misses += 1
            continue
        for r in rows:
            r["image_id"] = f"{source}_{r['source_id']}"
            r["group_key"] = f"{source}:{r['group_id']}"
        with LOCK:
            rows = [r for r in rows if r["image_id"] not in taken and r["group_key"] not in taken]
            if not rows:
                misses += 1
                continue
            row = rng.choice(rows)
            taken.update((row["image_id"], row["group_key"]))
        file = f"images/{row['image_id']}.jpg"
        try:
            row["width"], row["height"] = save_image(row["image_url"], DATA / file)
        except Exception as e:
            print(f"[{source}/{city}] download failed {row['image_url']}: {type(e).__name__}: {e}", file=sys.stderr)
            misses += 1
            continue
        row.update(source_platform=source, city=city, file=file, stratum=stratum)
        with LOCK:
            writer.writerow(row)
        have += 1
        misses = 0
        if have % 10 == 0:
            print(f"{source:10} {city:12} {have}/{args.per_city}", flush=True)
    print(f"{source:10} {city:12} done: {have}/{args.per_city}" + ("" if have >= args.per_city else " (gave up)"), flush=True)


def main():
    global USER_AGENT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", choices=list(SEARCH), default=list(SEARCH))
    ap.add_argument("--cities", nargs="+", choices=list(CITIES), default=list(CITIES))
    ap.add_argument("--per-city", type=int, default=60, help="images per (source, city)")
    ap.add_argument("--patience", type=int, default=40, help="consecutive misses before giving up a (source, city)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stratum", choices=["random", *STRATA], default="random",
                    help="where the sample points come from (see the module docstring)")
    args = ap.parse_args()

    load_env()
    contact = os.environ.get("CONTACT_EMAIL", "").strip()
    if "wikimedia" in args.sources and not contact:
        sys.exit("Set CONTACT_EMAIL in .env: the Wikimedia API policy requires contact info in the User-Agent.")
    USER_AGENT = f"heat-refuge-dataset/0.1 (UMich MSAI course project; {contact or 'no contact given'})"

    (DATA / "images").mkdir(parents=True, exist_ok=True)
    done = list(csv.DictReader(CSV_PATH.open())) if CSV_PATH.exists() else []
    if done and "stratum" not in done[0]:  # file from before strata existed: all random sampling
        for r in done:
            r["stratum"] = "random"
        with CSV_PATH.open("w", newline="") as f:
            writer = csv.DictWriter(f, FIELDS)
            writer.writeheader()
            writer.writerows(done)
    # Every id ever collected stays taken, so pruned images are not downloaded again,
    # but only images still on disk count towards --per-city.
    taken = {r["image_id"] for r in done} | {f"{r['source_platform']}:{r['group_id']}" for r in done}
    counts = Counter((r["source_platform"], r["city"]) for r in done
                     if r["stratum"] == args.stratum and (DATA / r["file"]).exists())

    # Line-buffered, so every row hits disk at once and an interrupted run loses nothing.
    with CSV_PATH.open("a", newline="", buffering=1) as f:
        writer = csv.DictWriter(f, FIELDS, extrasaction="ignore")
        if not done:
            writer.writeheader()
        # Interleave sources so the workers spread over the three hosts.
        pairs = [(s, c) for c in args.cities for s in args.sources]
        with ThreadPoolExecutor(args.workers) as pool:
            for fut in [pool.submit(collect, s, c, args, taken, writer, counts[s, c]) for s, c in pairs]:
                fut.result()


if __name__ == "__main__":
    main()
