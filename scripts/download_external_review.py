"""下载具有明确来源与许可的外部展示图片，并保存文件及像素指纹。"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT/"assets/external_review_sources.json"
DESTINATION = ROOT/"data/external_review_20261109"
SELECTION = [
    ("E1", "File:Apple scab.jpg", "海棠黑星病", "related_leaf"),
    ("E2", "File:Healthy tomato leaves (7871755330).jpg", "番茄叶片", "covered_crop_leaf"),
    ("E3", "File:Guignardia bidwellii (black rot) on grape 1.jpg", "葡萄黑腐病叶片", "covered_crop_leaf"),
    ("E4", "File:Northern corn leaf blight.JPG", "玉米北方叶枯病", "covered_crop_leaf"),
    ("E5", "File:Acer palmatum atropurpureum, Rentilly (leaves).jpg", "鸡爪槭叶片", "unknown_crop_leaf"),
    ("E6", "File:Basketball ball385428 9836.jpg", "篮球", "non_leaf"),
]


def plain(value):
    return html.unescape(re.sub(r"<[^>]+>", "", value)).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", help="可选HTTP代理地址")
    args = parser.parse_args()
    session = requests.Session()
    session.headers["User-Agent"] = "PlantLeafCoursework/1.0 (educational image attribution audit)"
    retry = Retry(total=3, backoff_factor=5, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET",), respect_retry_after_header=True)
    session.mount("https://", HTTPAdapter(max_retries=retry))
    if args.proxy:
        session.proxies.update(http=args.proxy, https=args.proxy)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        records = manifest["images"]
    else:
        response = session.get("https://commons.wikimedia.org/w/api.php", params={
            "action": "query", "format": "json", "prop": "imageinfo", "iiprop": "url|extmetadata|sha1|timestamp",
            "titles": "|".join(row[1] for row in SELECTION)}, timeout=60)
        response.raise_for_status()
        pages = {page["title"]: page for page in response.json()["query"]["pages"].values()}
        records = []
        for identifier, title, description, kind in SELECTION:
            info = pages[title]["imageinfo"][0]
            metadata = info["extmetadata"]
            records.append({"id": identifier, "title": title, "description_zh": description, "kind": kind,
                            "filename": identifier+".jpg", "page_url": info["descriptionurl"], "download_url": info["url"],
                            "author": plain(metadata.get("Artist", {}).get("value", "")),
                            "license": plain(metadata.get("LicenseShortName", {}).get("value", "")),
                            "license_url": plain(metadata.get("LicenseUrl", {}).get("value", "")),
                            "source_timestamp": info["timestamp"], "commons_sha1": info["sha1"]})
        MANIFEST.write_text(json.dumps({"images": records, "download_complete": False}, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    for record in records:
        destination = DESTINATION/record["filename"]
        if not destination.exists():
            response = session.get(record["download_url"], timeout=90)
            response.raise_for_status()
            destination.write_bytes(response.content)
            time.sleep(2)
        raw = destination.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if record.get("sha256") and record["sha256"] != digest:
            raise RuntimeError("Downloaded external image differs from the pinned source")
        if hashlib.sha1(raw).hexdigest() != record["commons_sha1"]:
            raise RuntimeError("Downloaded file does not match Commons SHA1")
        with Image.open(destination) as source:
            image = source.convert("RGB")
            pixels = np.asarray(image)
            record["width"], record["height"] = image.size
            record["pixel_sha256"] = hashlib.sha256(str(pixels.shape).encode()+pixels.tobytes()).hexdigest()
        record["sha256"] = digest
        print(json.dumps({key: record[key] for key in ("id", "license", "author", "width", "height")}, ensure_ascii=False), flush=True)
    pixel_sources = {}
    for split in ("train", "val", "test", "quarantine"):
        with (ROOT/"data/manifests"/f"{split}.csv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                pixel_sources.setdefault(row["pixel_sha256"], []).append({"split": split, "path": row["path"]})
    for record in records:
        record["identical_plantvillage_pixels"] = pixel_sources.get(record["pixel_sha256"], [])
        if record["identical_plantvillage_pixels"]:
            raise RuntimeError("Selected external case exactly duplicates PlantVillage pixels")
    manifest = {"downloaded_utc": datetime.now(timezone.utc).isoformat(), "images": records, "download_complete": True,
                "purpose": "Exploratory external-source top5 and rejection demonstration; source descriptions are not expert-verified disease ground truth.",
                "processing": "Downloaded originals retained byte-for-byte; model uses the same RGB resize and normalization as CLI.",
                "selection_frozen_before_prediction": True}
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
