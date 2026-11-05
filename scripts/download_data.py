"""Download pinned public source files with resume and checksum verification.

Uses Python's standard library; the local base Python can run it independently
from the training interpreter. Never disables HTTPS certificate validation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://huggingface.co/datasets/mohanty/PlantVillage"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, destination, expected_size=None, sha256=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        actual = digest(destination)
        if (expected_size is None or destination.stat().st_size == expected_size) and (
            sha256 is None or actual == sha256
        ):
            print(f"Already verified: {destination.name}", flush=True)
            return actual
        raise RuntimeError(f"Existing file failed verification: {destination}")
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(6):
        offset = partial.stat().st_size if partial.exists() else 0
        if expected_size is not None and offset == expected_size:
            break
        headers = {"User-Agent": "PlantLeafCNN-coursework/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            # Refresh redirect URL each retry; signed CDN links may expire.
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                if offset and response.status != 206:
                    offset = 0
                if response.status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise RuntimeError(f"Unexpected range response: {content_range}")
                mode = "ab" if offset else "wb"
                start = time.perf_counter()
                last_report = start
                done = offset
                with partial.open(mode) as f:
                    while True:
                        chunk = response.read(2 * 1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        now = time.perf_counter()
                        if now - last_report >= 10:
                            speed = (done - offset) / (now - start) / 2**20
                            print(f"{destination.name}: {done / 2**20:.1f} MiB, {speed:.1f} MiB/s", flush=True)
                            last_report = now
            if expected_size is not None and partial.stat().st_size != expected_size:
                raise RuntimeError(f"Incomplete file: {partial.stat().st_size} != {expected_size}")
            break
        except (OSError, urllib.error.URLError, RuntimeError) as exc:
            print(f"Download attempt {attempt + 1}: {exc}", flush=True)
            if attempt == 5:
                raise
            time.sleep(min(2 ** attempt, 10))
    actual = digest(partial)
    if sha256 is not None and actual != sha256:
        raise RuntimeError(f"SHA256 mismatch for {destination.name}: {actual}")
    partial.replace(destination)
    print(f"Verified {destination.name}: {actual}", flush=True)
    return actual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    raw = ROOT / "data" / "downloads"
    raw.mkdir(parents=True, exist_ok=True)
    record_path = raw / "source_record.json"
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
    elif (ROOT / "configs/dataset_source.json").exists():
        # 从公开实验配置恢复固定数据版本，避免新环境默默下载更新后的版本。
        record = json.loads((ROOT / "configs/dataset_source.json").read_text(encoding="utf-8"))
        record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    else:
        with urllib.request.urlopen("https://huggingface.co/api/datasets/mohanty/PlantVillage", timeout=30) as f:
            info = json.load(f)
        revision = info["sha"]
        with urllib.request.urlopen(f"https://huggingface.co/api/datasets/mohanty/PlantVillage/tree/{revision}?recursive=true&limit=1000", timeout=30) as f:
            tree = json.load(f)
        record = {"repository": REPO, "revision": revision,
                  "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "tree": tree, "files": {}}
        record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    names = ["README.md", "plant_village.py", "leaf_grouping/leaf-map.json",
             "splits/color_train.txt", "splits/color_test.txt"]
    if not args.metadata_only:
        names.append("data.zip")
    entries = {entry["path"]: entry for entry in record["tree"]}
    for name in names:
        entry = entries[name]
        url = f"{REPO}/resolve/{record['revision']}/{name}"
        checksum = download(url, raw / name, entry["size"], entry.get("lfs", {}).get("oid"))
        record["files"][name] = {"url": url, "size": (raw / name).stat().st_size, "sha256": checksum}
        record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("Download complete.", flush=True)


if __name__ == "__main__":
    main()
