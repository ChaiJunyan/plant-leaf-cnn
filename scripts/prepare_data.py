"""Audit color images and create grouped manifests; never score the test set."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import random
import time
import zipfile

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        self.parent[self.find(b)] = self.find(a)


def leaf_id(path, leaf_map):
    parts = path.split("/")
    cls, filename = parts[2], parts[3]
    identifier = filename.replace("_final_masked", "").split("___")[-1].split("copy")[0]
    identifier = identifier.replace(".jpg", "").replace(".JPG", "").replace(".png", "").replace(".PNG", "").strip()
    choices = leaf_map.get(identifier.lower().strip(), [])
    if len(choices) == 1:
        return choices[0], True
    for choice in choices:
        if cls in choice:
            return choice, True
    # Namespace fallback IDs by class to avoid merging unrelated naming systems.
    return f"fallback::{cls}::{identifier.lower()}", False


def json_write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--val-fraction", type=float, default=0.125,
                        help="Fraction of retained recommended train pool, by grouped image count")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    download = ROOT / "data/downloads"
    data_root = ROOT / "data/images"
    output = ROOT / "data/manifests"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "audit.json").exists():
        raise RuntimeError("Manifests already exist. Audit before making a new split version; refusing silent overwrite.")
    leaf_map = json.loads((download / "leaf_grouping/leaf-map.json").read_text(encoding="utf-8"))
    original = {split: (download / f"splits/color_{split}.txt").read_text(encoding="utf-8").splitlines()
                for split in ("train", "test")}
    paths = original["train"] + original["test"]
    if len(paths) != len(set(paths)):
        raise RuntimeError("Recommended manifests contain repeated paths")
    expected = set(paths)
    archive = download / "data.zip"
    print("Extracting only raw/color images...", flush=True)
    data_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        members = {n: n for n in z.namelist() if n in expected}
        missing = expected - members.keys()
        if missing:
            raise RuntimeError(f"Missing {len(missing)} expected image paths in archive: {list(missing)[:3]}")
        extra_color = [n for n in z.namelist() if n.startswith("raw/color/") and not n.endswith("/") and n not in expected]
        for count, path in enumerate(paths):
            dest = (data_root / path).resolve()
            if not dest.is_relative_to(data_root.resolve()):
                raise RuntimeError(f"Unsafe archive path: {path}")
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(path) as source, dest.open("wb") as target:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
            if (count + 1) % 10000 == 0:
                print(f"Extracted {count + 1}/{len(paths)}", flush=True)
    classes = sorted({p.split("/")[2] for p in paths})
    class_to_idx = {c: i for i, c in enumerate(classes)}
    train_paths = set(original["train"])

    def inspect(path):
        group, mapped = leaf_id(path, leaf_map)
        with Image.open(data_root / path) as source:
            source.load()
            image = source.convert("RGB")
            pixels = np.asarray(image)
            pixel_hash = hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()
            gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
            gray_arr = np.asarray(gray)
            bits = (gray_arr[:, 1:] > gray_arr[:, :-1]).flatten()
            dhash = int.from_bytes(np.packbits(bits).tobytes(), "big")
            small = np.asarray(image.resize((32, 32), Image.Resampling.BILINEAR)).astype(np.float32) / 255.0
            return {"path": path, "class_name": path.split("/")[2], "label": class_to_idx[path.split("/")[2]],
                    "leaf_id": group, "mapped_leaf": mapped, "pixel_sha256": pixel_hash,
                    "dhash": dhash, "width": image.width, "height": image.height,
                    "original_split": "train" if path in train_paths else "test",
                    "rgb_mean": small.mean((0, 1)), "rgb_sqmean": (small * small).mean((0, 1))}

    print("Decoding every image and checking grouping / exact duplicates...", flush=True)
    rows = []
    with ThreadPoolExecutor(args.workers) as ex:
        for row in ex.map(inspect, paths):
            rows.append(row)
            if len(rows) % 10000 == 0:
                print(f"Audited {len(rows)}/{len(paths)}", flush=True)
    uf = UnionFind(len(rows))
    first_leaf, first_hash = {}, {}
    for i, row in enumerate(rows):
        for key, lookup in ((row["leaf_id"], first_leaf), (row["pixel_sha256"], first_hash)):
            if key in lookup:
                uf.union(i, lookup[key])
            else:
                lookup[key] = i
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[uf.find(i)].append(i)
    # Retain the recommended test pool; quarantine train images connected to it.
    quarantine = []
    training_groups = defaultdict(list)
    mixed_label_groups = []
    original_cross_groups = []
    for indexes in groups.values():
        labels = {rows[i]["label"] for i in indexes}
        if len(labels) != 1:
            mixed_label_groups.append([rows[i]["path"] for i in indexes])
        splits = {rows[i]["original_split"] for i in indexes}
        if len(splits) > 1:
            original_cross_groups.append([rows[i]["path"] for i in indexes])
        gid = hashlib.sha256("\n".join(sorted(rows[i]["path"] for i in indexes)).encode()).hexdigest()[:20]
        if "test" in splits:
            for i in indexes:
                rows[i]["split"] = "test" if rows[i]["original_split"] == "test" else "quarantine"
                if rows[i]["split"] == "quarantine":
                    quarantine.append(rows[i])
                rows[i]["group_id"] = gid
        elif len(labels) == 1:
            training_groups[next(iter(labels))].append((gid, indexes))
        else:
            for i in indexes:
                rows[i]["split"] = "quarantine"
                rows[i]["group_id"] = gid
                quarantine.append(rows[i])
    rng = random.Random(args.seed)
    for label in sorted(training_groups):
        grouped = sorted(training_groups[label])
        rng.shuffle(grouped)
        target = sum(len(ids) for _, ids in grouped) * args.val_fraction
        selected = 0
        for group_idx, (gid, indexes) in enumerate(grouped):
            split = "val" if selected < target and group_idx < len(grouped) - 1 else "train"
            if split == "val":
                selected += len(indexes)
            for i in indexes:
                rows[i]["split"] = split
                rows[i]["group_id"] = gid
    # A three-band dHash index finds all Hamming<=2 pairs. Report candidates;
    # do not silently treat perceptual similarity as verified duplicate identity.
    print("Screening perceptual near-duplicate candidates...", flush=True)
    buckets = defaultdict(list)
    near_pairs = []
    cross_split_pairs = []
    for i, row in enumerate(rows):
        value = row["dhash"]
        candidates = set()
        for band, (shift, width) in enumerate(((0, 22), (22, 21), (43, 21))):
            candidates.update(buckets[(band, (value >> shift) & ((1 << width) - 1))])
        for j in candidates:
            other = rows[j]
            if row["group_id"] == other["group_id"]:
                continue
            distance = (value ^ other["dhash"]).bit_count()
            if distance <= 2:
                pair = {"path_a": other["path"], "path_b": row["path"], "distance": distance,
                        "split_a": other["split"], "split_b": row["split"],
                        "same_label": row["label"] == other["label"]}
                near_pairs.append(pair)
                if row["split"] != other["split"] and "quarantine" not in (row["split"], other["split"]):
                    cross_split_pairs.append(pair)
        for band, (shift, width) in enumerate(((0, 22), (22, 21), (43, 21))):
            buckets[(band, (value >> shift) & ((1 << width) - 1))].append(i)
    fields = ["path", "label", "class_name", "leaf_id", "group_id", "pixel_sha256", "original_split", "width", "height"]
    split_rows = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test", "quarantine")}
    for split, subset in split_rows.items():
        write_csv(output / f"{split}.csv", subset, fields)
    # Verify disjointness for both physical-leaf and decoded-pixel identity.
    for key in ("leaf_id", "pixel_sha256", "group_id", "path"):
        seen = {}
        for split in ("train", "val", "test"):
            for row in split_rows[split]:
                if row[key] in seen and seen[row[key]] != split:
                    raise AssertionError(f"Cross-split {key}: {row[key]}")
                seen[row[key]] = split
    for split in ("train", "val", "test"):
        if len({r["label"] for r in split_rows[split]}) != len(classes):
            raise RuntimeError(f"Missing class in {split}")
    train = split_rows["train"]
    mean = np.stack([r["rgb_mean"] for r in train]).mean(0)
    std = np.sqrt(np.maximum(np.stack([r["rgb_sqmean"] for r in train]).mean(0) - mean ** 2, 1e-12))
    json_write(output / "classes.json", classes)
    json_write(output / "normalization.json", {"mean": mean.tolist(), "std": std.tolist(),
               "source": "training split only, all images resized to 32x32 RGB for channel moment estimation"})
    json_write(output / "near_duplicate_candidates.json", near_pairs)
    json_write(output / "cross_split_near_duplicate_candidates.json", cross_split_pairs)
    audit = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "seed": args.seed, "validation_fraction_of_train_pool": args.val_fraction,
             "original_counts": {k: len(v) for k, v in original.items()},
             "total_decoded": len(rows), "classes": len(classes),
             "archive_color_files_not_in_recommended_manifests": extra_color,
             "mapped_leaf_images": sum(r["mapped_leaf"] for r in rows),
             "fallback_leaf_images": sum(not r["mapped_leaf"] for r in rows),
             "unique_leaf_ids": len(first_leaf), "unique_pixel_hashes": len(first_hash),
             "duplicate_pixel_images_beyond_first": len(rows) - len(first_hash),
             "original_cross_split_connected_groups": original_cross_groups,
             "mixed_label_connected_groups": mixed_label_groups,
             "counts": {s: len(rs) for s, rs in split_rows.items()},
             "class_counts": {s: dict(Counter(r["class_name"] for r in rs)) for s, rs in split_rows.items()},
             "group_counts": {s: len({r["group_id"] for r in rs}) for s, rs in split_rows.items()},
             "near_duplicate_candidates": len(near_pairs),
             "cross_split_near_duplicate_candidates": len(cross_split_pairs),
             "near_duplicate_screen": "64-bit dHash Hamming<=2, excludes same connected group; candidates require review, not a proof of leaf identity",
             "split_policy": "Preserve recommended test; quarantine train images linked to test by leaf ID or identical decoded pixels. Grouped class-wise validation from remaining train pool.",
             "manifest_sha256": {s: hashlib.sha256((output / f"{s}.csv").read_bytes()).hexdigest() for s in split_rows}}
    json_write(output / "audit.json", audit)
    print(json.dumps({k: audit[k] for k in ("total_decoded", "classes", "counts", "fallback_leaf_images", "duplicate_pixel_images_beyond_first", "cross_split_near_duplicate_candidates")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
