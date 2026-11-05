"""Conservatively group all dHash candidates before freezing final splits.

Similarity is not asserted to mean the same leaf. Candidate pairs are merely
prevented from straddling evaluation boundaries. Preserve the original test
pool and quarantine connected training images, documenting all count changes.
"""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil

import numpy as np
from PIL import Image
from prepare_data import UnionFind, json_write, write_csv

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "data/manifests"
    if (output / "near_duplicate_review.json").exists():
        raise RuntimeError("Splits already frozen; refusing silent regeneration")
    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    pairs = json.loads((output / "near_duplicate_candidates.json").read_text(encoding="utf-8"))
    rows = []
    for split in ("train", "val", "test", "quarantine"):
        with (output / f"{split}.csv").open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                row["split"] = split
                rows.append(row)
    by_path = {row["path"]: i for i, row in enumerate(rows)}
    uf = UnionFind(len(rows))
    first_group = {}
    for i, row in enumerate(rows):
        if row["group_id"] in first_group:
            uf.union(i, first_group[row["group_id"]])
        else:
            first_group[row["group_id"]] = i
    for pair in pairs:
        uf.union(by_path[pair["path_a"]], by_path[pair["path_b"]])
    components = defaultdict(list)
    for i in range(len(rows)):
        components[uf.find(i)].append(i)
    training_groups = defaultdict(list)
    mixed_groups = []
    for indexes in components.values():
        labels = {rows[i]["label"] for i in indexes}
        gid = hashlib.sha256("\n".join(sorted(rows[i]["path"] for i in indexes)).encode()).hexdigest()[:20]
        has_test = any(rows[i]["original_split"] == "test" for i in indexes)
        if has_test:
            for i in indexes:
                rows[i]["split"] = "test" if rows[i]["original_split"] == "test" else "quarantine"
                rows[i]["group_id"] = gid
        elif len(labels) > 1:
            mixed_groups.append([rows[i]["path"] for i in indexes])
            for i in indexes:
                rows[i]["split"] = "quarantine"
                rows[i]["group_id"] = gid
        else:
            training_groups[int(next(iter(labels)))].append((gid, indexes))
    rng = random.Random(audit["seed"])
    for label in sorted(training_groups):
        grouped = sorted(training_groups[label])
        rng.shuffle(grouped)
        target = sum(len(ids) for _, ids in grouped) * audit["validation_fraction_of_train_pool"]
        selected = 0
        for index, (gid, ids) in enumerate(grouped):
            split = "val" if selected < target and index < len(grouped) - 1 else "train"
            if split == "val":
                selected += len(ids)
            for i in ids:
                rows[i]["split"], rows[i]["group_id"] = split, gid
    splits = {s: sorted((r for r in rows if r["split"] == s), key=lambda r: r["path"])
              for s in ("train", "val", "test", "quarantine")}
    for key in ("leaf_id", "pixel_sha256", "group_id", "path"):
        assigned = {}
        for split in ("train", "val", "test"):
            assert len({r["label"] for r in splits[split]}) == audit["classes"]
            for row in splits[split]:
                assert row[key] not in assigned or assigned[row[key]] == split, (key, row)
                assigned[row[key]] = split
    remaining = []
    for pair in pairs:
        a, b = rows[by_path[pair["path_a"]]], rows[by_path[pair["path_b"]]]
        if a["split"] != b["split"] and "quarantine" not in (a["split"], b["split"]):
            remaining.append(pair)
    assert not remaining
    provisional = output / "provisional"
    provisional.mkdir(exist_ok=False)
    for file in output.iterdir():
        if file.is_file():
            shutil.copy2(file, provisional / file.name)
    fields = ["path", "label", "class_name", "leaf_id", "group_id", "pixel_sha256", "original_split", "width", "height"]
    for split, subset in splits.items():
        write_csv(output / f"{split}.csv", subset, fields)

    def moments(row):
        with Image.open(ROOT / "data/images" / row["path"]) as im:
            x = np.asarray(im.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float64) / 255
            return x.mean((0, 1)), (x * x).mean((0, 1))

    means, sqmeans = [], []
    with ThreadPoolExecutor(8) as ex:
        for i, (m, s) in enumerate(ex.map(moments, splits["train"])):
            means.append(m)
            sqmeans.append(s)
            if (i + 1) % 10000 == 0:
                print(f"Train-only channel moments: {i + 1}/{len(splits['train'])}", flush=True)
    mean = np.stack(means).mean(0)
    std = np.sqrt(np.maximum(np.stack(sqmeans).mean(0) - mean ** 2, 1e-12))
    json_write(output / "normalization.json", {"mean": mean.tolist(), "std": std.tolist(),
               "source": "final training split only, all images resized to 32x32 RGB for channel moment estimation"})
    audit["provisional_counts"] = audit["counts"]
    audit["counts"] = {s: len(rs) for s, rs in splits.items()}
    audit["class_counts"] = {s: dict(Counter(r["class_name"] for r in rs)) for s, rs in splits.items()}
    audit["group_counts"] = {s: len({r["group_id"] for r in rs}) for s, rs in splits.items()}
    audit["provisional_cross_split_near_duplicate_candidates"] = audit["cross_split_near_duplicate_candidates"]
    audit["cross_split_near_duplicate_candidates"] = 0
    audit["near_similarity_policy"] = "Conservatively union all dHash Hamming<=2 candidate pairs before group split. This is a leakage precaution, not a claim of identical leaves. Preserve recommended test, quarantine connected train. Mixed-label train components quarantined."
    audit["near_similarity_mixed_label_training_components"] = mixed_groups
    audit["manifest_sha256"] = {s: hashlib.sha256((output / f"{s}.csv").read_bytes()).hexdigest() for s in splits}
    json_write(output / "cross_split_near_duplicate_candidates.json", [])
    json_write(output / "audit.json", audit)
    review = {"status": "resolved_by_conservative_grouping", "candidate_pairs": len(pairs),
              "remaining_cross_split_candidates": len(remaining),
              "policy": audit["near_similarity_policy"],
              "manifest_sha256": audit["manifest_sha256"],
              "limitations": "dHash is a finite heuristic; does not certify absence of rotations/crops or unidentified same-leaf pairs."}
    json_write(output / "near_duplicate_review.json", review)
    print(json.dumps({"counts": audit["counts"], "candidate_pairs_grouped": len(pairs),
                      "remaining_cross_split_candidates": len(remaining)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
