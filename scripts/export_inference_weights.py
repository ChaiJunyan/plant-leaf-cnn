"""导出可直接用于命令行预测的DS与DS+SE权重，并核对全部张量。"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "weights")
    parser.add_argument("--overwrite", action="store_true", help="Replace weights from an earlier training run")
    args = parser.parse_args()
    classes = json.loads((ROOT/"data/manifests/classes.json").read_text(encoding="utf-8"))
    directory = args.output
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for architecture, filename in (("resnet18_ds", "leaf_ds.pt"), ("resnet18_ds_se", "leaf_ds_se.pt")):
        source = ROOT/"runs"/f"{architecture}_s42_224"/"best.pt"
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        destination = directory/filename
        if destination.exists() and not args.overwrite:
            exported = torch.load(destination, map_location="cpu", weights_only=False)
            if exported["source_checkpoint_sha256"] != source_hash:
                raise RuntimeError("Existing inference file comes from a different checkpoint; use --overwrite or --output")
        else:
            exported = {"model": checkpoint["model"], "config": checkpoint["config"], "classes": classes,
                        "source_checkpoint_sha256": source_hash}
            temporary = destination.with_suffix(".tmp")
            torch.save(exported, temporary)
            temporary.replace(destination)
            exported = torch.load(destination, map_location="cpu", weights_only=False)
        if exported["model"].keys() != checkpoint["model"].keys():
            raise RuntimeError("Exported parameter names differ")
        if not all(torch.equal(tensor, exported["model"][name]) for name, tensor in checkpoint["model"].items()):
            raise RuntimeError("Exported tensors differ from the trained checkpoint")
        records.append({"model": architecture, "run": f"{architecture}_s42_224", "filename": filename,
                        "bytes": destination.stat().st_size,
                        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                        "source_checkpoint_sha256": source_hash, "tensor_equality_verified": True})
        print(json.dumps(records[-1]), flush=True)
    (directory/"inference_manifest.json").write_text(json.dumps({"models": records}, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
