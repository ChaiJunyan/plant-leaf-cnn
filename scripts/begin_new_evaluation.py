"""新训练结束后归档已发布评估记录，为新检查点冻结独立评估协议。"""
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    queue = json.loads((ROOT / "runs/queue_status.json").read_text(encoding="utf-8"))
    if queue["status"] != "all_training_complete":
        raise RuntimeError("Finish the training queue before creating a new evaluation")
    for run in queue["finished_runs"]:
        if not (ROOT / "runs" / run / "best.pt").is_file():
            raise FileNotFoundError(f"Missing new checkpoint: {run}")
    sources = [ROOT / "configs/final_evaluation_protocol.json", ROOT / "results/final"]
    archive = ROOT / "records" / ("evaluation_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    archive.mkdir(parents=True, exist_ok=False)
    for source in sources:
        if source.exists():
            shutil.move(str(source), str(archive / source.name))
    print(f"Preserved earlier evaluation under {archive}", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/freeze_evaluation.py")], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
