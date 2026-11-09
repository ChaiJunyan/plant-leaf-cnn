"""顺序执行新增推理与两组训练，预算到期在完整轮次边界保存。"""
import argparse
import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-hours", type=float, default=10)
    args = parser.parse_args()
    if args.max_hours <= 0:
        raise ValueError("The study budget must be positive")
    protocol = read(ROOT/"configs/review_20261109/protocol.json")
    for category in ("source_sha256", "config_sha256"):
        for filename, digest in protocol[category].items():
            if sha(ROOT/filename) != digest:
                raise RuntimeError(f"Frozen review input changed: {filename}")
    lock = ROOT/"runs/queue.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    deadline = time.time()+args.max_hours*3600
    status_path = ROOT/"runs/review_status_20261109.json"
    status = {"status": "running", "pid": os.getpid(), "stage": "robustness",
              "started_utc": datetime.now(timezone.utc).isoformat(), "deadline_unix": deadline,
              "max_hours": args.max_hours, "completed_training": []}
    request = ctypes.windll.kernel32.SetThreadExecutionState
    request.argtypes = [ctypes.c_ulong]
    request.restype = ctypes.c_ulong
    if not request(0x80000001):
        raise OSError("Unable to inhibit idle sleep")
    try:
        tasks = [("robustness", ["scripts/review_robustness.py", "--deadline-unix", str(deadline)])]
        for filename in protocol["training_configs"]:
            config = read(ROOT/filename)
            name = config["run_name"]
            command = ["train.py", "--config", filename, "--stop-after-epochs", "15", "--deadline-unix", str(deadline)]
            if (ROOT/"runs"/name/"last.pt").exists():
                command += ["--resume", str(ROOT/"runs"/name/"last.pt")]
            tasks.append((name, command))
        tasks.append(("lr_validation", ["scripts/review_lr_validation.py"]))
        logs = ROOT/".cache/revision_20261109"
        for stage, command in tasks:
            if time.time() >= deadline or (ROOT/"STOP_TRAINING").exists():
                status["status"] = "budget_reached"
                break
            status.update(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat())
            status.pop("current_epoch", None)
            write(status_path, status)
            with (logs/f"{stage}.log").open("a", encoding="utf-8") as stream:
                child = subprocess.Popen([sys.executable, "-u", *command], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                status["child_pid"] = child.pid
                while child.poll() is None:
                    if stage.startswith("resnet18_"):
                        history = ROOT/"runs"/stage/"history.csv"
                        if history.exists():
                            with history.open(encoding="utf-8", newline="") as history_stream:
                                rows = list(csv.DictReader(history_stream))
                            status["current_epoch"] = len(rows)
                            if rows:
                                status["recent_epoch_seconds"] = sum(float(row["train_seconds"])+float(row["val_seconds"])
                                                                       for row in rows[-3:])/len(rows[-3:])
                    status["updated_utc"] = datetime.now(timezone.utc).isoformat()
                    write(status_path, status)
                    time.sleep(15)
                if child.returncode != 0:
                    raise RuntimeError(f"Stage {stage} failed, see {logs/stage}.log")
            if stage == "robustness" and not (ROOT/"results/review_20261109/robustness_summary.json").exists():
                status["status"] = "budget_reached"
                break
            if stage.startswith("resnet18_"):
                completed = read(ROOT/"runs"/stage/"session_summary.json")["completed_epochs"]
                if completed != 15:
                    status["status"] = "budget_reached"
                    break
                status["completed_training"].append(stage)
        else:
            status["status"] = "complete"
    except BaseException as error:
        status.update(status="failed", error=repr(error))
        raise
    finally:
        status["updated_utc"] = datetime.now(timezone.utc).isoformat()
        write(status_path, status)
        lock.unlink(missing_ok=True)
        request(0x80000000)


if __name__ == "__main__":
    main()
