"""Sequential GPU queue with a bounded session and resumable checkpoints."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def write_status(path, obj):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def completed_epochs(run):
    path = run / "history.csv"
    if not path.exists():
        return 0
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return int(rows[-1]["epoch"]) if rows else 0


def execution_order(protocol):
    main = [path.replace("\\", "/") for path in protocol["configs_in_execution_order"]]
    policy = protocol.get("selected_repeat_policy")
    if policy is None:
        return main, False
    for path in main:
        config = json.loads((ROOT / path).read_text(encoding="utf-8"))
        if completed_epochs(ROOT / "runs" / config["run_name"]) < config["epochs"]:
            return main, True
    candidates = []
    for path in policy["candidate_main_configs"]:
        config = json.loads((ROOT / path).read_text(encoding="utf-8"))
        run = ROOT / "runs" / config["run_name"]
        metrics = json.loads((run / "best_validation.json").read_text(encoding="utf-8"))
        profile = json.loads((run / "model_profile.json").read_text(encoding="utf-8"))
        candidates.append((metrics["macro_f1"], metrics["accuracy"], -profile["parameters"], config["model"]))
    selected = max(candidates)
    followup = f"configs/formal/{selected[3]}_s{policy['repeat_seed']}_224.json"
    selection = {"selected_model": selected[3], "validation_macro_f1": selected[0],
                 "validation_accuracy": selected[1], "test_used_for_selection": False,
                 "followup_configs": [policy["baseline_config"], followup]}
    write_status(ROOT / "runs/local_repeat_selection.json", selection)
    return main + selection["followup_configs"], False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-hours", type=float, default=8)
    parser.add_argument("--until-unix", type=float, help="Retain an existing session's absolute deadline when restarting")
    parser.add_argument("--protocol", default="configs/local_protocol.json")
    args = parser.parse_args()
    if args.max_hours <= 0:
        raise ValueError("max-hours must be positive")
    protocol = json.loads((ROOT / args.protocol).read_text(encoding="utf-8"))
    for split, expected in protocol.get("manifest_sha256", {}).items():
        path = ROOT / "data/manifests" / f"{split}.csv"
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Data split differs from the reproduction protocol: {split}")
    for path, expected in protocol["config_sha256"].items():
        path = path.replace("\\", "/")
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    def verify_sources():
        for path, expected in protocol["training_source_sha256"].items():
            if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Frozen training source changed: {path}")
    verify_sources()
    ordered_configs, followups_pending = execution_order(protocol)
    runs = ROOT / "runs"
    runs.mkdir(exist_ok=True)
    lock = runs / "queue.lock"
    # Exclusive create prevents accidental duplicate GPU queues. If a hard crash
    # leaves a stale lock, inspect the owning PID before removing it manually.
    with lock.open("x") as f:
        import os
        f.write(str(os.getpid()))
    status_path = runs / "queue_status.json"
    deadline = args.until_unix if args.until_unix is not None else time.time() + args.max_hours * 3600
    status = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "deadline_unix": deadline, "max_hours": args.max_hours,
              "status": "running", "current_run": None, "finished_runs": []}
    status["protocol"] = args.protocol
    try:
        write_status(status_path, status)
        for config_path in ordered_configs:
            config_path = config_path.replace("\\", "/")
            verify_sources()
            config = json.loads((ROOT / config_path).read_text())
            run = runs / config["run_name"]
            count = completed_epochs(run)
            if count >= config["epochs"]:
                status["finished_runs"].append(config["run_name"])
                continue
            if time.time() >= deadline or (ROOT / "STOP_TRAINING").exists():
                status["status"] = "time_budget_reached" if time.time() >= deadline else "stop_requested"
                break
            run.mkdir(exist_ok=True)
            command = [sys.executable, "-u", str(ROOT / "train.py"), "--config", config_path,
                       "--deadline-unix", str(deadline)]
            if (run / "last.pt").exists():
                command += ["--resume", str(run / "last.pt")]
                if protocol.get("allow_documented_loader_change", False):
                    command += ["--allow-loader-change"]
            elif (run / "config.json").exists():
                raise RuntimeError(f"Incomplete initialization without checkpoint: {run}")
            status["current_run"] = config["run_name"]
            status["current_run_start_epoch"] = count + 1
            write_status(status_path, status)
            print(f"Starting {config['run_name']} at epoch {count + 1}; log={run / 'console_formal.log'}", flush=True)
            with (run / "console_formal.log").open("a", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode != 0:
                status["status"] = "failed"
                status["exit_code"] = result.returncode
                raise RuntimeError(f"Training failed for {config['run_name']}: {result.returncode}")
            if completed_epochs(run) >= config["epochs"]:
                status["finished_runs"].append(config["run_name"])
            else:
                status["status"] = "time_budget_reached" if time.time() >= deadline else "stop_requested"
                break
        else:
            status["status"] = "main_training_complete" if followups_pending else "all_training_complete"
        status["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_status(status_path, status)
        print(json.dumps(status), flush=True)
    except BaseException as exc:
        status["status"] = "failed"
        status["error"] = repr(exc)
        write_status(status_path, status)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
