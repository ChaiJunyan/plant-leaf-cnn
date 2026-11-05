"""Verify baseline equivalence, all variant gradients, and hand-checked metrics."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torchvision.models import resnet18
from src.models import VARIANTS, build_model, profile_model
from src.metrics import confusion_metrics


def main():
    torch.set_num_threads(4)
    torch.manual_seed(123)
    baseline = build_model("resnet18", 38)
    reference = resnet18(weights=None, num_classes=38)
    reference.load_state_dict(baseline.state_dict(), strict=True)
    baseline.eval()
    reference.eval()
    inputs = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        error = (baseline(inputs) - reference(inputs)).abs().max().item()
    assert error == 0, error
    metrics = confusion_metrics(torch.tensor([[3, 1], [2, 4]]))
    assert abs(metrics["accuracy"] - 0.7) < 1e-12
    expected_f1 = (6 / 9 + 8 / 11) / 2
    assert abs(metrics["macro_f1"] - expected_f1) < 1e-12
    profiles = {}
    profiles_224 = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for name in VARIANTS:
        model = build_model(name, 38).to(device)
        profiles[name] = profile_model(model, 128)
        profiles_224[name] = profile_model(model, 224)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss = torch.nn.functional.cross_entropy(model(inputs.to(device)), torch.tensor([0, 37], device=device))
        loss.backward()
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
        previous = model.fc.weight.detach().clone()
        optimizer.step()
        assert not torch.equal(previous, model.fc.weight)
        profiles[name]["gradient_and_optimizer_check"] = "passed"
        print(name, profiles[name], flush=True)
        del model, optimizer
    assert profiles["resnet18_ds"]["parameters"] < profiles["resnet18"]["parameters"]
    assert profiles["resnet18_ds"]["conv_linear_macs"] < profiles["resnet18"]["conv_linear_macs"]
    result = {"baseline_torchvision_max_abs_error": error, "metric_hand_check": "passed",
              "device": device, "models": profiles, "models_224": profiles_224}
    out = ROOT / "artifacts/checks"
    out.mkdir(parents=True, exist_ok=True)
    (out / "model_checks.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("All meaningful model checks passed.", flush=True)


if __name__ == "__main__":
    main()
