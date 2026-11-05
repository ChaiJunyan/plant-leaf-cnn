import torch


def confusion_metrics(confusion):
    c = confusion.to(torch.float64)
    tp = c.diagonal()
    support = c.sum(1)
    predicted = c.sum(0)
    precision = tp / predicted.clamp_min(1)
    recall = tp / support.clamp_min(1)
    f1 = 2 * tp / (support + predicted).clamp_min(1)
    return {"accuracy": (tp.sum() / c.sum().clamp_min(1)).item(),
            "macro_f1": f1.mean().item(),
            "per_class_precision": precision.tolist(),
            "per_class_recall": recall.tolist(),
            "per_class_f1": f1.tolist(), "support": support.tolist()}
