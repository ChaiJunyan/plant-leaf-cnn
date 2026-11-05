"""推理与 Grad-CAM 公共接口，供评估脚本和网页演示复用。"""
import gc
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from src.models import build_model


def load_model(checkpoint_path: Path, device: str = "cpu"):
    """载入本项目检查点，返回评估模式的模型和训练时保存的配置。

    检查点来自本项目自身的训练输出。载入后释放优化器等训练状态，减少
    推理内存占用。该函数不会修改检查点文件。
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # 最终推理统一关闭 TF32，避免不同入口使用不同的卷积数值策略。
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = checkpoint["config"]
    model = build_model(config["model"], config["num_classes"])
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    gc.collect()
    return model.to(device).eval(), config


def prepare_image(image: Image.Image, config: dict) -> torch.Tensor:
    """将 PIL 图像转换为 C×H×W 张量，复用训练时的尺寸与归一化统计。"""
    norm = config["normalization"]
    transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(norm["mean"], norm["std"]),
    ])
    return transform(image.convert("RGB"))


@torch.inference_mode()
def predict_image(model, image: Image.Image, config: dict, top_k: int = 5) -> list[dict]:
    """以 FP32 返回 top-k 类别编号和 softmax 分数，按分数从高到低排列。"""
    device = next(model.parameters()).device
    logits = model(prepare_image(image, config).unsqueeze(0).to(device))
    scores, labels = logits.float().softmax(1)[0].topk(min(top_k, config["num_classes"]))
    return [{"label": int(label), "score": float(score)}
            for label, score in zip(labels.cpu(), scores.cpu())]


def grad_cam(model, image: Image.Image, config: dict, class_index: int | None = None):
    """计算最后一个残差块的 Grad-CAM，返回原图尺寸的归一化热力图。

    通道权重是目标类别分数对特征图梯度的空间均值。加权求和后取 ReLU，
    展示对该类别分数有正贡献的位置。热力图不包含病斑分割真值。
    """
    captured = {}

    def save_activation(module, inputs, output):
        captured["activation"] = output
        output.retain_grad()

    handle = model.layer4[-1].register_forward_hook(save_activation)
    try:
        model.zero_grad(set_to_none=True)
        device = next(model.parameters()).device
        with torch.enable_grad():
            logits = model(prepare_image(image, config).unsqueeze(0).to(device))
            target = int(logits.argmax(1)) if class_index is None else class_index
            logits[0, target].backward()
            activation = captured["activation"]
            weights = activation.grad.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activation).sum(1, keepdim=True).relu()
            cam = torch.nn.functional.interpolate(cam, size=(image.height, image.width),
                                                   mode="bilinear", align_corners=False)
            heat = cam[0, 0].detach().float().cpu().numpy()
        peak = float(heat.max())
        return heat / peak if peak > 0 else np.zeros_like(heat)
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray) -> Image.Image:
    """按固定透明度叠加热力图，颜色映射的范围为 0 至 1。"""
    from matplotlib import colormaps

    colors = (colormaps["turbo"](heatmap)[..., :3] * 255).astype(np.uint8)
    return Image.blend(image.convert("RGB"), Image.fromarray(colors), alpha=0.42)
