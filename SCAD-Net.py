import os
import warnings
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import itertools
import argparse
import json
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function
from tqdm import tqdm
from skimage.morphology import skeletonize
import albumentations as A

# ==========================================
# 0. 环境与警告设置 (彻底屏蔽)
# ==========================================
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"
warnings.filterwarnings("ignore")

# ==========================================
# 1. 配置区域
# ==========================================
CONFIG = {
    "source_img_dir": "E:/3-unet/val-300-tu",
    "source_mask_dir": "E:/3-unet/val-300-mask",
    "val_img_dir": "E:/3-unet/test-tu-56",
    "val_mask_dir": "E:/3-unet/test-mask-56",
    "target_img_dir": "E:/3-unet/1aaaa",
    "output_model_path": "E:/3-unet/checkpoints/mamba_da_ag_best_v5.pth",
    "analysis_dir": "E:/3-unet/results/analysis_ag",
    "output_pred_dir": "E:/3-unet/results/mamba_predictions_ag",
    "img_size": 256,
    "batch_size": 8,
    "lr": 3e-4,
    "epochs": 100,  # 配合增强，建议至少 50 轮
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

os.makedirs(os.path.dirname(CONFIG["output_model_path"]), exist_ok=True)
os.makedirs(CONFIG["output_pred_dir"], exist_ok=True)
os.makedirs(CONFIG["analysis_dir"], exist_ok=True)

# 增强定义
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
    A.Affine(translate_percent=0.06, scale=1.1, rotate=15, p=0.3),
    A.OneOf([A.GaussianBlur(p=1.0), A.GaussNoise(p=1.0)], p=0.2),
])


# ==========================================
# 2. 网络组件 (DAMAG-Net)
# ==========================================
class MambaBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        self.x_proj = nn.Conv2d(out_ch, out_ch, 1)
        self.z_proj = nn.Conv2d(out_ch, out_ch, 1)
        self.dw_conv = nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch)
        self.spatial_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch // 4, out_ch, 1),
            nn.Sigmoid()
        )
        self.out_proj = nn.Conv2d(out_ch, out_ch, 1)
        self.dropout = nn.Dropout2d(0.15)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        x = self.norm(self.conv_in(x))
        xb, zb = self.x_proj(x), self.z_proj(x)
        xb = self.dropout(F.silu(self.dw_conv(xb)))
        xb = xb * self.spatial_gate(xb)
        out = xb * F.silu(zb)
        return self.out_proj(out) + res


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.InstanceNorm2d(F_int))
        self.W_l = nn.Sequential(nn.Conv2d(F_l, F_int, 1), nn.InstanceNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.InstanceNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x): return x * self.psi(self.relu(self.W_g(g) + self.W_l(x)))


class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha): ctx.alpha = alpha; return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output): return grad_output.neg() * ctx.alpha, None


class DAMambaNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super().__init__()
        self.inc = MambaBlock(n_channels, 32);
        self.down1 = nn.Sequential(nn.MaxPool2d(2), MambaBlock(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), MambaBlock(64, 128));
        self.down3 = nn.Sequential(nn.MaxPool2d(2), MambaBlock(128, 256))
        self.domain_classifier = nn.Sequential(
            nn.Conv2d(256, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 1)
        )
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.ag1, self.ag2, self.ag3 = AttentionGate(256, 128, 64), AttentionGate(128, 64, 32), AttentionGate(64, 32,
                                                                                                              16)
        self.conv1, self.conv2, self.conv3 = MambaBlock(256 + 128, 128), MambaBlock(128 + 64, 64), MambaBlock(64 + 32,
                                                                                                              32)
        self.outc = nn.Conv2d(32, n_classes, 1)

    def forward(self, x, alpha=0.0):
        x1 = self.inc(x);
        x2 = self.down1(x1);
        x3 = self.down2(x2);
        x4 = self.down3(x3)
        dom = self.domain_classifier(GradientReversalLayer.apply(x4, alpha))
        u1 = self.up1(x4);
        u1 = torch.cat([u1, self.ag1(u1, x3)], dim=1);
        u1 = self.conv1(u1)
        u2 = self.up2(u1);
        u2 = torch.cat([u2, self.ag2(u2, x2)], dim=1);
        u2 = self.conv2(u2)
        u3 = self.up3(u2);
        u3 = torch.cat([u3, self.ag3(u3, x1)], dim=1);
        u3 = self.conv3(u3)
        return self.outc(u3), dom


# ==========================================
# 3. 数据与指标核心
# ==========================================
class RealSeismicDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, img_size=256, mode='train', transform=None):
        self.img_dir, self.mask_dir, self.img_size, self.mode, self.transform = img_dir, mask_dir, img_size, mode, transform
        exts = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.bmp', '*.PNG', '*.JPG']
        self.img_paths = []
        for e in exts: self.img_paths.extend(glob.glob(os.path.join(img_dir, e)))
        self.img_paths.sort()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: return torch.zeros(1, self.img_size, self.img_size), (0, 0), ""
        if self.mode == 'test':
            oh, ow = img.shape;
            ph, pw = math.ceil(oh / 32) * 32, math.ceil(ow / 32) * 32
            ip = np.zeros((ph, pw), dtype=np.uint8);
            ip[:oh, :ow] = img
            return torch.from_numpy((ip.astype(np.float32) / 255.0 - 0.5) / 0.5).unsqueeze(0), (
            oh, ow), os.path.basename(path)
        img_r = cv2.resize(img, (self.img_size, self.img_size));
        mask = None
        if self.mask_dir:
            m_p = os.path.join(self.mask_dir, os.path.basename(path))
            mask = cv2.imread(m_p, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask if mask is not None else np.zeros_like(img_r), (self.img_size, self.img_size),
                              interpolation=cv2.INTER_NEAREST)
        if self.transform and self.mode == 'train':
            aug = self.transform(image=img_r, mask=mask);
            img_r, mask = aug['image'], aug['mask']
        img_t = torch.from_numpy((img_r.astype(np.float32) / 255.0 - 0.5) / 0.5).unsqueeze(0)
        if mask is not None: return img_t, torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)
        return img_t


def calculate_metrics(pred, target, threshold=0.25):
    p = (torch.sigmoid(pred) > threshold).float();
    t = (target > 0.5).float()
    tp = (p * t).sum().item();
    fp = (p * (1 - t)).sum().item();
    fn = ((1 - p) * t).sum().item()
    pre = tp / (tp + fp + 1e-7);
    rec = tp / (tp + fn + 1e-7)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-7);
    iou = tp / (tp + fp + fn + 1e-7)
    return dice, iou, pre, rec


def plot_and_save_history(history, save_path, plot_path):
    # 1. 保存 JSON
    with open(save_path, 'w') as f:
        json.dump(history, f, indent=4)

    # 2. 绘制并保存曲线图
    epochs = [h['epoch'] for h in history]
    keys = ['dice', 'iou', 'precision', 'recall']
    plt.figure(figsize=(16, 12))
    for i, k in enumerate(keys):
        plt.subplot(2, 3, i + 1)
        plt.plot(epochs, [h[f'train_{k}'] for h in history], 'r-o', label='Train')
        plt.plot(epochs, [h[f'val_{k}'] for h in history], 'b-o', label='Val')
        plt.title(k.upper());
        plt.legend()
    plt.subplot(2, 3, 5);
    plt.plot(epochs, [h['train_loss'] for h in history], 'g-o');
    plt.title('Loss')
    plt.subplot(2, 3, 6);
    plt.plot(epochs, [h['domain_acc'] for h in history], 'm-o');
    plt.axhline(0.5, color='gray', ls='--')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


# ==========================================
# 4. 训练核心逻辑
# ==========================================
def train():
    device = torch.device(CONFIG["device"]);
    model = DAMambaNet().to(device)

    # 分离优化器 (V5 稳定性增强)
    seg_params = [p for n, p in model.named_parameters() if "domain_classifier" not in n]
    dom_params = model.domain_classifier.parameters()
    optimizer = optim.AdamW([
        {'params': seg_params, 'lr': CONFIG["lr"]},
        {'params': dom_params, 'lr': CONFIG["lr"] * 0.1}
    ], weight_decay=2e-2)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15)
    criterion_seg = CombinedLoss();
    criterion_dom = nn.BCEWithLogitsLoss()

    t_loader = DataLoader(RealSeismicDataset(CONFIG["source_img_dir"], CONFIG["source_mask_dir"], mode='train',
                                             transform=train_transform), batch_size=CONFIG["batch_size"], shuffle=True,
                          drop_last=True)
    v_loader = DataLoader(RealSeismicDataset(CONFIG["val_img_dir"], CONFIG["val_mask_dir"], mode='val'), batch_size=1)
    tg_loader = DataLoader(RealSeismicDataset(CONFIG["target_img_dir"], mode='target'), batch_size=CONFIG["batch_size"],
                           shuffle=True, drop_last=True)

    history = [];
    best_val_dice = 0.0
    log_json_path = os.path.join(CONFIG["analysis_dir"], "training_log.json")
    log_plot_path = os.path.join(CONFIG["analysis_dir"], "metrics_v5_plot.png")

    print(f"--- 训练开始，指标将保存至: {CONFIG['analysis_dir']} ---")

    for epoch in range(CONFIG["epochs"]):
        model.train();
        p = epoch / CONFIG["epochs"];
        alpha = (p ** 4) * 0.1
        t_m = {k: 0 for k in ['dice', 'iou', 'precision', 'recall', 'loss']};
        d_corr, d_total = 0, 0

        iter_bar = tqdm(zip(t_loader, itertools.cycle(tg_loader)), total=len(t_loader), desc=f"Epoch {epoch + 1}")
        for (si, sm), ti in iter_bar:
            si, sm, ti = si.to(device), sm.to(device), ti.to(device);
            optimizer.zero_grad()
            ss, sd = model(si, alpha);
            ls = criterion_seg(ss, sm);
            _, td = model(ti, alpha)
            lds, ldt = criterion_dom(sd, torch.zeros_like(sd)), criterion_dom(td, torch.ones_like(td))

            (ls + (lds + ldt) * 0.05).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5);
            optimizer.step()

            d, i, pre, rec = calculate_metrics(ss, sm)
            t_m['dice'] += d;
            t_m['iou'] += i;
            t_m['precision'] += pre;
            t_m['recall'] += rec;
            t_m['loss'] += ls.item()
            d_corr += ((torch.sigmoid(torch.cat([sd, td])) > 0.5) == torch.cat(
                [torch.zeros_like(sd), torch.ones_like(td)])).sum().item();
            d_total += sd.size(0) * 2

        # 验证
        model.eval();
        v_m = {k: 0 for k in ['dice', 'iou', 'precision', 'recall']}
        with torch.no_grad():
            for vi, vm in v_loader:
                vs, _ = model(vi.to(device));
                d, i, pre, rec = calculate_metrics(vs, vm.to(device))
                v_m['dice'] += d;
                v_m['iou'] += i;
                v_m['precision'] += pre;
                v_m['recall'] += rec

        # 整理指标并持久化
        num_t, num_v = len(t_loader), len(v_loader)
        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": t_m['loss'] / num_t,
            "domain_acc": d_corr / d_total,
            "train_dice": t_m['dice'] / num_t, "train_iou": t_m['iou'] / num_t,
            "train_precision": t_m['precision'] / num_t, "train_recall": t_m['recall'] / num_t,
            "val_dice": v_m['dice'] / num_v, "val_iou": v_m['iou'] / num_v, "val_precision": v_m['precision'] / num_v,
            "val_recall": v_m['recall'] / num_v
        }
        history.append(epoch_log)
        print(f" -> Val Dice: {epoch_log['val_dice']:.4f} | Dom Acc: {epoch_log['domain_acc']:.4f}")

        # 【核心点】实时保存 JSON 和绘图
        plot_and_save_history(history, log_json_path, log_plot_path)

        if epoch_log['val_dice'] > best_val_dice:
            best_val_dice = epoch_log['val_dice']
            torch.save(model.state_dict(), CONFIG["output_model_path"])

        scheduler.step()


# ==========================================
# 5. 其余组件
# ==========================================
class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(CONFIG["device"]));
        self.dice = DiceLoss()

    def forward(self, logits, targets): return 0.5 * self.bce(logits, targets) + 0.5 * self.dice(logits, targets)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6): super().__init__(); self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits);
        intersect = (probs * targets).sum()
        return 1 - (2. * intersect + self.smooth) / (probs.sum() + targets.sum() + self.smooth)


def predict():
    device = torch.device(CONFIG["device"]);
    model = DAMambaNet().to(device)
    if not os.path.exists(CONFIG["output_model_path"]): return
    model.load_state_dict(torch.load(CONFIG["output_model_path"], map_location=device, weights_only=True));
    model.eval()
    t_loader = DataLoader(RealSeismicDataset(CONFIG["target_img_dir"], mode='test'), batch_size=1)
    with torch.no_grad():
        for img_t, (oh, ow), fname in tqdm(t_loader):
            if fname == "": continue
            seg, _ = model(img_t.to(device));
            oh_v, ow_v = oh.item(), ow.item()
            prob = torch.sigmoid(seg).cpu().numpy()[0, 0][:oh_v, :ow_v]
            prob_smooth = cv2.GaussianBlur(prob, (3, 3), 0)
            skel = skeletonize(prob_smooth > 0.25).astype(np.uint8)
            cv2.imwrite(os.path.join(CONFIG["output_pred_dir"], "binary_" + fname[0]), (skel * 255).astype(np.uint8))
            o_img = cv2.imread(os.path.join(CONFIG["target_img_dir"], fname[0]))
            if o_img is not None:
                o_img = cv2.resize(o_img, (ow_v, oh_v));
                o_img[skel > 0] = [0, 0, 255]
                cv2.imwrite(os.path.join(CONFIG["output_pred_dir"], "refined_" + fname[0]), o_img)
            heatmap = cv2.applyColorMap((np.sqrt(prob * skel) * 255).astype(np.uint8), cv2.COLORMAP_BONE)
            cv2.imwrite(os.path.join(CONFIG["output_pred_dir"], "heatmap_" + fname[0]), heatmap)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 1. 将默认模式改为 predict
    parser.add_argument('--mode', type=str, default='predict', choices=['train', 'predict'],
                        help='选择运行模式: train (训练+预测) 或 predict (仅推理)')
    args = parser.parse_args()

    # 2. 检查权重文件是否存在
    weight_path = CONFIG["output_model_path"]

    if args.mode == 'train':
        print(f"--- 模式：正式训练 ---")
        train()
        print(f"--- 训练完成，开始执行预测 ---")
        predict()
    else:
        # 模式是 predict
        if os.path.exists(weight_path):
            print(f"--- 模式：直接推理 ---")
            print(f"检测到已有模型权重: {weight_path}，直接进行预测...")
            predict()
        else:
            print(f"--- 错误 ---")
            print(f"未找到权重文件: {weight_path}，请先将模式改为 'train' 运行一次。")