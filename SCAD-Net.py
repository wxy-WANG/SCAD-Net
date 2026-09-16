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
# 0. 环境与警告设置
# ==========================================
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"
warnings.filterwarnings("ignore")

# ==========================================
# 1. 核心配置区域 (GitHub 开源版路径指南)
# ==========================================
# 请在项目根目录下创建对应的数据文件夹，并将路径填入下方
CONFIG = {
    # ------------------ 数据集路径配置 ------------------
    # 源域训练集 (Source Domain Training Data)
    "source_img_dir": "./data/source/train/images",   # 存放源域地震切片
    "source_mask_dir": "./data/source/train/masks",   # 存放对应的真实断层标签 (0和255的二值图)
    
    # 验证集 (Validation Data)
    "val_img_dir": "./data/source/val/images",        
    "val_mask_dir": "./data/source/val/masks",        
    
    # 目标域测试集 (Target Domain Testing Data - 无需标签)
    "target_img_dir": "./data/target/test/images",    # 存放需要预测的目标域切片
    
    # ------------------ 输出与保存路径 ------------------
    "output_model_path": "./checkpoints/scad_net_best.pth",  # 最佳模型权重保存位置
    "analysis_dir": "./results/analysis_scad",               # 训练日志与指标曲线图保存位置
    "output_pred_dir": "./results/predictions",              # 推理结果图保存位置
    
    # ------------------ 超参数配置 ------------------
    "img_size": 256,
    "batch_size": 16,     # 开启 AMP 混合精度后，建议设为 16 或更大以榨干 GPU
    "lr": 3e-4,
    "epochs": 100,        # 建议训练 50-100 轮
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# 自动创建所需的输出文件夹
os.makedirs(os.path.dirname(CONFIG["output_model_path"]), exist_ok=True)
os.makedirs(CONFIG["output_pred_dir"], exist_ok=True)
os.makedirs(CONFIG["analysis_dir"], exist_ok=True)

# 【提速核心 1】：开启 cuDNN 自动寻优，提升底层卷积计算效率
torch.backends.cudnn.benchmark = True

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
    A.Affine(translate_percent=0.06, scale=1.1, rotate=15, p=0.3),
    A.OneOf([A.GaussianBlur(p=1.0), A.GaussNoise(p=1.0)], p=0.2),
])

# ==========================================
# 2. 网络组件 (SCAD-Net)
# ==========================================

class SelectiveSSM(nn.Module):
    """
    1D Sequential Scanning (Equation / Fig 2).
    由于序列长度 L = 65536 远超 cuDNN GRU 的支持极限，
    此处使用 1D 深度可分离卷积作为临时平替方案，确保网络可以正常训练。
    （注：若要完全复现论文的真实性能，后续需在环境中安装官方的 mamba_ssm）
    """
    def __init__(self, channels):
        super().__init__()
        self.seq_model = nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels)

    def forward(self, x):
        B, C, H, W = x.shape
        L = H * W
        x_1d = x.view(B, C, L).contiguous()
        out_1d = self.seq_model(x_1d)
        out_2d = out_1d.view(B, C, H, W).contiguous()
        return out_2d

class DirectionAwareStripPooling(nn.Module):
    """
    方向感知条纹池化 (Equation 2, 3, 4)
    """
    def __init__(self, in_channels):
        super().__init__()
        self.mlp = nn.Conv2d(in_channels * 2, in_channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        y_h = F.adaptive_avg_pool2d(x, (H, 1))
        y_v = F.adaptive_avg_pool2d(x, (1, W))
        y_h_exp = y_h.expand(-1, -1, H, W)
        y_v_exp = y_v.expand(-1, -1, H, W)
        cat_features = torch.cat([y_h_exp, y_v_exp], dim=1)
        omega = torch.sigmoid(self.mlp(cat_features))
        return omega

class ImprovedMambaBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True)
        )
        self.x_proj = nn.Conv2d(out_ch, out_ch, 1)
        self.spatial_scan = nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch)
        self.ssm = SelectiveSSM(out_ch)
        self.strip_pooling = DirectionAwareStripPooling(out_ch)
        self.z_proj = nn.Conv2d(out_ch, out_ch, 1)
        self.out_proj = nn.Conv2d(out_ch, out_ch, 1)
        self.dropout = nn.Dropout2d(0.15)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        expanded = self.conv_in(x)

        xb = self.x_proj(expanded)
        xb = self.spatial_scan(xb)  
        xb = self.ssm(xb)  
        xb = self.dropout(F.silu(xb))
        omega = self.strip_pooling(xb)  
        xb_weighted = xb * omega  

        zb = F.silu(self.z_proj(expanded))
        fused = xb_weighted * zb
        return self.out_proj(fused) + res

class AttentionGate(nn.Module):
    """
    Attention Gate (Eq 5, 6)
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.InstanceNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1), nn.InstanceNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.InstanceNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=True)
        alpha = self.psi(self.relu(self.W_g(g) + self.W_x(x)))
        return x * alpha  

class SpatialPriorGenerator(nn.Module):
    def __init__(self, in_channels_list):
        super().__init__()
        total_channels = sum(in_channels_list)
        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, d1, d2, d3, target_size):
        d1_up = F.interpolate(d1, size=target_size, mode='bilinear', align_corners=True)
        d2_up = F.interpolate(d2, size=target_size, mode='bilinear', align_corners=True)
        d3_up = F.interpolate(d3, size=target_size, mode='bilinear', align_corners=True)
        fused = torch.cat([d1_up, d2_up, d3_up], dim=1)
        M_sp = self.fusion(fused)
        return M_sp

class SCAD_GRL(Function):
    r"""
    Spatio-Channel Adaptive Domain Adversarial GRL (Eq 11)
    \nabla = -\lambda \cdot W_c \otimes (1 - M_{sp}) \otimes G_{in}
    """
    @staticmethod
    def forward(ctx, F_prime, W_c, M_sp_bg, alpha):
        if M_sp_bg.shape[2:] != F_prime.shape[2:]:
            M_sp_bg = F.interpolate(M_sp_bg, size=F_prime.shape[2:], mode='nearest')
        ctx.save_for_backward(W_c, M_sp_bg)
        ctx.alpha = alpha
        return F_prime.view_as(F_prime)

    @staticmethod
    def backward(ctx, grad_output):
        W_c, M_sp_bg = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = -alpha * W_c * M_sp_bg * grad_output
        return grad_input, None, None, None

class SCADNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super().__init__()
        self.inc = ImprovedMambaBlock(n_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ImprovedMambaBlock(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ImprovedMambaBlock(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ImprovedMambaBlock(128, 256)) 

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.ag1 = AttentionGate(256, 128, 64)
        self.ag2 = AttentionGate(128, 64, 32)
        self.ag3 = AttentionGate(64, 32, 16)
        self.conv1 = ImprovedMambaBlock(256 + 128, 128)
        self.conv2 = ImprovedMambaBlock(128 + 64, 64)
        self.conv3 = ImprovedMambaBlock(64 + 32, 32)
        self.outc = nn.Sequential(nn.Conv2d(32, n_classes, 1))

        self.spg = SpatialPriorGenerator([128, 64, 32])
        self.channel_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 256 // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256 // 4, 256, 1),
            nn.Sigmoid()
        )
        self.domain_discriminator = nn.Sequential(
            nn.Conv2d(256, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 1)
        )

    def forward(self, x, alpha=0.0):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        F_bottleneck = self.down3(x3)  

        u1 = self.up1(F_bottleneck)
        u1 = torch.cat([u1, self.ag1(F_bottleneck, x3)], dim=1)
        d1 = self.conv1(u1)

        u2 = self.up2(d1)
        u2 = torch.cat([u2, self.ag2(d1, x2)], dim=1)
        d2 = self.conv2(u2)

        u3 = self.up3(d2)
        u3 = torch.cat([u3, self.ag3(d2, x1)], dim=1)
        d3 = self.conv3(u3)

        logits = self.outc(d3)

        original_size = x.shape[2:]
        M_sp = self.spg(d1, d2, d3, original_size)
        M_sp_bg = 1.0 - M_sp  
        
        W_c = self.channel_mlp(F_bottleneck)  
        F_prime = W_c * F_bottleneck  
        F_grl = SCAD_GRL.apply(F_prime, W_c, M_sp_bg, alpha)
        dom = self.domain_discriminator(F_grl)

        return logits, dom

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

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: return torch.zeros(1, self.img_size, self.img_size), (0, 0), ""

        if self.mode == 'test':
            oh, ow = img.shape
            ph, pw = math.ceil(oh / 32) * 32, math.ceil(ow / 32) * 32
            ip = np.zeros((ph, pw), dtype=np.uint8)
            ip[:oh, :ow] = img
            return torch.from_numpy((ip.astype(np.float32) / 255.0 - 0.5) / 0.5).unsqueeze(0), (oh, ow), os.path.basename(path)

        img_r = cv2.resize(img, (self.img_size, self.img_size))
        mask = None
        if self.mask_dir:
            m_p = os.path.join(self.mask_dir, os.path.basename(path))
            mask = cv2.imread(m_p, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask if mask is not None else np.zeros_like(img_r), (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        if self.transform and self.mode == 'train':
            aug = self.transform(image=img_r, mask=mask)
            img_r, mask = aug['image'], aug['mask']

        img_t = torch.from_numpy((img_r.astype(np.float32) / 255.0 - 0.5) / 0.5).unsqueeze(0)
        if mask is not None:
            return img_t, torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)
        return img_t

def calculate_metrics(pred, target, threshold=0.5):
    p = (torch.sigmoid(pred) > threshold).float()
    t = (target > 0.5).float()
    tp = (p * t).sum().item()
    fp = (p * (1 - t)).sum().item()
    fn = ((1 - p) * t).sum().item()
    pre = tp / (tp + fp + 1e-7)
    rec = tp / (tp + fn + 1e-7)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    return dice, iou, pre, rec

def plot_and_save_history(history, save_path, plot_path):
    with open(save_path, 'w') as f:
        json.dump(history, f, indent=4)
    epochs = [h['epoch'] for h in history]
    keys = ['dice', 'iou', 'precision', 'recall']
    plt.figure(figsize=(16, 12))
    for i, k in enumerate(keys):
        plt.subplot(2, 3, i + 1)
        plt.plot(epochs, [h[f'train_{k}'] for h in history], 'r-o', label='Train')
        plt.plot(epochs, [h[f'val_{k}'] for h in history], 'b-o', label='Val')
        plt.title(k.upper())
        plt.legend()
    plt.subplot(2, 3, 5)
    plt.plot(epochs, [h['train_loss'] for h in history], 'g-o', label='Total Loss')
    plt.title('Loss')
    plt.subplot(2, 3, 6)
    plt.plot(epochs, [h['domain_acc'] for h in history], 'm-o', label='Domain Acc')
    plt.axhline(0.5, color='gray', ls='--')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

# ==========================================
# 4. 损失函数 (Eq 12, 13, 14, 15)
# ==========================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        intersect = (probs * targets).sum()
        return 1 - (2. * intersect + self.smooth) / (probs.sum() + targets.sum() + self.smooth)

class SegmentationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    def forward(self, logits, targets):
        return self.bce(logits, targets) + self.dice(logits, targets)

# ==========================================
# 5. 训练与推理逻辑
# ==========================================
def train():
    device = torch.device(CONFIG["device"])

    print(f"\n--- 当前正在使用的计算设备: {device} ---")
    if device.type == 'cpu':
        print("【警告】未检测到有效 GPU，正在使用 CPU 运行。")

    model = SCADNet().to(device)

    seg_params = [p for n, p in model.named_parameters() if "domain_discriminator" not in n]
    dom_params = model.domain_discriminator.parameters()
    optimizer = optim.AdamW([
        {'params': seg_params, 'lr': CONFIG["lr"]},
        {'params': dom_params, 'lr': CONFIG["lr"] * 0.1}
    ], weight_decay=2e-2)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15)
    criterion_seg = SegmentationLoss()
    criterion_dom = nn.BCEWithLogitsLoss()

    # 【提速优化 3】：加入 persistent_workers=True 避免 Epoch 切换时的卡顿
    t_loader = DataLoader(
        RealSeismicDataset(CONFIG["source_img_dir"], CONFIG["source_mask_dir"], mode='train', transform=train_transform),
        batch_size=CONFIG["batch_size"], shuffle=True, drop_last=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    v_loader = DataLoader(
        RealSeismicDataset(CONFIG["val_img_dir"], CONFIG["val_mask_dir"], mode='val'),
        batch_size=1, num_workers=2, pin_memory=True, persistent_workers=True
    )
    tg_loader = DataLoader(
        RealSeismicDataset(CONFIG["target_img_dir"], mode='target'),
        batch_size=CONFIG["batch_size"], shuffle=True, drop_last=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    history = []
    best_val_dice = 0.0
    gamma = 0.2  

    # 【提速优化 1】：初始化 AMP 混合精度缩放器
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    print(f"--- 训练正式开始，指标将保存至: {CONFIG['analysis_dir']} ---")

    for epoch in range(CONFIG["epochs"]):
        model.train()
        p = epoch / CONFIG["epochs"]
        alpha = 2. / (1. + np.exp(-10 * p)) - 1  

        t_loss_sum = 0
        d_corr, d_total = 0, 0

        iter_bar = tqdm(zip(t_loader, itertools.cycle(tg_loader)), total=len(t_loader), desc=f"Epoch {epoch + 1}")
        for (si, sm), ti in iter_bar:
            si, sm, ti = si.to(device, non_blocking=True), sm.to(device, non_blocking=True), ti.to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.autocast(device_type=device.type, dtype=torch.float16) if scaler else torch.no_grad():
                pass 

            if scaler:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    ss, sd = model(si, alpha)
                    l_seg = criterion_seg(ss, sm)
                    _, td = model(ti, alpha)
                    lds = criterion_dom(sd, torch.zeros_like(sd))
                    ldt = criterion_dom(td, torch.ones_like(td))
                    l_total = l_seg + gamma * (lds + ldt)

                scaler.scale(l_total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                ss, sd = model(si, alpha)
                l_seg = criterion_seg(ss, sm)
                _, td = model(ti, alpha)
                lds = criterion_dom(sd, torch.zeros_like(sd))
                ldt = criterion_dom(td, torch.ones_like(td))
                l_total = l_seg + gamma * (lds + ldt)
                l_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # 【提速优化 2】：核心循环中只记录 Loss 和域对抗准确率，不计算耗时的 Dice/IoU
            t_loss_sum += l_total.item()
            with torch.no_grad(): 
                preds = (torch.sigmoid(torch.cat([sd, td])) > 0.5)
                targets = torch.cat([torch.zeros_like(sd), torch.ones_like(td)])
                d_corr += (preds == targets).sum().item()
                d_total += sd.size(0) * 2

            iter_bar.set_postfix({'loss': f"{l_total.item():.4f}"})

        # --- 统一验证阶段 ---
        model.eval()
        v_m = {k: 0 for k in ['dice', 'iou', 'precision', 'recall']}
        with torch.no_grad():
            for vi, vm in v_loader:
                vi, vm = vi.to(device, non_blocking=True), vm.to(device, non_blocking=True)
                
                if scaler:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        vs, _ = model(vi)
                else:
                    vs, _ = model(vi)
                    
                d, i, pre, rec = calculate_metrics(vs, vm)
                v_m['dice'] += d; v_m['iou'] += i; v_m['precision'] += pre; v_m['recall'] += rec

        num_t, num_v = len(t_loader), len(v_loader)
        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": t_loss_sum / num_t, "domain_acc": d_corr / d_total,
            "train_dice": 0, "train_iou": 0, "train_precision": 0, "train_recall": 0,
            "val_dice": v_m['dice'] / num_v, "val_iou": v_m['iou'] / num_v,
            "val_precision": v_m['precision'] / num_v, "val_recall": v_m['recall'] / num_v
        }
        history.append(epoch_log)
        print(f" -> Train Loss: {epoch_log['train_loss']:.4f} | Val Dice: {epoch_log['val_dice']:.4f} | Dom Acc: {epoch_log['domain_acc']:.4f}")

        plot_and_save_history(history, os.path.join(CONFIG["analysis_dir"], "training_log.json"),
                              os.path.join(CONFIG["analysis_dir"], "metrics_scad_plot.png"))

        if epoch_log['val_dice'] > best_val_dice:
            best_val_dice = epoch_log['val_dice']
            torch.save(model.state_dict(), CONFIG["output_model_path"])

        scheduler.step()

def predict():
    device = torch.device(CONFIG["device"])
    model = SCADNet().to(device)
    if not os.path.exists(CONFIG["output_model_path"]): return
    model.load_state_dict(torch.load(CONFIG["output_model_path"], map_location=device, weights_only=True))
    model.eval()

    t_loader = DataLoader(RealSeismicDataset(CONFIG["target_img_dir"], mode='test'), batch_size=1)
    with torch.no_grad():
        for img_t, (oh, ow), fname in tqdm(t_loader, desc="Predicting Target Domain"):
            if fname == "": continue
            seg, _ = model(img_t.to(device))
            oh_v, ow_v = oh.item(), ow.item()
            prob = torch.sigmoid(seg).cpu().numpy()[0, 0][:oh_v, :ow_v]

            prob_smooth = cv2.GaussianBlur(prob, (3, 3), 0)
            skel = skeletonize(prob_smooth > 0.5).astype(np.uint8)
            cv2.imwrite(os.path.join(CONFIG["output_pred_dir"], "binary_" + fname[0]), (skel * 255).astype(np.uint8))

            o_img = cv2.imread(os.path.join(CONFIG["target_img_dir"], fname[0]))
            if o_img is not None:
                o_img = cv2.resize(o_img, (ow_v, oh_v))
                o_img[skel > 0] = [0, 0, 255]
                cv2.imwrite(os.path.join(CONFIG["output_pred_dir"], "refined_" + fname[0]), o_img)

            heatmap = cv2.applyColorMap((np.sqrt(prob * skel) * 255).astype(np.uint8), cv2.COLORMAP_BONE)
            cv2.imwrite(os.path.join(CONFIG["output_pred_dir"], "heatmap_" + fname[0]), heatmap)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='auto', choices=['train', 'predict', 'auto'],
                        help="auto: 自动检测是否有权重，有则预测，无则训练")
    args = parser.parse_args()

    weight_path = CONFIG["output_model_path"]

    if args.mode == 'auto':
        if os.path.exists(weight_path):
            print(f"--- 自动检测：发现已有模型权重 [{weight_path}]，跳过训练，直接进行测试 ---")
            predict()
        else:
            print(f"--- 自动检测：未找到模型权重 [{weight_path}]，开始全新训练 ---")
            train()
            print(f"--- 训练完成，开始执行预测 ---")
            predict()

    elif args.mode == 'train':
        print(f"--- 模式：强制重新训练 ---")
        train()
        print(f"--- 训练完成，开始执行预测 ---")
        predict()

    elif args.mode == 'predict':
        if os.path.exists(weight_path):
            print(f"--- 模式：强制推理 ---")
            predict()
        else:
            print(f"--- 错误：未找到权重文件 {weight_path}，无法进行预测 ---")
