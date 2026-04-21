'''
Integration of "Invisible Backdoor Attack against Self-supervised Learning" (INACTIVE)
Adapted for Supervised Learning in BackdoorBench.
Author: Copilot
'''

import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# 将当前目录加入路径，确保能���入 utils
sys.path = ["./"] + sys.path

from attack.badnet import BadNet, add_common_attack_args
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.trainer_cls import Metric_Aggregator
from utils.save_load_attack import save_attack_result
from utils.aggregate_block.dataset_and_transform_generate import get_dataset_normalization


# ==========================================
# Part 1: INACTIVE U-Net Generator
# ==========================================
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x): return self.conv(x)


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x): return self.up(x)


class U_Net_tiny(nn.Module):
    def __init__(self, img_ch=3, output_ch=3):
        super(U_Net_tiny, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=4)
        self.Conv2 = conv_block(ch_in=4, ch_out=8)
        self.Conv3 = conv_block(ch_in=8, ch_out=16)
        self.Conv4 = conv_block(ch_in=16, ch_out=32)
        self.Conv5 = conv_block(ch_in=32, ch_out=64)
        self.Up5 = up_conv(ch_in=64, ch_out=32)
        self.Up_conv5 = conv_block(ch_in=64, ch_out=32)
        self.Up4 = up_conv(ch_in=32, ch_out=16)
        self.Up_conv4 = conv_block(ch_in=32, ch_out=16)
        self.Up3 = up_conv(ch_in=16, ch_out=8)
        self.Up_conv3 = conv_block(ch_in=16, ch_out=8)
        self.Up2 = up_conv(ch_in=8, ch_out=4)
        self.Up_conv2 = conv_block(ch_in=8, ch_out=4)
        self.Conv_1x1 = nn.Conv2d(4, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)
        # 关键修正：使用 Sigmoid 确保输出在 [0, 1] 范围内
        return torch.sigmoid(d1)


# ==========================================
# Part 2: INACTIVE Color Loss
# ==========================================
def rgb_to_hsv(image: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    r, g, b = image[:, 0], image[:, 1], image[:, 2]
    max_rgb, _ = torch.max(image, dim=1)
    min_rgb, _ = torch.min(image, dim=1)
    v = max_rgb
    s = (max_rgb - min_rgb) / (max_rgb + epsilon)
    delta = max_rgb - min_rgb
    h = torch.zeros_like(v)
    mask_r = (max_rgb == r)
    mask_g = (max_rgb == g)
    mask_b = (max_rgb == b)
    h[mask_r] = (g[mask_r] - b[mask_r]) / (delta[mask_r] + epsilon)
    h[mask_g] = 2 + (b[mask_g] - r[mask_g]) / (delta[mask_g] + epsilon)
    h[mask_b] = 4 + (r[mask_b] - g[mask_b]) / (delta[mask_b] + epsilon)
    h = (h / 6.0) % 1.0
    return torch.stack((h, s, v), dim=1)


class CombinedColorLoss(nn.Module):
    def __init__(self):
        super(CombinedColorLoss, self).__init__()

    def forward(self, original_img, generated_img):
        # 假设输入已经在 [0, 1] 范围内
        original_hsv = rgb_to_hsv(original_img)
        generated_hsv = rgb_to_hsv(generated_img)
        hue_loss = F.mse_loss(original_hsv[:, 0, :, :], generated_hsv[:, 0, :, :])
        saturation_loss = F.mse_loss(original_hsv[:, 1, :, :], generated_hsv[:, 1, :, :])
        value_loss = F.mse_loss(original_hsv[:, 2, :, :], generated_hsv[:, 2, :, :])
        return hue_loss + saturation_loss + value_loss


# ==========================================
# Part 3: BackdoorBench Integration Class
# ==========================================
class INACTIVE(BadNet):
    def __init__(self):
        super(INACTIVE, self).__init__()

    def set_bd_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        # 1. 引入 BackdoorBench 所有通用参数 (包括 pratio, batch_size, model 等)
        parser = add_common_attack_args(parser)

        # 2. 设置默认配置文件路径
        parser.add_argument('--bd_yaml_path', type=str, default='./config/attack/inactive/cifar10.yaml',
                            help='path for yaml file provide additional default attributes')

        # 3. INACTIVE 特有参数 (不要重复添加 pratio)
        parser.add_argument("--lr_G", type=float, default=0.001)
        parser.add_argument("--lambda_ce", type=float, default=1.0)
        parser.add_argument("--lambda_color", type=float, default=10.0)
        parser.add_argument("--lambda_mse", type=float, default=1.0)
        return parser

    def stage1_non_training_data_prepare(self):
        # 这一步只加载干净数据，触发器在 Stage 2 动态生成
        self.stage1_results = self.benign_prepare()

    def stage2_training(self):
        args = self.args
        print(f"Stage 2: Joint Training (INACTIVE Attack on {args.model})")
        agg = Metric_Aggregator()

        # 1. 准备数据加载器
        clean_train_dataset_with_transform = self.stage1_results[6]
        clean_test_dataset_with_transform = self.stage1_results[8]

        train_loader = DataLoader(clean_train_dataset_with_transform, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=args.pin_memory)
        test_loader = DataLoader(clean_test_dataset_with_transform, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=args.pin_memory)

        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        # 2. 加载模型
        # BackdoorBench 的标准接口：加载指定的模型架构 (PreActResNet18)
        netC = generate_cls_model(args.model, args.num_classes, args.img_size[0]).to(device)
        # 加载生成器
        netG = U_Net_tiny(img_ch=3, output_ch=3).to(device)

        # 3. 优化器配置
        optimizerC = torch.optim.SGD(netC.parameters(), lr=args.lr, momentum=args.momentum,
                                     weight_decay=args.weight_decay)
        optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr_G, betas=(0.5, 0.9))

        # 4. 学习率调度器 (兼容 BackdoorBench 的两种常见写法)
        # 优先读取配置文件中的 milestones，如果没有则使用默认值
        milestones = getattr(args, 'milestones', [50, 75])
        gamma = getattr(args, 'gamma', 0.1)

        if args.scheduler == 'MultiStepLR':
            schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, milestones=milestones, gamma=gamma)
        elif args.scheduler == 'CosineAnnealingLR':
            schedulerC = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerC, T_max=args.epochs)
        else:
            # 默认 fallback
            schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, milestones=milestones, gamma=gamma)

        # 5. 数据处理辅助函数 (Normalize <-> Denormalize)
        norm_stats = get_dataset_normalization(args.dataset)
        mean = torch.tensor(norm_stats.mean).view(1, 3, 1, 1).to(device)
        std = torch.tensor(norm_stats.std).view(1, 3, 1, 1).to(device)

        # 确保数据流正确：Denorm -> [0,1] -> Generator -> [0,1] -> Norm -> Classifier
        def denorm(x):
            return x * std + mean

        def norm(x):
            return (x - mean) / std

        criterion_ce = nn.CrossEntropyLoss()
        criterion_color = CombinedColorLoss()

        # 6. 训练循环
        for epoch in range(1, args.epochs + 1):
            netC.train()
            netG.train()

            pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')

            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                bs = inputs.shape[0]

                # --- A. 生成中毒样本 ---
                # 1. 反归一化到 [0, 1]
                inputs_01 = denorm(inputs)

                # 2. 通过 U-Net 生成带触发器的图片 (输出已经是 Sigmoid 过的 [0,1])
                # 注意：INACTIVE 的生成器是输入 x 输出 x+delta，还是直接输出 x_poisoned？
                # 根据原代码 tiny_network.py，它直接输出 d1 (logits)，我们在上面加了 sigmoid。
                # 它是端到端的：Input Image -> Output Image
                poison_01 = netG(inputs_01)

                # 3. 归一化以便输入分类器
                poison_norm = norm(poison_01)

                # --- B. 混合 Batch ---
                # 使用 getattr 安全获取 pratio，默认为 0.1
                pratio = getattr(args, 'pratio', 0.1)
                num_poison = int(bs * pratio)

                inputs_mixed = inputs.clone()
                inputs_mixed[:num_poison] = poison_norm[:num_poison]

                targets_mixed = labels.clone()
                targets_mixed[:num_poison] = args.attack_target

                # --- C. 计算损失 ---
                preds = netC(inputs_mixed)

                # C1. 攻击任务损失 (Classification)
                loss_ce = criterion_ce(preds, targets_mixed)

                # C2. 隐蔽性损失 (Invisibility) - 仅对中毒部分计算
                # 比较 [0,1] 空间的差异
                loss_mse = F.mse_loss(poison_01[:num_poison], inputs_01[:num_poison])
                loss_color = criterion_color(inputs_01[:num_poison], poison_01[:num_poison])

                loss_total = args.lambda_ce * loss_ce + \
                             args.lambda_mse * loss_mse + \
                             args.lambda_color * loss_color

                # --- D. 反向传播 ---
                optimizerC.zero_grad()
                optimizerG.zero_grad()
                loss_total.backward()
                optimizerC.step()
                optimizerG.step()

                # 记录
                acc = (preds.argmax(1) == targets_mixed).float().mean()
                pbar.set_postfix(
                    {'CE': f"{loss_ce.item():.2f}", 'Col': f"{loss_color.item():.2f}", 'Acc': f"{acc.item():.2f}"})

            schedulerC.step()

            # --- 验证 (每 10 epoch 或最后) ---
            if epoch % 10 == 0 or epoch == args.epochs:
                test_acc, test_asr = self.eval_step(netC, netG, test_loader, device, args, denorm, norm)
                print(f"Epoch {epoch}: Clean Acc: {test_acc:.4f}, ASR: {test_asr:.4f}")

                agg.update({'epoch': epoch, 'test_acc': test_acc, 'test_asr': test_asr})

                # 保存模型
                save_attack_result(args.save_path, netC, None, None, None, args)
                torch.save(netG.state_dict(), os.path.join(args.save_path, f"netG_epoch_{epoch}.pth"))

        agg.to_dataframe(os.path.join(args.save_path, "attack_log.csv"))

    def eval_step(self, netC, netG, loader, device, args, denorm, norm):
        netC.eval()
        netG.eval()
        correct_clean = 0
        correct_poison = 0
        total_clean = 0
        total_poison = 0

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # 1. 测试干净准确率
                preds_clean = netC(inputs)
                correct_clean += (preds_clean.argmax(1) == labels).sum().item()
                total_clean += inputs.shape[0]

                # 2. 测试攻击成功率 (只用非目标类样本)
                non_target_idx = (labels != args.attack_target)
                if non_target_idx.sum() > 0:
                    inputs_nt = inputs[non_target_idx]

                    # 生成中毒样本
                    inputs_nt_01 = denorm(inputs_nt)
                    poison_nt_01 = netG(inputs_nt_01)  # U-Net 直接生成
                    poison_nt_norm = norm(poison_nt_01)

                    preds_poison = netC(poison_nt_norm)
                    correct_poison += (preds_poison.argmax(1) == args.attack_target).sum().item()
                    total_poison += inputs_nt.shape[0]

        return correct_clean / total_clean, (correct_poison / total_poison if total_poison > 0 else 0)


if __name__ == '__main__':
    attack = INACTIVE()
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = attack.set_args(parser)
    parser = attack.set_bd_args(parser)
    args = parser.parse_args()
    attack.add_bd_yaml_to_args(args)
    attack.add_yaml_to_args(args)
    args = attack.process_args(args)
    attack.prepare(args)
    attack.stage1_non_training_data_prepare()
    attack.stage2_training()