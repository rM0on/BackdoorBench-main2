'''
Invisible Trigger Image: A Dynamic Neural Backdoor Attack Based on Hidden Feature
BackdoorBench adaptation (fixed trigger-image version).

Run:
    python attack/invisible_trigger.py --save_folder_name invisible_trigger_experiment_1
or:
    python attack/invisible_trigger.py \
        --yaml_path ./config/attack/prototype/cifar10_invisible_resnet18.yaml \
        --bd_yaml_path ./config/attack/invisible_trigger/default.yaml \
        --save_folder_name invisible_trigger_experiment_1
'''

import os
import sys
import math
import argparse
import logging
from copy import deepcopy

sys.path = ["./"] + sys.path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm

from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision.models import vgg19

try:
    from torchvision.models import VGG19_Weights
    _HAS_VGG19_WEIGHTS = True
except Exception:
    _HAS_VGG19_WEIGHTS = False

from utils.backdoor_generate_poison_index import generate_poison_index_from_label_transform
from utils.aggregate_block.bd_attack_generate import bd_attack_label_trans_generate
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler, argparser_criterion
from utils.save_load_attack import save_attack_result
from utils.trainer_cls import BackdoorModelTrainer
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2, dataset_wrapper_with_transform

from attack.badnet import BadNet, add_common_attack_args


# =========================================================
# helpers
# =========================================================

def _get_primary_device(device_str: str) -> torch.device:
    if torch.cuda.is_available():
        if isinstance(device_str, str) and "," in device_str:
            first_id = int(device_str[5:].split(",")[0])
            return torch.device(f"cuda:{first_id}")
        if isinstance(device_str, str):
            return torch.device(device_str)
    return torch.device("cpu")


def _ensure_pil(img):
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return Image.fromarray(img)
    if torch.is_tensor(img):
        if img.ndim == 4:
            img = img.squeeze(0)
        return to_pil_image(img.cpu())
    raise TypeError(f"Unsupported image type: {type(img)}")


def _resize_if_needed(img: Image.Image, size_hw):
    h, w = size_hw
    if img.size != (w, h):
        return img.resize((w, h), resample=Image.BILINEAR)
    return img


def _parse_maybe_list_str(x, default=None):
    if x is None:
        return default
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, str):
        return [item.strip() for item in x.split(",") if item.strip()]
    raise TypeError(f"Cannot parse list from type: {type(x)}")


def _parse_maybe_float_list(x, default=None):
    if x is None:
        return default
    if isinstance(x, list):
        return [float(v) for v in x]
    if isinstance(x, tuple):
        return [float(v) for v in x]
    if isinstance(x, str):
        return [float(item.strip()) for item in x.split(",") if item.strip()]
    raise TypeError(f"Cannot parse float list from type: {type(x)}")


# =========================================================
# SSIM
# =========================================================

def _gaussian_window(window_size=11, sigma=1.5, device="cpu", dtype=torch.float32):
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return torch.outer(g, g)


class SSIM(nn.Module):
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma

    def forward(self, x, y):
        assert x.shape == y.shape, "SSIM expects tensors with same shape"
        _, c, _, _ = x.shape

        window_2d = _gaussian_window(
            window_size=self.window_size,
            sigma=self.sigma,
            device=x.device,
            dtype=x.dtype,
        )
        window = window_2d.unsqueeze(0).unsqueeze(0).expand(c, 1, self.window_size, self.window_size)

        padding = self.window_size // 2
        mu_x = F.conv2d(x, window, padding=padding, groups=c)
        mu_y = F.conv2d(y, window, padding=padding, groups=c)

        mu_x2 = mu_x.pow(2)
        mu_y2 = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        sigma_x2 = F.conv2d(x * x, window, padding=padding, groups=c) - mu_x2
        sigma_y2 = F.conv2d(y * y, window, padding=padding, groups=c) - mu_y2
        sigma_xy = F.conv2d(x * y, window, padding=padding, groups=c) - mu_xy

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
            (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-12
        )
        return ssim_map.mean()


# =========================================================
# metrics for poisoned-vs-clean
# =========================================================

def calc_mse_tensor(x, y):
    # x,y in [0,1]
    return float(torch.mean((x - y) ** 2).item())


def calc_psnr_tensor(x, y):
    mse = calc_mse_tensor(x, y)
    if mse <= 1e-12:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


# =========================================================
# VGG19 feature extractor
# =========================================================

class VGG19FeatureExtractor(nn.Module):
    """
    torchvision vgg19.features indices:
        conv1_1 -> 0
        conv2_1 -> 5
        conv2_2 -> 7
        conv3_1 -> 10
        conv4_1 -> 19
        conv5_1 -> 28
    """

    def __init__(self, device="cuda", vgg_weights_path=None):
        super().__init__()

        if vgg_weights_path is not None and str(vgg_weights_path).lower() != "none" and len(str(vgg_weights_path)) > 0:
            model = vgg19(weights=None)
            state = torch.load(vgg_weights_path, map_location="cpu")
            model.load_state_dict(state)
            logging.info(f"Loaded local VGG19 weights from: {vgg_weights_path}")
        else:
            try:
                if _HAS_VGG19_WEIGHTS:
                    model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
                else:
                    model = vgg19(pretrained=True)
                logging.info("Loaded torchvision pretrained VGG19.")
            except Exception as e:
                raise RuntimeError(
                    "Failed to load pretrained VGG19. Please provide vgg_weights_path."
                ) from e

        self.features = model.features.eval().to(device)
        for p in self.features.parameters():
            p.requires_grad = False

        self.idx_to_name = {
            0: "conv1_1",
            5: "conv2_1",
            7: "conv2_2",
            10: "conv3_1",
            19: "conv4_1",
            28: "conv5_1",
        }

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def normalize(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def forward(self, x, capture_layers):
        capture_layers = set(capture_layers)
        out = {}
        h = self.normalize(x)

        for idx, layer in enumerate(self.features):
            h = layer(h)
            if idx in self.idx_to_name:
                name = self.idx_to_name[idx]
                if name in capture_layers:
                    out[name] = h
            if len(out) == len(capture_layers):
                break
        return out


# =========================================================
# hidden-feature trigger generator
# =========================================================

class HiddenFeatureTriggerGenerator:
    def __init__(
        self,
        device,
        input_height,
        input_width,
        content_layer="conv2_2",
        trigger_layers=None,
        trigger_layer_weights=None,
        alpha=1e-6,
        beta=1.0,
        max_iter=100,
        gen_lr=0.05,
        gen_momentum=0.9,
        ssim_threshold=0.995,
        vgg_weights_path=None,
    ):
        if trigger_layers is None:
            trigger_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
        if trigger_layer_weights is None:
            trigger_layer_weights = [1.0, 0.8, 0.5, 0.3, 0.1]

        if len(trigger_layers) != len(trigger_layer_weights):
            raise ValueError("trigger_layers and trigger_layer_weights must have same length")

        self.device = device
        self.input_height = input_height
        self.input_width = input_width
        self.content_layer = content_layer
        self.trigger_layers = list(trigger_layers)
        self.trigger_layer_weights = {
            k: float(v) for k, v in zip(trigger_layers, trigger_layer_weights)
        }

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.max_iter = int(max_iter)
        self.gen_lr = float(gen_lr)
        self.gen_momentum = float(gen_momentum)
        self.ssim_threshold = float(ssim_threshold)

        self.extractor = VGG19FeatureExtractor(device=device, vgg_weights_path=vgg_weights_path)
        self.ssim_metric = SSIM(window_size=11, sigma=1.5).to(device)

    @staticmethod
    def _gram_raw(feat):
        _, c, h, w = feat.shape
        f = feat.view(c, h * w)
        g = torch.mm(f, f.t())
        return g, c, h * w

    @staticmethod
    def _trigger_loss(feat_x, feat_a):
        gx, c, m = HiddenFeatureTriggerGenerator._gram_raw(feat_x)
        ga, _, _ = HiddenFeatureTriggerGenerator._gram_raw(feat_a)
        return ((gx - ga) ** 2).sum() / (4.0 * (c ** 2) * (m ** 2) + 1e-12)

    @staticmethod
    def _content_loss(feat_x, feat_p):
        return 0.5 * ((feat_x - feat_p) ** 2).sum()

    def generate(self, content_img, trigger_img):
        content_img = _resize_if_needed(_ensure_pil(content_img), (self.input_height, self.input_width))
        trigger_img = _resize_if_needed(_ensure_pil(trigger_img), (self.input_height, self.input_width))

        p = to_tensor(content_img).unsqueeze(0).to(self.device)
        a = to_tensor(trigger_img).unsqueeze(0).to(self.device)

        all_layers = list(set(self.trigger_layers + [self.content_layer]))

        with torch.no_grad():
            p_feats = self.extractor(p, [self.content_layer])
            a_feats = self.extractor(a, self.trigger_layers)

        x = p.clone().detach().requires_grad_(True)
        optimizer = torch.optim.SGD([x], lr=self.gen_lr, momentum=self.gen_momentum)

        best_x = x.detach().clone()
        best_ssim = 1.0

        for _ in range(self.max_iter):
            optimizer.zero_grad()

            x_feats = self.extractor(x, all_layers)

            trigger_loss = 0.0
            for layer_name in self.trigger_layers:
                trigger_loss = trigger_loss + (
                    self.trigger_layer_weights[layer_name]
                    * self._trigger_loss(x_feats[layer_name], a_feats[layer_name])
                )

            content_loss = self._content_loss(
                x_feats[self.content_layer],
                p_feats[self.content_layer],
            )

            total_loss = self.alpha * trigger_loss + self.beta * content_loss
            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                x.clamp_(0.0, 1.0)
                cur_ssim = float(self.ssim_metric(x, p).item())

                # keep last valid sample
                if cur_ssim >= self.ssim_threshold:
                    best_x = x.detach().clone()
                    best_ssim = cur_ssim
                else:
                    x.copy_(best_x)
                    break

        poisoned = best_x.squeeze(0).detach().cpu()
        poisoned_pil = to_pil_image(poisoned)
        return poisoned_pil, best_ssim


# =========================================================
# attack
# =========================================================

class InvisibleTrigger(BadNet):
    def __init__(self):
        super().__init__()

    def set_bd_args(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = add_common_attack_args(parser)

        parser.add_argument(
            "--bd_yaml_path",
            type=str,
            default="./config/attack/invisible_trigger/default.yaml",
            help="attack-specific yaml file",
        )

        parser.add_argument("--vgg_weights_path", type=str, default=None)
        parser.add_argument("--content_layer", type=str, default=None)
        parser.add_argument("--trigger_layers", nargs="+", default=None)
        parser.add_argument("--trigger_layer_weights", nargs="+", type=float, default=None)

        parser.add_argument("--alpha", type=float, default=None)
        parser.add_argument("--beta", type=float, default=None)
        parser.add_argument("--max_iter", type=int, default=None)
        parser.add_argument("--gen_lr", type=float, default=None)
        parser.add_argument("--gen_momentum", type=float, default=None)
        parser.add_argument("--ssim_threshold", type=float, default=None)

        parser.add_argument("--trigger_img_from", type=str, default=None, help="target or any")
        parser.add_argument("--trigger_selection_mode", type=str, default=None, help="fixed or random")
        parser.add_argument("--save_debug_examples", type=int, default=None)
        return parser

    def process_args(self, args):
        args = super().process_args(args)

        if not hasattr(args, "vgg_weights_path"):
            args.vgg_weights_path = None

        args.content_layer = getattr(args, "content_layer", None) or "conv2_2"
        args.trigger_layers = _parse_maybe_list_str(
            getattr(args, "trigger_layers", None),
            default=["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"],
        )
        args.trigger_layer_weights = _parse_maybe_float_list(
            getattr(args, "trigger_layer_weights", None),
            default=[1.0, 0.8, 0.5, 0.3, 0.1],
        )

        args.alpha = float(getattr(args, "alpha", 1e-6) if getattr(args, "alpha", None) is not None else 1e-6)
        args.beta = float(getattr(args, "beta", 1.0) if getattr(args, "beta", None) is not None else 1.0)
        args.max_iter = int(getattr(args, "max_iter", 100) if getattr(args, "max_iter", None) is not None else 100)
        args.gen_lr = float(getattr(args, "gen_lr", 0.05) if getattr(args, "gen_lr", None) is not None else 0.05)
        args.gen_momentum = float(
            getattr(args, "gen_momentum", 0.9) if getattr(args, "gen_momentum", None) is not None else 0.9
        )
        args.ssim_threshold = float(
            getattr(args, "ssim_threshold", 0.995) if getattr(args, "ssim_threshold", None) is not None else 0.995
        )

        args.trigger_img_from = getattr(args, "trigger_img_from", None) or "target"
        args.trigger_selection_mode = getattr(args, "trigger_selection_mode", None) or "fixed"
        args.save_debug_examples = int(
            getattr(args, "save_debug_examples", 10) if getattr(args, "save_debug_examples", None) is not None else 10
        )

        if args.attack_label_trans != "all2one":
            raise ValueError("This implementation currently supports all2one only.")

        if len(args.trigger_layers) != len(args.trigger_layer_weights):
            raise ValueError("len(trigger_layers) must equal len(trigger_layer_weights)")

        if args.trigger_img_from not in ["target", "any"]:
            raise ValueError("trigger_img_from must be 'target' or 'any'")

        if args.trigger_selection_mode not in ["fixed", "random"]:
            raise ValueError("trigger_selection_mode must be 'fixed' or 'random'")

        return args

    def _build_trigger_pool(self, clean_train_dataset_targets, args):
        targets = np.array(clean_train_dataset_targets)
        if args.trigger_img_from == "target":
            pool = np.where(targets == args.attack_target)[0]
        else:
            pool = np.arange(len(targets))

        if len(pool) == 0:
            raise RuntimeError("Trigger pool is empty.")
        return pool.tolist()

    def _make_generator(self, args):
        gen_device = _get_primary_device(getattr(args, "device", "cuda:0"))
        logging.info(f"Offline trigger generation device: {gen_device}")

        generator = HiddenFeatureTriggerGenerator(
            device=gen_device,
            input_height=getattr(args, "input_height", 32),
            input_width=getattr(args, "input_width", 32),
            content_layer=getattr(args, "content_layer", "conv2_2"),
            trigger_layers=getattr(args, "trigger_layers", ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]),
            trigger_layer_weights=getattr(args, "trigger_layer_weights", [1.0, 0.8, 0.5, 0.3, 0.1]),
            alpha=getattr(args, "alpha", 1e-6),
            beta=getattr(args, "beta", 1.0),
            max_iter=getattr(args, "max_iter", 100),
            gen_lr=getattr(args, "gen_lr", 0.05),
            gen_momentum=getattr(args, "gen_momentum", 0.9),
            ssim_threshold=getattr(args, "ssim_threshold", 0.995),
            vgg_weights_path=getattr(args, "vgg_weights_path", None),
        )
        return generator

    def _save_example(self, folder, idx, content_img, trigger_img, poisoned_img):
        os.makedirs(folder, exist_ok=True)
        content_pil = _ensure_pil(content_img)
        trigger_pil = _ensure_pil(trigger_img)
        poisoned_pil = _ensure_pil(poisoned_img)

        content_pil.save(os.path.join(folder, f"{idx:04d}_content.png"))
        trigger_pil.save(os.path.join(folder, f"{idx:04d}_trigger.png"))
        poisoned_pil.save(os.path.join(folder, f"{idx:04d}_poisoned.png"))

    def stage1_non_training_data_prepare(self):
        logging.info("stage1 start")
        assert "args" in self.__dict__
        args = self.args

        train_dataset_without_transform, \
        train_img_transform, \
        train_label_transform, \
        test_dataset_without_transform, \
        test_img_transform, \
        test_label_transform, \
        clean_train_dataset_with_transform, \
        clean_train_dataset_targets, \
        clean_test_dataset_with_transform, \
        clean_test_dataset_targets = self.benign_prepare()

        bd_label_transform = bd_attack_label_trans_generate(args)

        train_poison_index = generate_poison_index_from_label_transform(
            clean_train_dataset_targets,
            label_transform=bd_label_transform,
            train=True,
            pratio=args.pratio if "pratio" in args.__dict__ else None,
            p_num=args.p_num if "p_num" in args.__dict__ else None,
        )
        torch.save(train_poison_index, args.save_path + "/train_poison_index_list.pickle")

        test_poison_index = generate_poison_index_from_label_transform(
            clean_test_dataset_targets,
            label_transform=bd_label_transform,
            train=False,
        )

        # IMPORTANT:
        # initialize with all-zero poison indicator, then manually fill poisoned samples
        empty_train_poison_indicator = np.zeros(len(train_dataset_without_transform), dtype=np.int64)
        empty_test_poison_indicator = np.zeros(len(test_dataset_without_transform), dtype=np.int64)

        bd_train_dataset = prepro_cls_DatasetBD_v2(
            deepcopy(train_dataset_without_transform),
            poison_indicator=empty_train_poison_indicator,
            save_folder_path=f"{args.save_path}/bd_train_dataset",
        )

        bd_test_dataset = prepro_cls_DatasetBD_v2(
            deepcopy(test_dataset_without_transform),
            poison_indicator=empty_test_poison_indicator,
            save_folder_path=f"{args.save_path}/bd_test_dataset",
        )

        trigger_pool = self._build_trigger_pool(clean_train_dataset_targets, args)
        logging.info(f"Trigger pool size = {len(trigger_pool)}")

        rng = np.random.default_rng(int(getattr(args, "random_seed", 0)))
        generator = self._make_generator(args)

        # fixed trigger image version
        fixed_trigger_index = None
        fixed_trigger_img = None
        if args.trigger_selection_mode == "fixed":
            fixed_trigger_index = int(rng.choice(trigger_pool))
            fixed_trigger_img, _ = train_dataset_without_transform[fixed_trigger_index]
            logging.info(f"Fixed trigger image index = {fixed_trigger_index}")

        train_poison_ids = np.where(train_poison_index == 1)[0].tolist()
        test_poison_ids = np.where(test_poison_index == 1)[0].tolist()

        logging.info(f"Generate {len(train_poison_ids)} poisoned train samples offline")
        logging.info(f"Generate {len(test_poison_ids)} poisoned test samples offline")

        train_ssim_scores, train_mse_scores, train_psnr_scores = [], [], []
        test_ssim_scores, test_mse_scores, test_psnr_scores = [], [], []

        debug_folder = os.path.join(args.save_path, "debug_examples")
        debug_saved = 0

        # -------- train --------
        for selected_index in tqdm(train_poison_ids, desc="offline_gen_train"):
            content_img, label = train_dataset_without_transform[selected_index]

            if args.trigger_selection_mode == "fixed":
                trigger_img = fixed_trigger_img
            else:
                trigger_index = int(rng.choice(trigger_pool))
                trigger_img, _ = train_dataset_without_transform[trigger_index]

            poisoned_img, cur_ssim = generator.generate(content_img, trigger_img)

            x_clean = to_tensor(_ensure_pil(content_img)).unsqueeze(0)
            x_poison = to_tensor(_ensure_pil(poisoned_img)).unsqueeze(0)

            cur_mse = calc_mse_tensor(x_clean, x_poison)
            cur_psnr = calc_psnr_tensor(x_clean, x_poison)

            train_ssim_scores.append(cur_ssim)
            train_mse_scores.append(cur_mse)
            train_psnr_scores.append(cur_psnr)

            bd_label = bd_label_transform(label)
            bd_train_dataset.set_one_bd_sample(
                selected_index,
                poisoned_img,
                bd_label,
                label,
            )

            if debug_saved < args.save_debug_examples:
                self._save_example(debug_folder, debug_saved, content_img, trigger_img, poisoned_img)
                debug_saved += 1

        # -------- test --------
        for selected_index in tqdm(test_poison_ids, desc="offline_gen_test"):
            content_img, label = test_dataset_without_transform[selected_index]

            if args.trigger_selection_mode == "fixed":
                trigger_img = fixed_trigger_img
            else:
                trigger_index = int(rng.choice(trigger_pool))
                trigger_img, _ = train_dataset_without_transform[trigger_index]

            poisoned_img, cur_ssim = generator.generate(content_img, trigger_img)

            x_clean = to_tensor(_ensure_pil(content_img)).unsqueeze(0)
            x_poison = to_tensor(_ensure_pil(poisoned_img)).unsqueeze(0)

            cur_mse = calc_mse_tensor(x_clean, x_poison)
            cur_psnr = calc_psnr_tensor(x_clean, x_poison)

            test_ssim_scores.append(cur_ssim)
            test_mse_scores.append(cur_mse)
            test_psnr_scores.append(cur_psnr)

            bd_label = bd_label_transform(label)
            bd_test_dataset.set_one_bd_sample(
                selected_index,
                poisoned_img,
                bd_label,
                label,
            )

        # ASR evaluation only on poisoned subset
        bd_test_dataset.subset(np.where(test_poison_index == 1)[0])

        if len(train_ssim_scores) > 0:
            logging.info(
                f"[Train poison metrics] SSIM={np.mean(train_ssim_scores):.6f}, "
                f"MSE={np.mean(train_mse_scores):.6f}, "
                f"PSNR={np.mean(train_psnr_scores):.6f}"
            )
        if len(test_ssim_scores) > 0:
            logging.info(
                f"[Test poison metrics] SSIM={np.mean(test_ssim_scores):.6f}, "
                f"MSE={np.mean(test_mse_scores):.6f}, "
                f"PSNR={np.mean(test_psnr_scores):.6f}"
            )

        bd_train_dataset_with_transform = dataset_wrapper_with_transform(
            bd_train_dataset,
            train_img_transform,
            train_label_transform,
        )

        bd_test_dataset_with_transform = dataset_wrapper_with_transform(
            bd_test_dataset,
            test_img_transform,
            test_label_transform,
        )

        self.stage1_results = (
            clean_train_dataset_with_transform,
            clean_test_dataset_with_transform,
            bd_train_dataset_with_transform,
            bd_test_dataset_with_transform,
        )

        del generator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def stage2_training(self):
        logging.info("stage2 start")
        assert "args" in self.__dict__
        args = self.args

        clean_train_dataset_with_transform, \
        clean_test_dataset_with_transform, \
        bd_train_dataset_with_transform, \
        bd_test_dataset_with_transform = self.stage1_results

        self.net = generate_cls_model(
            model_name=args.model,
            num_classes=args.num_classes,
            image_size=args.img_size[0],
        )

        self.device = _get_primary_device(args.device)

        if isinstance(args.device, str) and "," in args.device and torch.cuda.is_available():
            self.net = torch.nn.DataParallel(
                self.net,
                device_ids=[int(i) for i in args.device[5:].split(",")]
            )

        self.net = self.net.to(self.device, non_blocking=args.non_blocking)

        trainer = BackdoorModelTrainer(self.net)
        criterion = argparser_criterion(args)
        optimizer, scheduler = argparser_opt_scheduler(self.net, args)

        from torch.utils.data.dataloader import DataLoader

        trainer.train_with_test_each_epoch_on_mix(
            DataLoader(
                bd_train_dataset_with_transform,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True,
                pin_memory=args.pin_memory,
                num_workers=args.num_workers,
            ),
            DataLoader(
                clean_test_dataset_with_transform,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                pin_memory=args.pin_memory,
                num_workers=args.num_workers,
            ),
            DataLoader(
                bd_test_dataset_with_transform,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                pin_memory=args.pin_memory,
                num_workers=args.num_workers,
            ),
            args.epochs,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            frequency_save=args.frequency_save,
            save_folder_path=args.save_path,
            save_prefix="attack",
            amp=args.amp,
            prefetch=args.prefetch,
            prefetch_transform_attr_name="ori_image_transform_in_loading",
            non_blocking=args.non_blocking,
        )

        save_attack_result(
            model_name=args.model,
            num_classes=args.num_classes,
            model=trainer.model.cpu().state_dict(),
            data_path=args.dataset_path,
            img_size=args.img_size,
            clean_data=args.dataset,
            bd_train=bd_train_dataset_with_transform,
            bd_test=bd_test_dataset_with_transform,
            save_path=args.save_path,
        )


if __name__ == "__main__":
    attack = InvisibleTrigger()
    parser = argparse.ArgumentParser(description=sys.argv[0])

    parser = attack.set_args(parser)
    parser = attack.set_bd_args(parser)

    args = parser.parse_args()

    logging.debug("bd yaml first, then normal yaml")
    attack.add_bd_yaml_to_args(args)
    attack.add_yaml_to_args(args)

    args = attack.process_args(args)
    attack.prepare(args)
    attack.stage1_non_training_data_prepare()
    attack.stage2_training()