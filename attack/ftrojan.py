'''
PyTorch Implementation of FTrojan (Frequency Domain Backdoor Attack)
Based on: https://github.com/SoftWiser-group/FTrojan
'''

import argparse
import sys
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from tqdm import tqdm

sys.path = ["./"] + sys.path

from attack.badnet import BadNet, add_common_attack_args
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2, dataset_wrapper_with_transform


class FTrojan(BadNet):
    def __init__(self):
        super(FTrojan, self).__init__()

    def set_bd_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = add_common_attack_args(parser)
        # FTrojan specific arguments
        parser.add_argument('--bd_yaml_path', type=str, default='./config/attack/ftrojan/cifar10.yaml',
                            help='path for yaml file provide additional default attributes')
        parser.add_argument('--yuv', type=bool, default=True, help='Whether to attack in YUV space')
        parser.add_argument('--channel_list', type=int, nargs='+', default=[1, 2],
                            help='Channels to attack (e.g., 1 2 for UV)')
        parser.add_argument('--magnitude', type=float, default=30.0, help='Magnitude of frequency injection')
        parser.add_argument('--window_size', type=int, default=32, help='DCT window size')
        parser.add_argument('--pos_list', type=str, default='[(31, 31), (15, 15)]',
                            help='Frequency positions to inject')
        return parser

    def process_args(self, args):
        args.terminal_info = sys.argv
        args.num_classes = 10
        args.input_height, args.input_width, args.input_channel = args.img_size
        # Parse pos_list from string to list of tuples if needed,
        # but usually YAML handles lists well. We assume it comes as a list or string representation.
        if isinstance(args.pos_list, str):
            args.pos_list = eval(args.pos_list)
        return args

    # =========================================================
    # FTrojan Core Logic (Replica of SoftWiser-group/FTrojan)
    # =========================================================

    def rgb2yuv(self, x_rgb):
        # x_rgb: (N, H, W, 3) in [0, 255]
        x_yuv = np.zeros(x_rgb.shape, dtype=np.float32)
        for i in range(x_rgb.shape[0]):
            # cv2 expects uint8 for correct color conversion logic usually, or float 0-1
            # The original code casts to uint8 before conversion
            img = cv2.cvtColor(x_rgb[i].astype(np.uint8), cv2.COLOR_RGB2YCrCb)
            x_yuv[i] = img
        return x_yuv

    def yuv2rgb(self, x_yuv):
        x_rgb = np.zeros(x_yuv.shape, dtype=np.float32)
        for i in range(x_yuv.shape[0]):
            img = cv2.cvtColor(x_yuv[i].astype(np.uint8), cv2.COLOR_YCrCb2RGB)
            x_rgb[i] = img
        return x_rgb

    def dct_transform(self, x_train, window_size):
        # x_train: (N, C, H, W)
        # Input needs to be permuted to (N, H, W, C) for cv2? No, cv2.dct takes 2D array.
        # Original code logic:
        # 1. Transpose to (idx, ch, w, h) -> done by caller?
        # Let's stick to (N, C, H, W) layout for PyTorch convenience, but convert internally.

        N, C, H, W = x_train.shape
        x_dct = np.zeros_like(x_train, dtype=np.float32)

        for i in range(N):
            for ch in range(C):
                for w in range(0, H, window_size):
                    for h in range(0, W, window_size):
                        block = x_train[i, ch, w:w + window_size, h:h + window_size].astype(np.float32)
                        x_dct[i, ch, w:w + window_size, h:h + window_size] = cv2.dct(block)
        return x_dct

    def idct_transform(self, x_train, window_size):
        N, C, H, W = x_train.shape
        x_idct = np.zeros_like(x_train, dtype=np.float32)

        for i in range(N):
            for ch in range(C):
                for w in range(0, H, window_size):
                    for h in range(0, W, window_size):
                        block = x_train[i, ch, w:w + window_size, h:h + window_size].astype(np.float32)
                        x_idct[i, ch, w:w + window_size, h:h + window_size] = cv2.idct(block)
        return x_idct

    def inject_trigger(self, x_train, args):
        # x_train: (N, C, H, W) normalized to [0, 1] usually, but FTrojan operates on [0, 255]
        # We need to handle the scale carefully.

        # 1. Scale to 0-255
        x_train = x_train * 255.0

        # 2. Permute to (N, H, W, C) for Color Conversion
        x_train = np.transpose(x_train, (0, 2, 3, 1))  # N, H, W, C

        # 3. YUV Conversion
        if args.yuv:
            x_train = self.rgb2yuv(x_train)

        # 4. Permute back to (N, C, H, W) for DCT processing loop
        x_train = np.transpose(x_train, (0, 3, 1, 2))  # N, C, H, W

        # 5. DCT
        x_train = self.dct_transform(x_train, args.window_size)

        # 6. Inject Frequency Trigger
        # channel_list index: if YUV, 0=Y, 1=U, 2=V.
        for i in range(x_train.shape[0]):
            for ch in args.channel_list:
                for w in range(0, x_train.shape[2], args.window_size):
                    for h in range(0, x_train.shape[3], args.window_size):
                        for pos in args.pos_list:
                            # pos is (u, v) frequency coordinate
                            # Ensure we don't go out of bounds
                            if w + pos[0] < x_train.shape[2] and h + pos[1] < x_train.shape[3]:
                                x_train[i, ch, w + pos[0], h + pos[1]] += args.magnitude

        # 7. Inverse DCT
        x_train = self.idct_transform(x_train, args.window_size)

        # 8. Permute to (N, H, W, C) for Color Conversion
        x_train = np.transpose(x_train, (0, 2, 3, 1))

        # 9. Inverse YUV
        if args.yuv:
            x_train = self.yuv2rgb(x_train)

        # 10. Scale back to 0-1 and Clip
        x_train = x_train / 255.0
        x_train = np.clip(x_train, 0, 1)

        # 11. Final permute to (N, C, H, W) to match PyTorch format if needed,
        # but Dataset wrappers usually expect HWC for PIL or C H W for tensor.
        # We will return N H W C (numpy) for compatibility with ToPILImage in generation loop.
        # Wait, let's look at `stage1` logic below.
        return x_train

    def stage1_non_training_data_prepare(self):
        print(f"Stage 1 Start: Preparing FTrojan Data")
        args = self.args

        # Load Benign Data
        _, _, _, _, _, _, \
            clean_train_dataset_with_transform, clean_train_dataset_targets, \
            clean_test_dataset_with_transform, clean_test_dataset_targets = self.benign_prepare()

        # Select indices to poison
        train_poison_index = np.zeros(len(clean_train_dataset_targets), dtype=int)
        p_num = int(len(clean_train_dataset_targets) * args.pratio)
        train_poison_index[np.random.permutation(len(clean_train_dataset_targets))[:p_num]] = 1

        test_poison_index = np.ones(len(clean_test_dataset_targets), dtype=int)

        # Helper to generate poisoned dataset
        def create_ftrojan_ds(base_dataset, poison_idx, is_train):
            save_dir = f"{args.save_path}/{'bd_train' if is_train else 'bd_test'}"

            # Use dummy transform trick to initialize wrapper
            bd_ds = prepro_cls_DatasetBD_v2(
                base_dataset,
                poison_indicator=poison_idx,
                bd_image_pre_transform=lambda x, y, z: x,  # Dummy
                bd_label_pre_transform=None,
                save_folder_path=save_dir
            )

            # Load all data to memory for processing
            loader = DataLoader(base_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
            to_pil = ToPILImage()

            print(f"Injecting FTrojan Triggers (Train={is_train})...")

            cnt = 0
            # Collect all images that need poisoning
            # Optimization: Process batch by batch

            for imgs, labels in tqdm(loader):
                # imgs: Tensor (B, C, H, W) [0, 1]
                # Convert to numpy
                imgs_np = imgs.numpy()

                # Check which samples in this batch are poisoned
                batch_poison_indices = poison_idx[cnt: cnt + len(labels)]

                if np.sum(batch_poison_indices) > 0:
                    # Only process poisoned samples to save time
                    indices_in_batch = np.where(batch_poison_indices == 1)[0]
                    subset_imgs = imgs_np[indices_in_batch]

                    # Apply FTrojan Injection
                    # inject_trigger returns (N, H, W, C)
                    poisoned_imgs = self.inject_trigger(subset_imgs, args)

                    # Save back
                    sub_cnt = 0
                    for local_idx in indices_in_batch:
                        global_idx = cnt + local_idx
                        # poisoned_imgs[sub_cnt] is (H, W, C) numpy array
                        p_img = poisoned_imgs[sub_cnt]

                        # Convert to uint8 for PIL
                        p_img_uint8 = (p_img * 255).astype(np.uint8)
                        pil_img = to_pil(p_img_uint8)  # ToPILImage handles HWC uint8

                        bd_ds.set_one_bd_sample(
                            selected_index=global_idx,
                            img=pil_img,
                            bd_label=args.attack_target,
                            label=labels[local_idx].item()
                        )
                        sub_cnt += 1

                cnt += len(labels)

            return bd_ds

        bd_train = create_ftrojan_ds(clean_train_dataset_with_transform, train_poison_index, True)
        bd_test = create_ftrojan_ds(clean_test_dataset_with_transform, test_poison_index, False)

        _, train_tf, _, _, test_tf, _, _, _, _, _ = self.benign_prepare()
        self.stage1_results = clean_train_dataset_with_transform, clean_test_dataset_with_transform, \
            dataset_wrapper_with_transform(bd_train, train_tf, None), \
            dataset_wrapper_with_transform(bd_test, test_tf, None)


if __name__ == '__main__':
    attack = FTrojan()
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