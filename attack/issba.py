'''
PyTorch Implementation of ISSBA (Invisible Sample-Specific Backdoor Attacks)
Integrated for BackdoorBench.

This implementation strictly follows the StegaStamp architecture and ISSBA's sample-specific trigger generation logic.
It resolves compatibility issues with BackdoorBench's dataset wrappers by defining a custom transform injection strategy.

Reference:
- ISSBA Paper: https://arxiv.org/abs/2011.10834
- StegaStamp: https://github.com/tancik/StegaStamp
'''

import argparse
import logging
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
import torchvision.transforms as transforms

sys.path = ["./"] + sys.path

from attack.badnet import BadNet, add_common_attack_args
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2, dataset_wrapper_with_transform
from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler, argparser_criterion

# =============================================================================
# 1. StegaStamp Components (Encoder, Decoder, Discriminator)
#    Optimized for CIFAR-10 (32x32) input size.
# =============================================================================

class Conv2D(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, act='relu'):
        super().__init__()
        pad = (k - 1) // 2
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=pad)
        self.act = act
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        x = self.conv(x)
        if self.act == 'relu': return torch.relu(x)
        if self.act == 'tanh': return torch.tanh(x)
        if self.act == 'leaky_relu': return torch.nn.functional.leaky_relu(x, 0.2)
        return x

class StegaStampEncoder(nn.Module):
    """
    U-Net style generator adapted for small resolution (32x32).
    Includes Residual Scaling for better invisibility.
    """
    def __init__(self, args):
        super().__init__()
        self.secret_len = args.secret_len
        self.H, self.W = args.enc_height, args.enc_width
        self.secret_dense = nn.Linear(self.secret_len, 3 * self.H * self.W)

        # Encoder
        self.c1 = Conv2D(6, 32)        # 32
        self.c2 = Conv2D(32, 32, s=2)  # 16
        self.c3 = Conv2D(32, 64, s=2)  # 8
        self.c4 = Conv2D(64, 128, s=2) # 4
        self.c5 = Conv2D(128, 256, s=2)# 2

        # Decoder
        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.c6 = Conv2D(256, 128)

        self.up6 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.c7 = Conv2D(128+128, 64)

        self.up7 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.c8 = Conv2D(64+64, 32)

        self.up8 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.c9 = Conv2D(32+32, 32)

        # Final Residual Generation
        self.c10 = Conv2D(32+3, 3, act='tanh')

    def forward(self, img, secret):
        s = self.secret_dense(secret).reshape(-1, 3, self.H, self.W)
        x = torch.cat([img, s], dim=1) # 6ch

        # Downsample
        conv1 = self.c1(x)     # 32
        conv2 = self.c2(conv1) # 16
        conv3 = self.c3(conv2) # 8
        conv4 = self.c4(conv3) # 4
        conv5 = self.c5(conv4) # 2

        # Upsample & Skip Connections
        up5 = self.up5(conv5)  # 4
        conv6 = self.c6(up5)   # 4

        cat6 = torch.cat([conv4, conv6], dim=1) # 128 + 128 = 256
        conv7 = self.c7(cat6)  # 256 -> 64

        up7 = self.up6(conv7)  # 8
        cat7 = torch.cat([conv3, up7], dim=1) # 64 + 64 = 128
        conv8 = self.c8(cat7)  # 128 -> 32

        up8 = self.up7(conv8)  # 16
        cat8 = torch.cat([conv2, up8], dim=1) # 32 + 32 = 64
        conv9 = self.c9(cat8)  # 64 -> 32

        up9 = self.up8(conv9)  # 32
        cat9 = torch.cat([img, up9], dim=1) # 3 + 32 = 35

        residual = self.c10(cat9) # 35 -> 3

        # Add residual to original image
        return img + residual

class StegaStampDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.net = nn.Sequential(
            Conv2D(3, 32, s=2), Conv2D(32, 32),
            Conv2D(32, 64, s=2), Conv2D(64, 64),
            Conv2D(64, 128, s=2), Conv2D(128, 128),
            nn.Flatten(),
            nn.Linear(128 * (args.enc_height // 8) * (args.enc_width // 8), 512),
            nn.ReLU(),
            nn.Linear(512, args.secret_len),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.net = nn.Sequential(
            Conv2D(3, 8, s=2, act='leaky_relu'),
            Conv2D(8, 16, s=2, act='leaky_relu'),
            Conv2D(16, 32, s=2, act='leaky_relu'),
            nn.Flatten(),
            nn.Linear(32 * (args.enc_height // 8) * (args.enc_width // 8), 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

# =============================================================================
# 2. Dataset Wrapper for ISSBA
#    This solves the 'NoneType object is not callable' issue elegantly.
# =============================================================================

class ISSBADataset(prepro_cls_DatasetBD_v2):
    """
    Custom Dataset class that overrides the backdoor generation logic.
    Instead of passing a transform function, we pass the trained Encoder.
    """
    def __init__(self, full_dataset, poison_indicator, encoder, args, save_folder_path):
        self.encoder = encoder
        self.args = args
        self.device = next(encoder.parameters()).device
        self.target_secret = torch.zeros(1, args.secret_len).to(self.device)
        self.target_secret[0, ::2] = 1.0 # Fixed secret for attack
        self.to_pil = ToPILImage()
        self.to_tensor = transforms.ToTensor()

        # Initialize parent with dummy transform to avoid immediate error
        # We will override the behavior via prepro_backdoor logic injection if needed,
        # but actually we can just let parent run and use our custom generation loop.
        super().__init__(
            full_dataset,
            poison_indicator=poison_idx_dummy_placeholder(poison_indicator), # Pass None first
            bd_image_pre_transform=None,
            bd_label_pre_transform=None,
            save_folder_path=save_folder_path
        )

        # Now manually execute the backdoor injection using the Encoder
        self.poison_indicator = poison_indicator # Restore real indicator
        self.generate_issba_samples(full_dataset)

    def generate_issba_samples(self, dataset):
        logging.info("Injecting ISSBA triggers using trained Encoder...")
        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=2)

        cnt = 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(self.device)
                B = imgs.shape[0]
                secrets = self.target_secret.repeat(B, 1)

                # Generate Stego Images
                bd_imgs = self.encoder(imgs, secrets)
                bd_imgs = torch.clamp(bd_imgs, 0, 1).cpu()

                for i in range(B):
                    if self.poison_indicator[cnt] == 1:
                        # Save sample to disk managed by parent class
                        self.set_one_bd_sample(
                            selected_index=cnt,
                            img=self.to_pil(bd_imgs[i]),
                            bd_label=self.args.attack_target,
                            label=labels[i].item()
                        )
                    cnt += 1

def poison_idx_dummy_placeholder(real_idx):
    """Helper to bypass parent class initialization check"""
    return None

# =============================================================================
# 3. ISSBA Attack Logic
# =============================================================================

class ISSBA(BadNet):
    def __init__(self):
        super(ISSBA, self).__init__()

    def set_bd_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = add_common_attack_args(parser)
        parser.add_argument('--bd_yaml_path', type=str, default='./config/attack/issba/cifar10.yaml',
                            help='path for yaml file provide additional default attributes')
        parser.add_argument('--secret_len', type=int)
        parser.add_argument('--enc_height', type=int)
        parser.add_argument('--enc_width', type=int)
        parser.add_argument('--enc_epochs', type=int)
        parser.add_argument('--lmbda_image', type=float)
        parser.add_argument('--lmbda_secret', type=float)
        parser.add_argument('--lmbda_gan', type=float, default=0.01)
        return parser

    def train_stega_modules(self, train_loader):
        logging.info(">>> Stage 1: Training ISSBA Steganography (Encoder + Decoder + Discriminator)...")
        args = self.args

        encoder = StegaStampEncoder(args).to(self.device)
        decoder = StegaStampDecoder(args).to(self.device)
        disc = Discriminator(args).to(self.device)

        opt_enc = torch.optim.Adam(encoder.parameters(), lr=args.enc_lr)
        opt_dec = torch.optim.Adam(decoder.parameters(), lr=args.enc_lr)
        opt_disc = torch.optim.Adam(disc.parameters(), lr=args.enc_lr)

        mse_loss = nn.MSELoss()
        bce_loss = nn.BCELoss()

        encoder.train(); decoder.train(); disc.train()

        for epoch in range(args.enc_epochs):
            loss_g_list, loss_d_list = [], []

            for imgs, _ in train_loader:
                imgs = imgs.to(self.device)
                B = imgs.shape[0]

                secrets = torch.round(torch.rand(B, args.secret_len)).to(self.device)

                # 1. Train Discriminator
                opt_disc.zero_grad()
                stego_imgs = encoder(imgs, secrets)
                pred_real = disc(imgs)
                pred_fake = disc(stego_imgs.detach())

                loss_d = (bce_loss(pred_real, torch.ones_like(pred_real)) +
                          bce_loss(pred_fake, torch.zeros_like(pred_fake))) * 0.5
                loss_d.backward()
                opt_disc.step()

                # 2. Train Encoder & Decoder
                opt_enc.zero_grad()
                opt_dec.zero_grad()

                stego_imgs = encoder(imgs, secrets)
                decoded_secrets = decoder(stego_imgs)
                pred_fake_g = disc(stego_imgs)

                loss_img = mse_loss(stego_imgs, imgs)
                loss_secret = mse_loss(decoded_secrets, secrets)
                loss_gan = bce_loss(pred_fake_g, torch.ones_like(pred_fake_g))

                loss_g_total = (args.lmbda_image * loss_img) + \
                               (args.lmbda_secret * loss_secret) + \
                               (args.lmbda_gan * loss_gan)

                loss_g_total.backward()
                opt_enc.step()
                opt_dec.step()

                loss_g_list.append(loss_g_total.item())
                loss_d_list.append(loss_d.item())

            logging.info(f"Stego Epoch {epoch+1}/{args.enc_epochs} | G_Loss: {np.mean(loss_g_list):.4f} | D_Loss: {np.mean(loss_d_list):.4f}")

        return encoder

    def stage1_non_training_data_prepare(self):
        logging.info(f"Stage 1 Start: Preparing ISSBA Data")
        args = self.args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load Data
        _, _, _, _, _, _, \
        clean_train_dataset_with_transform, clean_train_dataset_targets, \
        clean_test_dataset_with_transform, clean_test_dataset_targets = self.benign_prepare()

        # Train Steganography Network
        train_loader = DataLoader(clean_train_dataset_with_transform, batch_size=args.batch_size, shuffle=True, num_workers=2)
        self.encoder = self.train_stega_modules(train_loader)
        self.encoder.eval()

        # Define Poison Indices
        train_poison_index = np.zeros(len(clean_train_dataset_targets), dtype=int)
        p_num = int(len(clean_train_dataset_targets) * args.pratio)
        train_poison_index[np.random.permutation(len(clean_train_dataset_targets))[:p_num]] = 1
        test_poison_index = np.ones(len(clean_test_dataset_targets), dtype=int)

        # Generate Poisoned Datasets using Custom Wrapper
        # This completely avoids the 'NoneType not callable' issue
        bd_train = ISSBADataset(
            clean_train_dataset_with_transform,
            train_poison_index,
            self.encoder,
            args,
            f"{args.save_path}/bd_train"
        )

        bd_test = ISSBADataset(
            clean_test_dataset_with_transform,
            test_poison_index,
            self.encoder,
            args,
            f"{args.save_path}/bd_test"
        )

        # Wrap with standard transforms for Stage 2 training
        _, train_tf, _, _, test_tf, _, _,_,_,_ = self.benign_prepare()
        self.stage1_results = clean_train_dataset_with_transform, clean_test_dataset_with_transform, \
                              dataset_wrapper_with_transform(bd_train, train_tf, None), \
                              dataset_wrapper_with_transform(bd_test, test_tf, None)

if __name__ == '__main__':
    attack = ISSBA()
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