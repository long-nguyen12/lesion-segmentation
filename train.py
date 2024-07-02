import argparse
import os
from datetime import datetime
from glob import glob

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch.autograd import Variable

from mmseg import __version__
from mmseg.models.segmentors import LesionSegmentation as UNet
from schedulers import WarmupPolyLR
from utils import AvgMeter, clip_gradient, BceDiceLoss
from val import inference


class Dataset(torch.utils.data.Dataset):

    def __init__(self, img_paths, mask_paths, transform=None, mask_transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)

        if self.transform is not None:
            img_aug = self.transform(image=image, mask=mask)
            image = img_aug["image"]
            mask = img_aug["mask"]

        # if self.mask_transform is not None:
        #     mask_aug = self.mask_transform(image=mask)
        #     mask = mask_aug["image"]

        image = image.astype("float32") / 255
        image = image.transpose((2, 0, 1))

        mask = mask[:, :, np.newaxis]
        mask = mask.astype("float32") / 255
        mask = mask.transpose((2, 0, 1))
        return np.asarray(image), np.asarray(mask)


epsilon = 1e-7


def recall_m(y_true, y_pred):
    true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    possible_positives = torch.sum(torch.round(torch.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + epsilon)
    return recall


def precision_m(y_true, y_pred):
    true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    predicted_positives = torch.sum(torch.round(torch.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + epsilon)
    return precision


def dice_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + epsilon))


def iou_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return recall * precision / (recall + precision - recall * precision + epsilon)


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=100, help="epoch number")
    parser.add_argument("--init_lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--batchsize", type=int, default=32, help="training batch size")
    parser.add_argument(
        "--init_trainsize", type=int, default=256, help="training dataset size"
    )
    parser.add_argument(
        "--clip", type=float, default=0.5, help="gradient clipping margin"
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default="./data/isic-2017/training",
        help="path to train dataset",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="./data/isic-2017/validation",
        help="path to train dataset",
    )
    parser.add_argument("--train_save", type=str, default="lesion-seg")
    args = parser.parse_args()

    epochs = args.num_epochs
    save_path = "snapshots/{}/".format(args.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    else:
        print("Save path existed")

    device = torch.device("cuda")

    train_img_paths = []
    train_mask_paths = []
    train_img_paths = glob("{}/images/*".format(args.train_path))
    train_mask_paths = glob("{}/masks/*".format(args.train_path))
    train_img_paths.sort()
    train_mask_paths.sort()

    transform = A.Compose(
        [
            A.Resize(height=256, width=256),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(),
            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    mask_transform = A.Compose(
        [
            A.Resize(height=256, width=256),
        ]
    )

    train_dataset = Dataset(
        train_img_paths,
        train_mask_paths,
        transform=transform,
        mask_transform=mask_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    val_img_paths = []
    val_mask_paths = []
    val_img_paths = glob("{}/images/*".format(args.val_path))
    val_mask_paths = glob("{}/masks/*".format(args.val_path))
    val_img_paths.sort()
    val_mask_paths.sort()

    val_dataset = Dataset(
        val_img_paths,
        val_mask_paths,
        transform=transform,
        mask_transform=mask_transform,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )

    _total_step = len(train_loader)

    model = UNet(
        backbone=dict(type="PVTv2"),
        decode_head=dict(
            type="UPerHead",
            in_channels=[64, 128, 320, 512],
            in_index=[0, 1, 2, 3],
            channels=128,
            dropout_ratio=0.1,
            num_classes=1,
            norm_cfg=dict(type="BN", requires_grad=True),
            align_corners=False,
            loss_decode=dict(
                type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0
            ),
        ),
        neck=None,
        auxiliary_head=None,
        train_cfg=dict(),
        test_cfg=dict(mode="whole"),
        pretrained="pretrained/pvt_v2_b2.pth",
    ).to(device)

    # ---- flops and params ----
    params = model.parameters()
    optimizer = torch.optim.AdamW(
        params, args.init_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader) * args.num_epochs,
        eta_min=args.init_lr / 100,
    )

    start_epoch = 1

    best_iou = 0
    loss_record = AvgMeter()
    dice, iou = AvgMeter(), AvgMeter()
    criterion = BceDiceLoss()

    print("#" * 20, "Start Training", "#" * 20)
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        with torch.autograd.set_detect_anomaly(True):
            for i, pack in enumerate(train_loader, start=1):
                if epoch <= 1:
                    optimizer.param_groups[0]["lr"] = (
                        (epoch * i) / (1.0 * _total_step) * args.init_lr
                    )
                else:
                    lr_scheduler.step()

                optimizer.zero_grad()
                # ---- data prepare ----
                images, gts = pack

                images = images.cuda(non_blocking=True).float()
                gts = gts.cuda(non_blocking=True).float()

                # ---- forward ----
                map4, map3, map2, map1 = model(images)
                loss = (
                    criterion(map1, gts)
                    + criterion(map2, gts)
                    + criterion(map3, gts)
                    + criterion(map4, gts)
                )

                # ---- metrics ----
                dice_score = dice_m(map4, gts)
                iou_score = iou_m(map4, gts)
                # ---- backward ----
                loss.backward()
                clip_gradient(optimizer, args.clip)
                optimizer.step()
                # ---- recording loss ----

                loss_record.update(loss.data, args.batchsize)
                dice.update(dice_score.data, args.batchsize)
                iou.update(iou_score.data, args.batchsize)

                torch.cuda.synchronize()

        # ---- train visualization ----
        print(
            "{} Training Epoch [{:03d}/{:03d}], "
            "[loss: {:0.4f}, dice: {:0.4f}, iou: {:0.4f}]".format(
                datetime.now(),
                epoch,
                args.num_epochs,
                loss_record.show(),
                dice.show(),
                iou.show(),
            )
        )

        _miou, _, _, _ = inference(model, val_loader, device)
        if _miou > best_iou:
            best_iou = _miou
            ckpt_path = save_path + "best.pth"
            print("[Saving Best Checkpoint:]", ckpt_path)
            torch.save(model.state_dict(), ckpt_path)

        ckpt_path = save_path + "last.pth"
        print("[Saving Checkpoint:]", ckpt_path)
        torch.save(model.state_dict(), ckpt_path)
