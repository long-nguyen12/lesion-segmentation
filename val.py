import argparse
from glob import glob

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from mmseg import __version__
from mmseg.models.segmentors import LesionSegmentation as UNet
from eval_func import Fmeasure_calu, StructureMeasure, EnhancedMeasure, MAE
import os


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
            img_aug = self.transform(image=image)
            image = img_aug["image"]

        if self.mask_transform is not None:
            mask_aug = self.mask_transform(image=mask)
            mask = mask_aug["image"]

        image = image.astype("float32") / 255
        image = image.transpose((2, 0, 1))

        mask = mask[:, :, np.newaxis]
        mask = mask.astype("float32") / 255
        mask = mask.transpose((2, 0, 1))

        img_name = img_path.split("\\")[-1]

        return np.asarray(image), np.asarray(mask), img_name


epsilon = 1e-7


def recall_np(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + epsilon)
    return recall


def precision_np(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + epsilon)
    return precision


def dice_np(y_true, y_pred):
    precision = precision_np(y_true, y_pred)
    recall = recall_np(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + epsilon))


def iou_np(y_true, y_pred):
    intersection = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / (union + epsilon)


def get_scores(gts, prs, imgs, ds=None):
    mean_precision = 0
    mean_recall = 0
    mean_iou = 0
    mean_dice = 0
    s_measure = 0
    e_measure = 0
    mae = 0
    f_measure = 0
    for gt, pr, im in zip(gts, prs, imgs):
        mean_precision += precision_np(gt, pr)
        mean_recall += recall_np(gt, pr)
        mean_iou += iou_np(gt, pr)
        mean_dice += dice_np(gt, pr)
        s_measure += StructureMeasure(pr, gt)
        e_measure += EnhancedMeasure(pr, gt)
        mae += MAE(pr, gt)
        f_measure += Fmeasure_calu(pr, gt)
    mean_precision /= len(gts)
    mean_recall /= len(gts)
    mean_iou /= len(gts)
    mean_dice /= len(gts)
    s_measure /= len(gts)
    e_measure /= len(gts)
    f_measure /= len(gts)
    mae /= len(gts)

    print(
        "scores: dice={}, miou={}, precision={}, recall={}, s_measure={}, e_measure={}, f_measure={}, mae={}".format(
            mean_dice,
            mean_iou,
            mean_precision,
            mean_recall,
            s_measure,
            e_measure,
            f_measure,
            mae,
        )
    )

    return mean_iou, mean_dice, mean_precision, mean_recall


@torch.no_grad()
def inference(model, test_loader, device, ds=None):
    print("#" * 20)
    torch.cuda.empty_cache()
    model.to(device)
    model.eval()
    gts = []
    prs = []
    imgs = []
    for i, pack in enumerate(test_loader, start=1):
        image, gt, name = pack
        gt = gt[0][0]
        gt = np.asarray(gt, np.float32)
        image = image.to(device)
        res = model(image)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        gts.append(gt)
        prs.append(res)
        imgs.append(name)

    return get_scores(gts, prs, imgs, ds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, default="")
    parser.add_argument(
        "--test_path", type=str, default="./data/TestDataset/", help="path to dataset"
    )
    parser.add_argument(
        "--init_trainsize", type=str, default=352, help="path to dataset"
    )
    parser.add_argument("--train_save", type=str, default="colonnext")
    args = parser.parse_args()

    device = torch.device("cpu")

    ds = ["CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "Kvasir", "CVC-300"]
    for _ds in ds:
        model = UNet(
            backbone=dict(type="MSCAN"),
            decode_head=dict(
                type="MLPPanHead",
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
            pretrained="pretrained/mscan_b.pth",
        ).to(device)

        # x = torch.randn(1, 3, 352, 352)
        # CalParams(model, x)

        checkpoint = torch.load(
            f"snapshots/{args.train_save}/last.pth", map_location="cpu"
        )

        model.load_state_dict(checkpoint, strict=True)
        print(f"Trained on {_ds}")
        data_path = f"{args.test_path}/{_ds}/"
        # inference(model, data_path, device, args)

        X_test = glob("{}/images/*".format(data_path))
        X_test.sort()
        y_test = glob("{}/masks/*".format(data_path))
        y_test.sort()

        transform = A.Compose(
            [
                A.Resize(height=352, width=352),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        mask_transform = A.Compose(
            [
                A.Resize(height=352, width=352),
            ]
        )

        test_dataset = Dataset(
            X_test, y_test, transform=transform, mask_transform=mask_transform
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False, pin_memory=True, drop_last=True
        )

        save_path = "worst_results/{}/".format(_ds)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        else:
            print("Save path existed")

        inference(model, test_loader, device, _ds)
