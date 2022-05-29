import torch 
import torchvision
from dataset import SkinDataset
from torch.utils.data import DataLoader 
import numpy as np 

def save_checkpoint(state, filename="checkpoint/my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = SkinDataset(image_dir=train_dir,mask_dir=train_maskdir,transform=train_transform,)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True,)

    val_ds = SkinDataset(image_dir=val_dir, mask_dir=val_maskdir, transform=val_transform,)

    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False,)

    return train_loader, val_loader

def get_test_loaders(
    test_img_dir,
    test_mask_dir,
    batch_size,
    test_transform,
    num_workers=4,
    pin_memory=True,
):
    test_ds = SkinDataset(image_dir=test_img_dir,mask_dir=test_mask_dir,transform=test_transform,)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True,)
    return test_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1) # we do this as the label doesn't have a channel
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum())/ (
                (preds + y).sum() + 1e-8)
    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(
    loader, model, folder="data/saved_images/", device="cuda", pred = False
):
    model.eval()
    if pred:
        for idx, x in enumerate(loader):
            x = x.to(device=device)
            with torch.no_grad():
                preds= torch.sigmoid(model(x))
                preds = (preds > 0.5).float()
            torchvision.utils.save_image(
                preds, f"{folder}/pred_{idx}.png"
            )
    else: 
        for idx, (x, y) in enumerate(loader):
            x = x.to(device=device)
            with torch.no_grad():
                preds= torch.sigmoid(model(x))
                preds = (preds > 0.5).float()
            torchvision.utils.save_image(
                preds, f"{folder}/pred_{idx}.png"
            )
            torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/GT_{idx}.png")

        model.train()

from typing import NamedTuple, Dict

class ConfusionItems(NamedTuple):
    TP: int
    TN: int
    FP: int
    FN: int


def compute_confusion_results(items: ConfusionItems):
    TP, TN, FP, FN = items
    return {
        "recall": TP / (TP + FN) if TP > 0 else 0.,
        "precision": TP / (TP + FP) if TP > 0 else 0.,
        "specificity": TN / (TN + FP) if TN > 0 else 0.,
        "accuracy": (TP + TN) / (TP + TN + FP + FN),
        "F1": _get_f_score(items, beta=1),
        "F2": _get_f_score(items, beta=2),
    }
def _get_f_score(items: ConfusionItems, beta: float = 1) -> float:
    TP, _, FP, FN = items
    if TP == 0:
        return 0.
    beta_2 = beta ** 2
    num = (1 + beta_2) * TP
    denom = num + beta_2 * FN + FP
    return num / denom



def _get_global_confusion_items(_actual_predicted):
    TP = np.diag(_actual_predicted[1:, 1:]).sum()
    TN = _actual_predicted[0, 0]
    FP_A = _actual_predicted[1:, 1:].sum() - TP
    FP_B = _actual_predicted[0, 1:].sum()
    FN = _actual_predicted[1:, 0].sum()
    return TP, TN, FP_A, FP_B, FN


def _compute_segment_metrics(TP, TN, FP_A, FP_B, FN) -> Dict[str, float]:
    intersection = TP
    union = TP + FP_A + FP_B + FN
    actual_cnt = TP + FP_A + FN + FP_B
    predicted_cnt = TP + FP_A + FP_B

    items = ConfusionItems(TP, TN, FP_A + FP_B, FN)
    results = compute_confusion_results(items)

    return {
        "IoU": intersection / union if intersection > 0 else 0.,
        "Dice": 2 * intersection / (actual_cnt + predicted_cnt) if intersection > 0 else 0.,
        "recall": results["recall"],
        "precision": results["precision"],
        "accuracy": results["accuracy"],
    }
