from train import LOAD_MODEL
from zmq import device
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import UNET
import os 
from utils import (
    load_checkpoint,
    get_test_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_EPOCHS = 1
NUM_WORKERS = 2
IMAGE_HEIGHT = 288 
IMAGE_WIDTH = 384
PIN_MEMORY = True
LOAD_MODEL = True
TEST_IMG_DIR = "data/test_images/"
TEST_MASK_DIR = "data/test_masks/"
UNET_MASKS_DIR = "data/predicted_Unet/"
if not os.path.exists(UNET_MASKS_DIR):
    os.mkdir(UNET_MASKS_DIR)
CHECKPOINT_PATH = "checkpoint/my_checkpoint.pth.tar"

def main():
    test_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),

        ],
    )
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    test_loader = get_test_loaders(
        test_img_dir=TEST_IMG_DIR,
        test_mask_dir=TEST_MASK_DIR, 
        batch_size=BATCH_SIZE,
        test_transform=test_transform,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,

    )

    if LOAD_MODEL:
        load_checkpoint(torch.load(CHECKPOINT_PATH), model)

    #check_accuracy(val_loader, model, device=DEVICE)
    # scaler = torch.cuda.amp.grad_scaler()

    for epoch in range(NUM_EPOCHS):

        # check accuracy
        print('Check accuracy')
        check_accuracy(test_loader, model, device=DEVICE)
    
        # print some examples to a folder
        print('save_predictions_as_imgs')        
        save_predictions_as_imgs(
            test_loader, model, folder=UNET_MASKS_DIR, device=DEVICE, pred=False
        )


if __name__ == "__main__":
    main()





