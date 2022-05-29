from zmq import device
import torch 
import albumentations as A 
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim 
from model import UNET 
import timeit
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 10
NUM_WORKERS = 2
IMAGE_HEIGHT = 288 
IMAGE_WIDTH = 384
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"
SAVE_IMG = "data/saved_images/"
CHECKPOINT_PATH = "checkpoint/my_checkpoint.pth.tar"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets =targets.float().unsqueeze(1).to(device=DEVICE)

        starttime = timeit.default_timer()
        # forward
        print('forward')
        with torch.cuda.amp.autocast():
            print('predictions')
            predictions = model(data)
            print('loss_calcul')
            loss = loss_fn(predictions, targets)
        print('Loss: ', loss)
        print("The forward time is :", timeit.default_timer() - starttime)
        # backward
        print('backward')
        starttime = timeit.default_timer()
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        print("The backward time is :", timeit.default_timer() - starttime)

        # update tqdm loop
        loop.set_postfix(loss=loss.item())



def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomContrast(limit=0.6),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            # A.HueSaturationValue(val_shift_limit=50),
            ToTensorV2(),

        ],
    )

    val_transform = A.Compose(
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
    loss_fn = nn.BCEWithLogitsLoss() 
    # Note: 
    #  - we can use BinaryCrossEntropy but we should modify the output of the model in model.py from self.final_conv(x) to torch.sigmoid()
    #  - if we are using multiclass we change out_channels to Nb of outputs & loss_fn to cross entropy loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        train_dir=TRAIN_IMG_DIR,
        train_maskdir=TRAIN_MASK_DIR,
        val_dir=VAL_IMG_DIR,
        val_maskdir=VAL_MASK_DIR,
        batch_size=BATCH_SIZE,
        train_transform=train_transform,
        val_transform=val_transform,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    print('train_loader: ', len(train_loader.dataset), 'val_loader: ', len(val_loader.dataset))
    if LOAD_MODEL:
        print('Loading the model')
        load_checkpoint(torch.load(CHECKPOINT_PATH), model)

    #check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()
    print('Start training')
    for epoch in range(NUM_EPOCHS):
        print('Epoch: ', epoch)
        train_fn(train_loader,model, optimizer, loss_fn, scaler)
        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),           
        }
        print('saving checkpoint')
        save_checkpoint(checkpoint)

        # check accuracy
        print('Check accuracy')        
        check_accuracy(val_loader, model, device=DEVICE)
    
        # print some examples to a folder
        print('save_predictions_as_imgs')        

        save_predictions_as_imgs(
            val_loader, model, folder=SAVE_IMG, device=DEVICE
        )


if __name__ == "__main__":
    main()





