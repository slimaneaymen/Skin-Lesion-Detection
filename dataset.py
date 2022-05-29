import os 
from PIL import Image 
from torch.utils.data import Dataset
import numpy as np

class SkinDataset(Dataset):

    def __init__(self, image_dir, mask_dir, transform=None):
        # super(SkinDataset, self).__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir 
        self.transform = transform 
        self.images = os.listdir(image_dir)
        self.idx = []

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        if self.mask_dir is not None:
            mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_seg.png"))
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
            mask[mask == 255.0] = 1.0
        image = np.array(Image.open(img_path).convert("RGB"))


        if self.transform is not None:
            if self.mask_dir is not None:
                augmentations = self.transform(image=image, mask=mask)
                image = augmentations["image"]
                mask = augmentations["mask"]
                return image, mask
            else:
                augmentations = self.transform(image=image)
                image = augmentations["image"]
                return image

