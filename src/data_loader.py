import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np

class CityscapesDataset(Dataset):
    """
    Custom dataset class for loading Cityscapes images and segmentation masks.
    """
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask, dtype=np.int64)  # Convert to class indices
        
        if self.transform:
            image = self.transform(image)
            mask = torch.tensor(mask, dtype=torch.long)
        
        return image, mask

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    dataset = CityscapesDataset("data/images", "data/masks", transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    sample_image, sample_mask = next(iter(dataloader))
    print(sample_image.shape, sample_mask.shape)
