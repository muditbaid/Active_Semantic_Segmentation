import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset_loader import CityscapesDataset
from unet_model import UNet

def evaluate_model(model, dataloader):
    """
    Evaluate the model on test data and compute mean IoU.
    """
    model.eval()
    iou_scores = []
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.cuda(), masks.cuda()
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            intersection = torch.logical_and(preds, masks).sum().item()
            union = torch.logical_or(preds, masks).sum().item()
            iou = intersection / union if union != 0 else 0
            iou_scores.append(iou)
    print(f"Mean IoU: {sum(iou_scores) / len(iou_scores)}")

if __name__ == "__main__":
    dataset = CityscapesDataset("data/images", "data/masks")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    model = UNet().cuda()
    model.load_state_dict(torch.load("U-Net.pth"))
    evaluate_model(model, dataloader)
