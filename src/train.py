import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_loader import CityscapesDataset
from unet_model import UNet

# Define training function
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, masks in dataloader:
            images, masks = images.cuda(), masks.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")

if __name__ == "__main__":
    transform = None  # Define transformations if needed
    dataset = CityscapesDataset("data/images", "data/masks", transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model = UNet().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_model(model, dataloader, criterion, optimizer, num_epochs=10)
