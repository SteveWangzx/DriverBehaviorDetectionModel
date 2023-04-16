"""
    This script contains the process of using PyTorch to train the model based on Res-Net 18

    Author: Zhixiang Wang
    Date: April 13, 2023
"""

import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import PIL.Image
import random

# define the device for training -- 3050ti laptop GPU in my device
device = torch.device("cuda")

# Transformations for input images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Dataset
df = pd.read_csv("driver_behavior.csv")

# Dictionary for labels
dic = {
    'other_activities': 4,
    'safe_driving': 0,
    'talking_phone': 1,
    'texting_phone': 2,
    'turning': 3
}


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df_class, transform_class, device_class):
        self.df = df_class
        self.transform = transform_class
        self.device = device_class

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = row['path']
        label = row['activity_type']
        label = dic[label]
        try:
            image = PIL.Image.open(image_path).convert('RGB')
        except PIL.UnidentifiedImageError:
            # skip this image if it can't be read
            return None, None
        image = self.transform(image)
        image = image.to(self.device)
        return image, label


dataset = CustomDataset(df, transform, device)
print(dataset.__len__())

# Create data loaders
batch_size = 32
train_size = int(0.8 * dataset.__len__())
val_size = dataset.__len__() - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Load the model
resnet = models.resnet18(weights='DEFAULT')
num_classes = 5
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
resnet = resnet.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)

optimizer = optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        images, labels = data
        if images is None:
            continue  # skip unreadable image
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = resnet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

# Validation
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = resnet(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

# Save the model
scripted_model = torch.jit.script(resnet)
scripted_model.save('driver_behavior_model_v1.pt')
