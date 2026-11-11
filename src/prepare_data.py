
import torch
from torchvision import transforms
from torchvision.datasets import STL10
from torch.utils.data import DataLoader, random_split

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load STL-10 training dataset
full_train_dataset = STL10(
    root="data",
    split="train",
    download=False,
    transform=transform
)

# Split into training and validation sets
train_size = 4000
val_size = 1000

generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [train_size, val_size],
    generator=generator
)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

# Check the data
images, labels = next(iter(train_loader))

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Batch image shape:", images.shape)
print("Batch label shape:", labels.shape)
