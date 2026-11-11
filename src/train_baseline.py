
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import STL10
from torchvision.models import resnet18, ResNet18_Weights

# --------------------------------
# 1. Configuration
# --------------------------------

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# --------------------------------
# 2. Image transformations
# --------------------------------

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------
# 3. Load datasets
# --------------------------------

train_data = STL10(
    root="data",
    split="train",
    download=False,
    transform=train_transform
)

val_data = STL10(
    root="data",
    split="train",
    download=False,
    transform=val_transform
)

# --------------------------------
# 4. Stratified train/validation split
# --------------------------------

generator = torch.Generator().manual_seed(42)

train_indices = []
val_indices = []

for class_id in range(10):

    class_indices = [
        i for i, label in enumerate(train_data.labels)
        if label == class_id
    ]

    shuffled = torch.randperm(
        len(class_indices),
        generator=generator
    ).tolist()

    class_indices = [class_indices[i] for i in shuffled]

    train_indices.extend(class_indices[:400])
    val_indices.extend(class_indices[400:])

train_dataset = Subset(train_data, train_indices)
val_dataset = Subset(val_data, val_indices)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))

# --------------------------------
# 5. Load pretrained ResNet18
# --------------------------------

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

# Freeze the pretrained layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, 10)

model = model.to(device)

# --------------------------------
# 6. Loss and optimizer
# --------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE
)

# --------------------------------
# 7. Training loop
# --------------------------------

os.makedirs("models", exist_ok=True)

best_val_accuracy = 0.0

for epoch in range(EPOCHS):

    model.eval()
    model.fc.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_accuracy = 100 * correct / total
    train_loss = running_loss / len(train_loader)

    # --------------------------------
    # 8. Validation
    # --------------------------------

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_accuracy = 100 * val_correct / val_total

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(f"Training loss: {train_loss:.4f}")
    print(f"Training accuracy: {train_accuracy:.2f}%")
    print(f"Validation accuracy: {val_accuracy:.2f}%")

    # Save the best model
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "models/resnet18_stl10_best.pth"
        )

        print("Best model saved!")

print("\nTraining completed!")
print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
