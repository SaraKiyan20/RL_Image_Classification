
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import STL10
from torchvision.models import resnet18, ResNet18_Weights


# -------------------------
# Configuration
# -------------------------

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -------------------------
# Image transformations
# -------------------------

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


# -------------------------
# Random-region dataset
# -------------------------

class RandomRegionDataset(Dataset):

    def __init__(self, dataset, indices, transform, training):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.training = training

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):

        original_index = self.indices[position]

        image, label = self.dataset[original_index]

        # Select one of the nine regions
        if self.training:
            region = random.randrange(9)
        else:
            rng = random.Random(original_index + 42)
            region = rng.randrange(9)

        row = region // 3
        col = region % 3

        left = col * 32
        top = row * 32

        right = left + 32
        bottom = top + 32

        image = image.crop((left, top, right, bottom))

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# -------------------------
# Load training dataset
# -------------------------

print("\nLoading STL-10...")

base_dataset = STL10(
    root="data",
    split="train",
    download=False
)

labels = torch.tensor(base_dataset.labels)

# Recreate the stratified 4000/1000 split
generator = torch.Generator().manual_seed(42)

train_indices = []
val_indices = []

for class_id in range(10):

    class_indices = torch.where(labels == class_id)[0]

    shuffled = class_indices[
        torch.randperm(len(class_indices), generator=generator)
    ]

    train_indices.extend(shuffled[:400].tolist())
    val_indices.extend(shuffled[400:].tolist())


train_dataset = RandomRegionDataset(
    base_dataset,
    train_indices,
    train_transform,
    training=True
)

val_dataset = RandomRegionDataset(
    base_dataset,
    val_indices,
    val_transform,
    training=False
)


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


# -------------------------
# Load pretrained ResNet18
# -------------------------

print("\nLoading ResNet18...")

model = resnet18(weights=ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 10)

model = model.to(device)


# -------------------------
# Training setup
# -------------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE
)

best_accuracy = 0.0


# -------------------------
# Training loop
# -------------------------

print("\nStarting random-region training...")

for epoch in range(EPOCHS):

    model.eval()
    model.fc.train()

    running_loss = 0.0

    for images, labels_batch in train_loader:

        images = images.to(device)
        labels_batch = labels_batch.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels_batch)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / len(train_loader)

    # Validation
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels_batch in val_loader:

            images = images.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels_batch).sum().item()
            total += labels_batch.size(0)

    accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {average_loss:.4f} "
        f"Validation Accuracy: {accuracy:.2f}%"
    )

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        torch.save(
            model.state_dict(),
            "models/resnet18_random_region_best.pth"
        )


print("\nTraining completed!")
print(f"Best validation accuracy: {best_accuracy:.2f}%")
