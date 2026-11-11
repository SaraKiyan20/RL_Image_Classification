
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import STL10
from torchvision.models import resnet18


# -------------------------
# Configuration
# -------------------------

BATCH_SIZE = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -------------------------
# Image transformation
# -------------------------

test_transform = transforms.Compose([
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

    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):

        original_index = self.indices[position]

        image, label = self.dataset[original_index]

        # Reproduce the deterministic region selection
        rng = random.Random(original_index + 42)
        region = rng.randrange(9)

        row = region // 3
        col = region % 3

        left = col * 32
        top = row * 32
        right = left + 32
        bottom = top + 32

        image = image.crop((left, top, right, bottom))

        image = self.transform(image)

        return image, label


# -------------------------
# Load STL-10 test dataset
# -------------------------

print("\nLoading STL-10 test dataset...")

base_dataset = STL10(
    root="data",
    split="test",
    download=False
)

test_indices = list(range(len(base_dataset)))

test_dataset = RandomRegionDataset(
    base_dataset,
    test_indices,
    test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Test images:", len(test_dataset))


# -------------------------
# Load trained model
# -------------------------

print("\nLoading trained random-region model...")

model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)

model.load_state_dict(
    torch.load(
        "models/resnet18_random_region_best.pth",
        map_location=device,
        weights_only=True
    )
)

model = model.to(device)
model.eval()


# -------------------------
# Test evaluation
# -------------------------

confusion = torch.zeros(10, 10, dtype=torch.int64)

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        batch_counts = torch.bincount(
            labels * 10 + predictions,
            minlength=100
        ).reshape(10, 10)

        confusion += batch_counts.cpu()


# -------------------------
# Calculate metrics
# -------------------------

accuracy = 100 * correct / total

true_positives = confusion.diag().float()

precision = (
    true_positives /
    confusion.sum(dim=0).clamp(min=1)
)

recall = (
    true_positives /
    confusion.sum(dim=1).clamp(min=1)
)

f1 = (
    2 * precision * recall /
    (precision + recall).clamp(min=1e-12)
)

macro_f1 = f1.mean().item()


# -------------------------
# Print results
# -------------------------

print("\nRandom-Region Test Evaluation")
print("-----------------------------")

print("Test images:", total)
print(f"Test accuracy: {accuracy:.2f}%")
print(f"Macro-F1: {macro_f1:.4f}")

print("\nPer-class F1:")

for name, score in zip(base_dataset.classes, f1):
    print(f"{name}: {score.item():.4f}")

print("\nConfusion matrix:")
print(confusion)
