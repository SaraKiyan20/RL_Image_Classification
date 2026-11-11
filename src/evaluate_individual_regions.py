
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from torchvision.datasets import STL10
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, f1_score


# -----------------------------
# 1. Settings
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# -----------------------------
# 2. Load test dataset
# -----------------------------

transform = transforms.Compose([
    transforms.ToTensor()
])

test_dataset = STL10(
    root="data",
    split="test",
    download=False,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False
)

class_names = test_dataset.classes


# -----------------------------
# 3. Load patch classifier
# -----------------------------

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    10
)

model.load_state_dict(
    torch.load(
        "models/resnet18_random_region_best.pth",
        map_location=device,
        weights_only=True
    )
)

model = model.to(device)
model.eval()

for param in model.parameters():
    param.requires_grad = False


# -----------------------------
# 4. Image normalization
# -----------------------------

imagenet_mean = torch.tensor(
    [0.485, 0.456, 0.406]
).view(1, 3, 1, 1).to(device)

imagenet_std = torch.tensor(
    [0.229, 0.224, 0.225]
).view(1, 3, 1, 1).to(device)


def normalize(images):
    return (images - imagenet_mean) / imagenet_std


# -----------------------------
# 5. Evaluate all regions
# -----------------------------

all_labels = []

region_predictions = [[] for _ in range(9)]

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        all_labels.extend(labels.tolist())

        batch_size = images.size(0)

        for row in range(3):

            for col in range(3):

                region_index = row * 3 + col

                # Extract the same region from every image
                patches = images[
                    :,
                    :,
                    row * 32:(row + 1) * 32,
                    col * 32:(col + 1) * 32
                ]

                # Resize patches
                patches = torch.nn.functional.interpolate(
                    patches,
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False
                )

                patches = normalize(patches)

                # Classify
                outputs = model(patches)

                predictions = outputs.argmax(dim=1)

                region_predictions[region_index].extend(
                    predictions.cpu().tolist()
                )


# -----------------------------
# 6. Calculate metrics
# -----------------------------

print("\nIndividual Region Evaluation")
print("----------------------------")

region_accuracies = []
region_f1_scores = []

for i in range(9):

    accuracy = accuracy_score(
        all_labels,
        region_predictions[i]
    )

    macro_f1 = f1_score(
        all_labels,
        region_predictions[i],
        average="macro"
    )

    region_accuracies.append(accuracy)
    region_f1_scores.append(macro_f1)

    print(
        f"Region {i + 1}: "
        f"Accuracy = {accuracy * 100:.2f}% | "
        f"Macro-F1 = {macro_f1:.4f}"
    )


# -----------------------------
# 7. Best fixed region
# -----------------------------

best_region = max(
    range(9),
    key=lambda i: region_accuracies[i]
)

print("\nBest Fixed Region")
print("----------------------------")

print(f"Region: {best_region + 1}")

print(
    f"Accuracy: "
    f"{region_accuracies[best_region] * 100:.2f}%"
)

print(
    f"Macro-F1: "
    f"{region_f1_scores[best_region]:.4f}"
)


# -----------------------------
# 8. Compare with DQN
# -----------------------------

print("\nComparison")
print("----------------------------")

print("DQN Accuracy: 70.80%")

print(
    f"Best Fixed Region Accuracy: "
    f"{region_accuracies[best_region] * 100:.2f}%"
)

print(
    f"DQN Improvement over Best Fixed Region: "
    f"{(0.7080 - region_accuracies[best_region]) * 100:.2f} "
    f"percentage points"
)
