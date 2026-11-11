
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from torchvision.datasets import STL10
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import random


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

patch_model = models.resnet18(weights=None)
patch_model.fc = nn.Linear(
    patch_model.fc.in_features,
    10
)

patch_model.load_state_dict(
    torch.load(
        "models/resnet18_random_region_best.pth",
        map_location=device,
        weights_only=True
    )
)

patch_model = patch_model.to(device)
patch_model.eval()

# We do not need gradients
for param in patch_model.parameters():
    param.requires_grad = False


# -----------------------------
# 4. Load state feature extractor
# -----------------------------

state_model = models.resnet18(weights=None)
state_model.fc = nn.Linear(
    state_model.fc.in_features,
    10
)

state_model.load_state_dict(
    torch.load(
        "models/resnet18_random_region_best.pth",
        map_location=device,
        weights_only=True
    )
)

state_model.fc = nn.Identity()
state_model = state_model.to(device)
state_model.eval()

for param in state_model.parameters():
    param.requires_grad = False


# -----------------------------
# 5. Load trained DQN
# -----------------------------

from dqn_agent import DQN

agent = DQN(
    state_dim=512,
    action_dim=9
)

agent.load_state_dict(
    torch.load(
        "models/dqn_agent_best.pth",
        map_location=device,
        weights_only=True
    )
)

agent = agent.to(device)
agent.eval()


# -----------------------------
# 6. Image preprocessing
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
# 7. Evaluation
# -----------------------------

all_labels = []
dqn_predictions = []
random_predictions = []

selected_regions = []

original_index = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        batch_size = images.size(0)

        # Create low-resolution overview
        overview = torch.nn.functional.interpolate(
            images,
            size=(32, 32),
            mode="bilinear",
            align_corners=False
        )

        overview = torch.nn.functional.interpolate(
            overview,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        overview = normalize(overview)

        # Extract state features
        states = state_model(overview)

        # Agent predicts Q-values
        q_values = agent(states)

        # Select region with highest Q-value
        actions = q_values.argmax(dim=1)

        # Extract all nine regions
        patches = []

        for row in range(3):
            for col in range(3):

                patch = images[
                    :,
                    :,
                    row * 32:(row + 1) * 32,
                    col * 32:(col + 1) * 32
                ]

                patch = torch.nn.functional.interpolate(
                    patch,
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False
                )

                patches.append(patch)

        # Shape: [batch_size, 9, 3, 224, 224]
        patches = torch.stack(patches, dim=1)

        # Classify all patches
        patches = patches.view(
            batch_size * 9,
            3,
            224,
            224
        )

        patches = normalize(patches)

        patch_outputs = patch_model(patches)

        patch_predictions = patch_outputs.argmax(dim=1)

        patch_predictions = patch_predictions.view(
            batch_size,
            9
        )

        # Get prediction from selected regions
        selected_predictions = patch_predictions.gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)

        # Random-region baseline
        random_actions = torch.tensor([
            random.Random(i + 42).randrange(9)
            for i in range(
                original_index,
                original_index + batch_size
            )
        ], device=device)

        random_selected_predictions = patch_predictions.gather(
            1,
            random_actions.unsqueeze(1)
        ).squeeze(1)

        # Store results
        all_labels.extend(labels.cpu().tolist())

        dqn_predictions.extend(
            selected_predictions.cpu().tolist()
        )

        random_predictions.extend(
            random_selected_predictions.cpu().tolist()
        )

        selected_regions.extend(
            (actions + 1).cpu().tolist()
        )

        original_index += batch_size


# -----------------------------
# 8. Calculate metrics
# -----------------------------

dqn_accuracy = accuracy_score(
    all_labels,
    dqn_predictions
)

random_accuracy = accuracy_score(
    all_labels,
    random_predictions
)

dqn_macro_f1 = f1_score(
    all_labels,
    dqn_predictions,
    average="macro"
)

random_macro_f1 = f1_score(
    all_labels,
    random_predictions,
    average="macro"
)


# -----------------------------
# 9. Print results
# -----------------------------

print("\nDQN Test Results")
print("----------------------")

print("Test images:", len(all_labels))

print(f"DQN Accuracy: {dqn_accuracy * 100:.2f}%")
print(f"DQN Macro-F1: {dqn_macro_f1:.4f}")

print("\nRandom Region Baseline")
print("----------------------")

print(f"Random Accuracy: {random_accuracy * 100:.2f}%")
print(f"Random Macro-F1: {random_macro_f1:.4f}")

print("\nImprovement")
print("----------------------")

print(
    f"Accuracy improvement: "
    f"{(dqn_accuracy - random_accuracy) * 100:.2f} percentage points"
)

print("\nDQN Per-Class F1")
print("----------------------")

class_f1 = f1_score(
    all_labels,
    dqn_predictions,
    average=None
)

for name, score in zip(class_names, class_f1):
    print(f"{name}: {score:.4f}")


# -----------------------------
# 10. Confusion matrix
# -----------------------------

print("\nDQN Confusion Matrix")
print("----------------------")

print(
    confusion_matrix(
        all_labels,
        dqn_predictions
    )
)


# -----------------------------
# 11. Region selection statistics
# -----------------------------

print("\nSelected Region Distribution")
print("----------------------")

for region in range(1, 10):

    count = selected_regions.count(region)

    percentage = count / len(selected_regions) * 100

    print(
        f"Region {region}: "
        f"{count} images ({percentage:.2f}%)"
    )
