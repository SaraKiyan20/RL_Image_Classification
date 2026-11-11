
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

from torchvision.datasets import STL10
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split


# -----------------------------
# 1. Settings
# -----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

batch_size = 4

print("Using device:", device)


# -----------------------------
# 2. Load STL-10
# -----------------------------

transform = transforms.ToTensor()

dataset = STL10(
    root="data",
    split="train",
    download=False,
    transform=transform
)

labels = dataset.labels

train_indices, val_indices = train_test_split(
    list(range(len(dataset))),
    test_size=0.2,
    stratify=labels,
    random_state=42
)

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# -----------------------------
# 3. Load patch classifier
# -----------------------------

print("\nLoading patch classifier...")

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

for param in patch_model.parameters():
    param.requires_grad = False


# -----------------------------
# 4. Load full-image state model
# -----------------------------

print("Loading full-image state extractor...")

state_model = models.resnet18(weights=None)

state_model.fc = nn.Linear(
    state_model.fc.in_features,
    10
)

state_model.load_state_dict(
    torch.load(
        "models/resnet18_stl10_best.pth",
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
# 5. Normalization
# -----------------------------

mean = torch.tensor(
    [0.485, 0.456, 0.406]
).view(1, 3, 1, 1).to(device)

std = torch.tensor(
    [0.229, 0.224, 0.225]
).view(1, 3, 1, 1).to(device)


def normalize(images):
    return (images - mean) / std


# -----------------------------
# 6. Prepare RL data
# -----------------------------

def prepare_data(loader, split_name):

    all_states = []
    all_rewards = []

    print(f"\nPreparing {split_name} data...")

    with torch.no_grad():

        for batch_index, (images, labels) in enumerate(loader):

            images = images.to(device)
            labels = labels.to(device)

            current_batch_size = images.size(0)

            # Create low-resolution overview
            overview = F.interpolate(
                images,
                size=(32, 32),
                mode="bilinear",
                align_corners=False
            )

            overview = F.interpolate(
                overview,
                size=(224, 224),
                mode="bilinear",
                align_corners=False
            )

            overview = normalize(overview)

            # Extract 512-dimensional state
            states = state_model(overview)

            # Extract all nine image regions
            patches = []

            for row in range(3):
                for col in range(3):

                    patch = images[
                        :,
                        :,
                        row * 32:(row + 1) * 32,
                        col * 32:(col + 1) * 32
                    ]

                    patch = F.interpolate(
                        patch,
                        size=(224, 224),
                        mode="bilinear",
                        align_corners=False
                    )

                    patches.append(patch)

            patches = torch.stack(
                patches,
                dim=1
            )

            patches = patches.view(
                current_batch_size * 9,
                3,
                224,
                224
            )

            patches = normalize(patches)

            # Classify all nine regions
            outputs = patch_model(patches)

            predictions = outputs.argmax(dim=1)

            predictions = predictions.view(
                current_batch_size,
                9
            )

            # Reward: +1 correct, -1 incorrect
            rewards = torch.where(
                predictions == labels.unsqueeze(1),
                1.0,
                -1.0
            )

            all_states.append(states.cpu())
            all_rewards.append(rewards.cpu())

            if (batch_index + 1) % 100 == 0:
                print(
                    f"Processed batches: "
                    f"{batch_index + 1}/{len(loader)}"
                )

    return (
        torch.cat(all_states),
        torch.cat(all_rewards)
    )


# -----------------------------
# 7. Generate datasets
# -----------------------------

train_states, train_rewards = prepare_data(
    train_loader,
    "training"
)

val_states, val_rewards = prepare_data(
    val_loader,
    "validation"
)


# -----------------------------
# 8. Save datasets
# -----------------------------

torch.save(
    {
        "states": train_states,
        "rewards": train_rewards
    },
    "results/rl_train_data_full_image.pt"
)

torch.save(
    {
        "states": val_states,
        "rewards": val_rewards
    },
    "results/rl_val_data_full_image.pt"
)


# -----------------------------
# 9. Display results
# -----------------------------

print("\nData Preparation Completed!")
print("----------------------------")

print("Training states:", train_states.shape)
print("Training rewards:", train_rewards.shape)

print("Validation states:", val_states.shape)
print("Validation rewards:", val_rewards.shape)

print(
    "Training average reward:",
    train_rewards.mean().item()
)

print(
    "Validation average reward:",
    val_rewards.mean().item()
)

print("\nNew RL datasets saved successfully!")
 