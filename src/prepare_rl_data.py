
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import STL10
from torchvision.models import resnet18
from torchvision import transforms


# -------------------------
# Configuration
# -------------------------

BATCH_SIZE = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

CHECKPOINT_PATH = "models/resnet18_random_region_best.pth"


# -------------------------
# Load STL-10
# -------------------------

print("\nLoading STL-10...")

base_dataset = STL10(
    root="data",
    split="train",
    download=False,
    transform=transforms.ToTensor()
)

labels = torch.tensor(base_dataset.labels)

# Recreate the same stratified 4000/1000 split
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

train_loader = DataLoader(
    Subset(base_dataset, train_indices),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

val_loader = DataLoader(
    Subset(base_dataset, val_indices),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Training images:", len(train_indices))
print("Validation images:", len(val_indices))


# -------------------------
# Load trained patch classifier
# -------------------------

print("\nLoading trained patch classifier...")

patch_model = resnet18(weights=None)
patch_model.fc = nn.Linear(patch_model.fc.in_features, 10)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=True
)

patch_model.load_state_dict(checkpoint)
patch_model = patch_model.to(device)
patch_model.eval()


# -------------------------
# Create feature extractor
# -------------------------

print("Preparing state feature extractor...")

state_model = resnet18(weights=None)

# First recreate the 10-class classification layer
state_model.fc = nn.Linear(state_model.fc.in_features, 10)

# Load the trained checkpoint
state_model.load_state_dict(checkpoint)

# Remove the classification layer to extract 512 features
state_model.fc = nn.Identity()

state_model = state_model.to(device)
state_model.eval()

state_model = state_model.to(device)
state_model.eval()


# -------------------------
# Image normalization
# -------------------------

mean = torch.tensor(
    [0.485, 0.456, 0.406],
    device=device
).view(1, 3, 1, 1)

std = torch.tensor(
    [0.229, 0.224, 0.225],
    device=device
).view(1, 3, 1, 1)


# -------------------------
# Prepare states and rewards
# -------------------------

def prepare_split(loader, split_name):

    all_states = []
    all_rewards = []
    all_labels = []

    print(f"\nPreparing {split_name} data...")

    with torch.inference_mode():

        for batch_number, (images, labels_batch) in enumerate(loader):

            images = images.to(device)
            labels_batch = labels_batch.to(device)

            # Create low-resolution overview:
            # 96x96 -> 32x32 -> 224x224
            overview = F.interpolate(
                images,
                size=(32, 32),
                mode="bilinear",
                align_corners=False,
                antialias=True
            )

            overview = F.interpolate(
                overview,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
                antialias=True
            )

            overview = (overview - mean) / std

            # Extract the 9 non-overlapping 32x32 regions
            patches = []

            for row in range(3):
                for col in range(3):

                    patch = images[
                        :,
                        :,
                        row * 32:(row + 1) * 32,
                        col * 32:(col + 1) * 32
                    ]

                    patches.append(patch)

            # Shape: [batch_size, 9, 3, 32, 32]
            patches = torch.stack(patches, dim=1)

            batch_size = images.size(0)

            # Combine image regions into one batch
            patches = patches.reshape(
                batch_size * 9, 3, 32, 32
            )

            # Resize each patch to 224x224
            patches = F.interpolate(
                patches,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
                antialias=True
            )

            patches = (patches - mean) / std

            # Extract state features
            states = state_model(overview)

            # Classify all nine patches
            outputs = patch_model(patches)

            predictions = outputs.argmax(dim=1)

            predictions = predictions.reshape(batch_size, 9)

            # Reward: +1 for correct, -1 for incorrect
            rewards = torch.where(
                predictions == labels_batch.unsqueeze(1),
                1.0,
                -1.0
            )

            all_states.append(states.cpu())
            all_rewards.append(rewards.cpu())
            all_labels.append(labels_batch.cpu())

            if (batch_number + 1) % 100 == 0:
                print(
                    f"Processed {(batch_number + 1) * BATCH_SIZE} "
                    f"images..."
                )

    return {
        "states": torch.cat(all_states),
        "rewards": torch.cat(all_rewards),
        "labels": torch.cat(all_labels)
    }


# -------------------------
# Process training and validation
# -------------------------

train_data = prepare_split(train_loader, "training")
val_data = prepare_split(val_loader, "validation")


# -------------------------
# Save prepared data
# -------------------------

print("\nSaving prepared RL data...")

torch.save(train_data, "results/rl_train_data.pt")
torch.save(val_data, "results/rl_val_data.pt")


# -------------------------
# Verify saved data
# -------------------------

print("\nData preparation completed!")
print("---------------------------")

print("Training states:", train_data["states"].shape)
print("Training rewards:", train_data["rewards"].shape)

print("Validation states:", val_data["states"].shape)
print("Validation rewards:", val_data["rewards"].shape)

print(
    "Training average reward:",
    train_data["rewards"].float().mean().item()
)

print(
    "Validation average reward:",
    val_data["rewards"].float().mean().item()
)

print("\nSaved:")
print("results/rl_train_data.pt")
print("results/rl_val_data.pt")
