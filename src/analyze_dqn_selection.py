
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F

from torchvision.datasets import STL10
from torch.utils.data import DataLoader

from dqn_agent import DQN


# -----------------------------
# 1. Settings
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# -----------------------------
# 2. Load test dataset
# -----------------------------

test_dataset = STL10(
    root="data",
    split="test",
    download=False,
    transform=transforms.ToTensor()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False
)


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


# -----------------------------
# 5. Load DQN agent
# -----------------------------

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
# 6. Normalization
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
# 7. Analysis
# -----------------------------

all_labels = []

all_region_correct = []
all_dqn_actions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        batch_size = images.size(0)

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

        # Extract state features
        states = state_model(overview)

        # DQN selects region
        q_values = agent(states)

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
            batch_size * 9,
            3,
            224,
            224
        )

        patches = normalize(patches)

        # Classify all regions
        outputs = patch_model(patches)

        predictions = outputs.argmax(dim=1)

        predictions = predictions.view(
            batch_size,
            9
        )

        # Determine which regions are correct
        correct = predictions.eq(
            labels.unsqueeze(1)
        )

        all_labels.extend(labels.cpu().tolist())

        all_region_correct.extend(
            correct.cpu().tolist()
        )

        all_dqn_actions.extend(
            actions.cpu().tolist()
        )


# -----------------------------
# 8. Calculate oracle accuracy
# -----------------------------

oracle_correct = 0

for correct_regions in all_region_correct:

    # Oracle succeeds if at least one region is correct
    if any(correct_regions):
        oracle_correct += 1

oracle_accuracy = oracle_correct / len(all_labels)


# -----------------------------
# 9. DQN selection analysis
# -----------------------------

dqn_correct = 0
available_opportunities = 0
dqn_success_when_available = 0

for correct_regions, action in zip(
    all_region_correct,
    all_dqn_actions
):

    selected_correctly = correct_regions[action]

    if selected_correctly:
        dqn_correct += 1

    # At least one correct region exists
    if any(correct_regions):

        available_opportunities += 1

        if selected_correctly:
            dqn_success_when_available += 1


dqn_accuracy = dqn_correct / len(all_labels)

selection_success = (
    dqn_success_when_available /
    available_opportunities
)


# -----------------------------
# 10. Region correctness
# -----------------------------

region_correctness = []

for region in range(9):

    correct_count = sum(
        sample[region]
        for sample in all_region_correct
    )

    accuracy = correct_count / len(all_labels)

    region_correctness.append(accuracy)


# -----------------------------
# 11. Print results
# -----------------------------

print("\nOracle Upper Bound")
print("----------------------------")

print(f"Oracle Accuracy: {oracle_accuracy * 100:.2f}%")

print("\nDQN Performance")
print("----------------------------")

print(f"DQN Accuracy: {dqn_accuracy * 100:.2f}%")

print(
    f"DQN Success When a Correct Region Exists: "
    f"{selection_success * 100:.2f}%"
)

print(
    f"Images with at least one correctly classified region: "
    f"{available_opportunities}"
)

print("\nOracle Opportunity Gap")
print("----------------------------")

print(
    f"Oracle - DQN: "
    f"{(oracle_accuracy - dqn_accuracy) * 100:.2f} "
    f"percentage points"
)

print("\nRegion Correctness")
print("----------------------------")

for region, accuracy in enumerate(region_correctness):

    print(
        f"Region {region + 1}: "
        f"{accuracy * 100:.2f}%"
    )

print("\nAnalysis completed!")
