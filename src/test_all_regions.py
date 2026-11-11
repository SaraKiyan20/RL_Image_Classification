

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import STL10
from torchvision.models import resnet18


# -------------------------
# Configuration
# -------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load one image
# -------------------------

dataset = STL10(
    root="data",
    split="train",
    download=False
)

image, true_label = dataset[0]

classes = dataset.classes

print("True class:", classes[true_label])


# -------------------------
# Image transformation
# -------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -------------------------
# Load trained patch classifier
# -------------------------

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
# Extract all nine regions
# -------------------------

regions = []

for row in range(3):
    for col in range(3):

        left = col * 32
        top = row * 32

        right = left + 32
        bottom = top + 32

        region = image.crop((left, top, right, bottom))

        region_tensor = transform(region)

        regions.append(region_tensor)


# Combine regions into one batch
regions = torch.stack(regions).to(device)


# -------------------------
# Classify all regions
# -------------------------

with torch.no_grad():

    outputs = model(regions)

    probabilities = torch.softmax(outputs, dim=1)

    predictions = outputs.argmax(dim=1)


# -------------------------
# Display results
# -------------------------

print("\nRegion classification results")
print("-----------------------------")

correct_regions = 0

for i in range(9):

    predicted_class = classes[predictions[i].item()]

    confidence = probabilities[i, predictions[i]].item() * 100

    is_correct = predictions[i].item() == true_label

    if is_correct:
        correct_regions += 1

    print(
        f"Region {i + 1}: "
        f"Predicted = {predicted_class}, "
        f"Confidence = {confidence:.2f}%, "
        f"Correct = {is_correct}"
    )

print("\nCorrectly classified regions:", correct_regions, "/ 9")
