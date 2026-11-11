
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import STL10
from torchvision.models import resnet18

# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load one image from STL-10
dataset = STL10(
    root="data",
    split="train",
    download=False
)

image, label = dataset[0]

# Create a low-resolution overview (32 x 32),
# then resize it for ResNet18
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

image_tensor = transform(image).unsqueeze(0).to(device)

# Load the trained random-region model
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)

model.load_state_dict(
    torch.load(
        "models/resnet18_random_region_best.pth",
        map_location=device,
        weights_only=True
    )
)

# Remove the classification layer.
# ResNet18 will now return its 512-dimensional features.
model.fc = nn.Identity()

model = model.to(device)
model.eval()

# Extract features
with torch.no_grad():
    features = model(image_tensor)

print("\nState representation test")
print("-------------------------")
print("Original image size:", image.size)
print("Input tensor shape:", image_tensor.shape)
print("Feature vector shape:", features.shape)
print("Number of features:", features.shape[1])
print("Class:", dataset.classes[label])

print("\nState representation is ready!")
