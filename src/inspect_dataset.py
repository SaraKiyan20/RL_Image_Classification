
from torchvision.datasets import STL10

DATA_DIR = "data"

print("Loading STL-10 dataset...")

train_dataset = STL10(
    root=DATA_DIR,
    split="train",
    download=False
)

test_dataset = STL10(
    root=DATA_DIR,
    split="test",
    download=False
)

print("\nDataset loaded successfully!")
print("Training images:", len(train_dataset))
print("Test images:", len(test_dataset))
print("Number of classes:", len(train_dataset.classes))
print("Classes:", train_dataset.classes)

# Inspect the first image
image, label = train_dataset[0]

print("\nFirst image size:", image.size)
print("First image label:", label)
print("First image class:", train_dataset.classes[label])
