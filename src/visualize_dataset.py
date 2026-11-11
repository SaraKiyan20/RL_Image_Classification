
import matplotlib.pyplot as plt
import random
from torchvision.datasets import STL10

# Load the training dataset
train_dataset = STL10(
    root="data",
    split="train",
    download=False
)

# Class names
classes = train_dataset.classes

# Select 10 random images
indices = random.sample(range(len(train_dataset)), 10)

# Create a figure
fig, axes = plt.subplots(2, 5, figsize=(14, 7))

for ax, idx in zip(axes.flat, indices):

    image, label = train_dataset[idx]

    ax.imshow(image)
    ax.set_title(classes[label])
    ax.axis("off")

plt.tight_layout()
plt.show()
