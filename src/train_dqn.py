
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from dqn_agent import DQN


# -----------------------------
# 1. Settings
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 128
learning_rate = 0.001
num_epochs = 50

print("Using device:", device)


# -----------------------------
# 2. Load prepared RL data
# -----------------------------

train_data = torch.load(
    "results/rl_train_data.pt",
    map_location="cpu",
    weights_only=False
)

val_data = torch.load(
    "results/rl_val_data.pt",
    map_location="cpu",
    weights_only=False
)

print("\nTraining data keys:", train_data.keys())
print("Validation data keys:", val_data.keys())


# -----------------------------
# 3. Extract states and rewards
# -----------------------------

train_states = train_data["states"].float()
train_rewards = train_data["rewards"].float()

val_states = val_data["states"].float()
val_rewards = val_data["rewards"].float()

print("\nTraining states:", train_states.shape)
print("Training rewards:", train_rewards.shape)
print("Validation states:", val_states.shape)
print("Validation rewards:", val_rewards.shape)


# -----------------------------
# 4. Create DataLoaders
# -----------------------------

train_dataset = TensorDataset(train_states, train_rewards)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)


# -----------------------------
# 5. Initialize the agent
# -----------------------------

agent = DQN(
    state_dim=512,
    action_dim=9
).to(device)

optimizer = torch.optim.Adam(
    agent.parameters(),
    lr=learning_rate
)

criterion = nn.SmoothL1Loss()


# -----------------------------
# 6. Training
# -----------------------------

best_val_accuracy = 0.0

for epoch in range(num_epochs):

    agent.train()
    total_loss = 0.0

    for states, rewards in train_loader:

        states = states.to(device)
        rewards = rewards.to(device)

        # Predict rewards for all nine regions
        predicted_rewards = agent(states)

        # Learn to predict the observed rewards
        loss = criterion(predicted_rewards, rewards)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    # -------------------------
    # Validation
    # -------------------------

    agent.eval()

    with torch.no_grad():

        val_states_device = val_states.to(device)
        val_rewards_device = val_rewards.to(device)

        predicted_q_values = agent(val_states_device)

        # Select the region with the highest predicted reward
        selected_actions = predicted_q_values.argmax(dim=1)

        # Get the actual reward of each selected region
        selected_rewards = val_rewards_device.gather(
            1,
            selected_actions.unsqueeze(1)
        ).squeeze(1)

        val_accuracy = (selected_rewards == 1).float().mean().item()

        # Expected accuracy of choosing a uniformly random region
        random_accuracy = (
            (val_rewards_device == 1).float().mean().item()
        )

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Loss: {average_loss:.4f} "
        f"Validation Accuracy: {val_accuracy * 100:.2f}% "
        f"Random Accuracy: {random_accuracy * 100:.2f}%"
    )

    # Save the best model
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            agent.state_dict(),
            "models/dqn_agent_best.pth"
        )


print("\nTraining completed!")
print("--------------------")
print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print("Best model saved to models/dqn_agent_best.pth")
