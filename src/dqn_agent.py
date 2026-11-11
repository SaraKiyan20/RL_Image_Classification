
import torch
import torch.nn as nn


class DQN(nn.Module):

    def __init__(self, state_dim=512, action_dim=9):

        super(DQN, self).__init__()

        self.network = nn.Sequential(

            # Input layer: 512 features
            nn.Linear(state_dim, 256),

            nn.ReLU(),

            # Hidden layer
            nn.Linear(256, 128),

            nn.ReLU(),

            # Output layer: 9 actions
            nn.Linear(128, action_dim)
        )

    def forward(self, state):

        return self.network(state)


# -------------------------
# Test the DQN architecture
# -------------------------

if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    agent = DQN(
        state_dim=512,
        action_dim=9
    ).to(device)

    # Create a sample state
    sample_state = torch.randn(1, 512).to(device)

    # Get Q-values
    q_values = agent(sample_state)

    print("DQN architecture test")
    print("---------------------")

    print("Input shape:", sample_state.shape)
    print("Output shape:", q_values.shape)

    print("\nQ-values:")

    for action in range(9):

        print(
            f"Region {action + 1}: "
            f"{q_values[0, action].item():.4f}"
        )

    selected_action = q_values.argmax(dim=1).item()

    print("\nSelected region:", selected_action + 1)

    print("\nDQN architecture is working!")
