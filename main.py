
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import syft as sy  # PySyft for federated learning and differential privacy

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize PySyft for federated learning
def initialize_workers():
    hook = sy.TorchHook(torch)
    alice = sy.VirtualWorker(hook, id="alice")
    bob = sy.VirtualWorker(hook, id="bob")
    return alice, bob

# Split data between workers
def split_data(data, targets, alice, bob):
    data_alice, data_bob = data.chunk(2)
    targets_alice, targets_bob = targets.chunk(2)

    dataset_alice = TensorDataset(data_alice, targets_alice)
    dataset_bob = TensorDataset(data_bob, targets_bob)

    federated_data = sy.FederatedDataLoader(
        {
            alice: DataLoader(dataset_alice, batch_size=32, shuffle=True),
            bob: DataLoader(dataset_bob, batch_size=32, shuffle=True),
        }
    )
    return federated_data

# Add differential privacy to the optimizer
def add_differential_privacy(optimizer, model, noise_multiplier=1.0, max_grad_norm=1.0):
    from opacus import PrivacyEngine

    privacy_engine = PrivacyEngine(
        model,
        batch_size=32,
        sample_size=60000,  # Adjust based on your dataset
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    privacy_engine.attach(optimizer)
    return privacy_engine

# Main function
def main():
    # Initialize workers
    alice, bob = initialize_workers()

    # Create synthetic data
    data = torch.randn(60000, 28, 28)
    targets = torch.randint(0, 10, (60000,))

    # Split data between workers
    federated_data = split_data(data, targets, alice, bob)

    # Initialize model and optimizer
    model = SimpleNN()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Add differential privacy
    privacy_engine = add_differential_privacy(optimizer, model)

    # Training loop
    model.train()
    for epoch in range(5):  # Number of epochs
        for batch_idx, (data, target) in enumerate(federated_data):
            optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

    # Detach privacy engine after training
    privacy_engine.detach()

if __name__ == "__main__":
    main()