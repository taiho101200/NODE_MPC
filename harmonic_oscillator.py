import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)


class HarmonicOscillatorDataset(Dataset):
    def __init__(self, n_traj=100, n_steps=200, T=10.0, omega=1.0, seed=42):
        """
        Plant:
            dx1/dt = x2
            dx2/dt = -omega^2 * x1
        Samples trajectories:
            x1(t) = x1(0) * cos(omega * t) + (x2(0)/omega) * sin(omega * t)
            x2(t) = -x1(0) * omega * sin(omega * t) + x2(0) * cos(omega * t)

            x0 = [x1(0), x2(0)]
            x  = [[x1(t0),x2(t0)], 
                  ..., 
                 [x1(tT), x2(tT)]]
        """

        self.n_traj = n_traj
        self.n_steps = n_steps
        self.T = T
        self.omega = omega

        generator = torch.Generator().manual_seed(seed)

        self.t = torch.linspace(0, T, n_steps) # Time vector

        self.x0 = torch.empty(n_traj, 2).uniform_(-2.0, 2.0, generator=generator) # Initial conditions 

        x1_0 = self.x0[:,0].view(n_traj, 1) # (n_traj, 1)
        x2_0 = self.x0[:,1].view(n_traj, 1) # (n_traj, 1)

        t = self.t.view(1, n_steps) # (1, n_steps)

        # Real trajectories based on the harmonic oscillator solution
        x1 = x1_0 * torch.cos(omega * t) + (x2_0 / omega) * torch.sin(omega * t) # (n_traj, n_steps)
        x2 = -x1_0 * omega * torch.sin(omega * t) + x2_0 * torch.cos(omega * t) # (n_traj, n_steps)

        
        self.x = torch.stack([x1, x2], dim=-1) # (n_traj, n_steps, 2)

    def __len__(self):
        return self.n_traj
    
    def __getitem__(self, idx):
        return {
            "x0": self.x0[idx], # (2,)
            "x": self.x[idx]  # (n_steps, 2)
        }
    


dataset = HarmonicOscillatorDataset(
    n_traj=100, 
    n_steps=200, 
    T=10.0, 
    omega=1.0, 
    seed=42
)

# sample = dataset[0]

# print("Sample initial condition:", sample["x0"])
# print("Sample trajectory shape:", sample["x"].shape)
# print("t shape", dataset.t.shape)


dataloader = DataLoader(
    dataset, 
    batch_size=16, 
    shuffle=True
)

# batch = next(iter(dataloader))

# print("Batch initial conditions shape:", batch["x0"].shape) # (batch_size, 2)
# print("Batch trajectories shape:", batch["x"].shape) # (batch_size, n_steps, 2)

class NeuralODEFunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        dxdt = self.net(x)
        return dxdt
    


def rk4_solve(func, x0, t):
    """
    func: neural network learning dx/dt
    x0: initial conditions (n_traj, 2)
    t: time vector [n_steps]

    return:
        xs: (batch_size, n_steps, 2)
    """
    xs = [x0]
    x = x0
    for k in range(len(t) - 1):
        dt = t[k+1] - t[k]
        k1 = func(x)
        k2 = func(x + 0.5 * dt * k1)
        k3 = func(x + 0.5 * dt * k2)
        k4 = func(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        xs.append(x)
    
    return torch.stack(xs, dim=1) # (n_traj, n_steps, 2)


t = dataset.t.to(device) # (n_steps,)
func = NeuralODEFunc().to(device)
optimizer = torch.optim.Adam(func.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
n_epochs = 2000

for epoch in range(n_epochs):
    total_loss = 0.0
    for batch in dataloader:
        x0 = batch["x0"].to(device) # (batch_size, 2)
        true_x = batch["x"].to(device) # (batch_size, n_steps, 2)

        optimizer.zero_grad()
        pred_x = rk4_solve(func, x0, t) # (batch_size, n_steps, 2)
        loss = loss_fn(pred_x, true_x)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")



func.eval()

idx = 10

with torch.no_grad():
    sample = dataset[idx]

    x0_test = sample["x0"].unsqueeze(0).to(device)   # [1, 2]
    true_x = sample["x"].unsqueeze(0).to(device)     # [1, n_steps, 2]

    pred_x = rk4_solve(func, x0_test, t)

true_x_cpu = true_x.cpu()
pred_x_cpu = pred_x.cpu()
t_cpu = t.cpu()

plt.figure(figsize=(9, 5))
plt.plot(t_cpu.numpy(), true_x_cpu[0, :, 0].numpy(), label="True x1")
plt.plot(t_cpu.numpy(), pred_x_cpu[0, :, 0].numpy(), "--", label="NODE x1")
plt.plot(t_cpu.numpy(), true_x_cpu[0, :, 1].numpy(), label="True x2")
plt.plot(t_cpu.numpy(), pred_x_cpu[0, :, 1].numpy(), "--", label="NODE x2")
plt.xlabel("Time")
plt.ylabel("State value")
plt.title("Harmonic Oscillator: True vs NODE Prediction")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(6, 6))
plt.plot(true_x_cpu[0, :, 0].numpy(), true_x_cpu[0, :, 1].numpy(), label="True trajectory")
plt.plot(pred_x_cpu[0, :, 0].numpy(), pred_x_cpu[0, :, 1].numpy(), "--", label="NODE prediction")
plt.xlabel("x1: position")
plt.ylabel("x2: velocity")
plt.title("Phase Portrait: Harmonic Oscillator")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()