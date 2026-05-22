import torch
import torch.nn as nn
import matplotlib.pyplot as plt


print("PyTorch version:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

#============ 
# Generate synthetic data
#============

torch.manual_seed(42)  # For reproducibility

a_true = 0.8 # True amplitude dx/dt= -0.8x
T = 5.0 # Total time simulated
n_steps = 100 # Number of time steps
n_traj = 50 # Number of trajectories

t = torch.linspace(0, T, n_steps).to(device) # Time vector
dt = t[1] - t[0] # Time step size

# Generate trajectories 
x0 = torch.linspace(0.5, 5.0, n_traj).view(n_traj, 1).to(device) # Initial conditions

print("Initial conditions shape:", x0.shape)

# x(t) = x0 * exp(-a_true * t)
true_x = x0.unsqueeze(1) * torch.exp(-a_true * t).view(1, n_steps, 1)

print("True trajectories shape:", true_x.shape)


class NeuralODEFunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        dxdt = self.net(x)
        return dxdt
    

#============ ODE Solver (Euler method for simplicity) ============
def euler_solve(func, x0, t):
    """
    func: neural network learning dx/dt
    x0: initial conditions (n_traj, 1)
    t: time vector [n_steps]
    """
    xs = [x0]
    x = x0
    for k in range(len(t) - 1):
        dt = t[k+1] - t[k]
        dxdt = func(x)
        x = x + dxdt * dt
        xs.append(x)
    
    return torch.stack(xs, dim=1) # (n_traj, n_steps, 1)


##============ Training loop ============
func = NeuralODEFunc().to(device)
ooptimizer = torch.optim.Adam(func.parameters(), lr=0.01)
n_epochs = 2000

for epoch in range(n_epochs):
    ooptimizer.zero_grad()
    pred_x = euler_solve(func, x0, t) # (n_traj, n_steps, 1)
    loss = torch.mean((pred_x - true_x)**2)
    loss.backward()
    ooptimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")


#============ Plotting results ============
with torch.no_grad():
    pred_x = euler_solve(func, x0, t).cpu().numpy() # (n_traj, n_steps, 1)
    true_x = true_x.cpu().numpy()

idx =45

# plt.figure(figsize=(10, 6))
# plt.plot(t.cpu(), true_x[idx, :, 0], label="True trajectory", linewidth=2)
# plt.plot(t.cpu(), pred_x[idx, :, 0], label="Predicted NODE trajectory", linestyle='dashed')
# plt.title("Exponential Decay: True vs Predicted Trajectory")
# plt.xlabel("Time")
# plt.ylabel("x(t)")
# plt.legend()
# plt.grid()
# plt.show()

test_x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]]).to(device)

with torch.no_grad():
    learned_dxdt = func(test_x)
    true_dxdt = -a_true * test_x

print("x        NODE dx/dt        True dx/dt")
for i in range(len(test_x)):
    print(
        f"{test_x[i].item():.1f}      "
        f"{learned_dxdt[i].item(): .4f}          "
        f"{true_dxdt[i].item(): .4f}"
    )


