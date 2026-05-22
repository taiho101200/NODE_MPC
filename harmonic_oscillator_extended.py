import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)


def true_controlled_oscillator_dynamics(x, u, omega=1.0, b= 1.0):
    """
        x: (batch_size, 2) where x[:,0] = x1 and x[:,1] = x2
        u: (batch_size, 1) control input
    Returns 
        dx/dt: (batch_size, 2)
    """
    x1 = x[:, 0:1]
    x2 = x[:, 1:2]

    dx1dt = x2
    dx2dt = -omega**2 * x1 + b * u # Control input affects the second derivative

    return torch.cat([dx1dt, dx2dt], dim=1)


def rk4_step_true(x, u, dt, omega=1.0, b=1.0):
    """
        Performs a single RK4 step for the controlled harmonic oscillator.
        x: (batch_size, 2) current state
        u: (batch_size, 1) control input
        dt: time step size
    Returns:
        x_next: (batch_size, 2) state at the next time step
    """
    k1 = true_controlled_oscillator_dynamics(x, u, omega, b)
    k2 = true_controlled_oscillator_dynamics(x + 0.5 * dt * k1, u, omega, b)
    k3 = true_controlled_oscillator_dynamics(x + 0.5 * dt * k2, u, omega, b)
    k4 = true_controlled_oscillator_dynamics(x + dt * k3, u, omega, b)

    x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return x_next



class ControlledHarmonicOscillatorDataset(Dataset):
    def __init__(
            self,
            n_traj=200,
            n_steps=200,
            T=10.0,
            omega=1.0,
            b=1.0,
            seed=42
    ):
        """
        Dataset for a controlled harmonic oscillator:
            dx1/dt = x2
            dx2/dt = -omega^2 * x1 + b * u
        """

        self.n_traj = n_traj
        self.n_steps = n_steps
        self.T = T
        self.omega = omega
        self.b = b

        generator = torch.Generator().manual_seed(seed)

        self.t = torch.linspace(0, T, n_steps) # Time vector

        self.x0 = torch.empty(n_traj, 2).uniform_(-2.0, 2.0, generator=generator) # Initial conditions 

        # Random control inputs for each trajectory and time step
        self.u = self._generate_smooth_control_inputs(generator)

        self.x = self._simulate_trajectories()

    def _generate_smooth_control_inputs(self, generator):
        """
            Generate smooth control inputs have random frequencies and amplitudes.

        return:
            u: (n_traj, n_steps - 1, 1) control inputs for each trajectory and time step
        """

        t_u = self.t[:-1].view(1, self.n_steps -1)

        # fixed frequencies for all trajectories
        freqs = torch.tensor([0.5, 1.0, 1.5]).view(1, 3, 1) 

        # Random amplitudes for each trajectory and frequency
        amps = torch.empty(self.n_traj,3,1).uniform_(-0.8, 0.8, generator=generator)

        # Random phases for each trajectory and frequency
        phases = torch.empty(self.n_traj, 3, 1).uniform_(0.0, 2.0 * torch.pi, generator=generator)

        # t_u: (1, 1, n_steps-1)
        t_u = t_u.unsqueeze(1)

        # U components: (n_traj, 3, n_steps-1)
        u_components = amps * torch.sin(freqs * t_u + phases)

        # Sum over the frequency components to get the final control input
        u = u_components.sum(dim=1) / 3
        
        u = torch.clamp(u, -1.0, 1.0)  

        u = u.unsqueeze(-1) # (n_traj, n_steps-1, 1)    

        return u
    
    def _simulate_trajectories(self):
        xs = [self.x0]
        x = self.x0

        for k in range(self.n_steps - 1):
            dt = self.t[k + 1] - self.t[k]
            u_k = self.u[:, k, :]

            x = rk4_step_true(
                x=x,
                u=u_k,
                dt=dt,
                omega=self.omega,
                b=self.b
            )

            xs.append(x)

        # shape [n_traj, n_steps, 2]
        return torch.stack(xs, dim=1)

    def __len__(self):
        return self.n_traj

    def __getitem__(self, idx):
        return {
            "x0": self.x0[idx],   # [2]
            "u": self.u[idx],     # [n_steps - 1, 1]
            "x": self.x[idx]      # [n_steps, 2]
        }
    

dataset = ControlledHarmonicOscillatorDataset(
    n_traj=200,
    n_steps=200,
    T=10.0,
    omega=1.0,
    b=1.0
)

# sample = dataset[0]

# print("t shape:", dataset.t.shape)
# print("x0 shape:", sample["x0"].shape)
# print("u shape:", sample["u"].shape)
# print("x shape:", sample["x"].shape)

# idx = 0

# sample = dataset[idx]
# t = dataset.t

# plt.figure(figsize=(9, 4))
# plt.plot(t[:-1].numpy(), sample["u"][:, 0].numpy())
# plt.xlabel("Time")
# plt.ylabel("u(t)")
# plt.title("Input trajectory u(t)")
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(9, 5))
# plt.plot(t.numpy(), sample["x"][:, 0].numpy(), label="x1: position")
# plt.plot(t.numpy(), sample["x"][:, 1].numpy(), label="x2: velocity")
# plt.xlabel("Time")
# plt.ylabel("State")
# plt.title("State trajectory under input u(t)")
# plt.legend()
# plt.grid(True)
# plt.show()


class ControlledNeuralODEFunc(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(3, 64),   
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)    # output = dx1dt, dx2dt
        )

    def forward(self, x, u):
        """
        x shape: [batch_size, 2]
        u shape: [batch_size, 1]

        return:
            dxdt shape: [batch_size, 2]
        """

        xu = torch.cat([x, u], dim=1)
        dxdt = self.net(xu)

        return dxdt
    

def rk4_solve_controlled_node(func, x0, u, t):
    """
    func: NODE model learn dx/dt = f_theta(x,u)
    x0: shape [batch_size, 2]
    u:  shape [batch_size, n_steps - 1, 1]
    t:  shape [n_steps]

    return:
        xs shape [batch_size, n_steps, 2]
    """

    xs = [x0]
    x = x0

    for k in range(len(t) - 1):
        dt = t[k + 1] - t[k]
        u_k = u[:, k, :]

        k1 = func(x, u_k)
        k2 = func(x + 0.5 * dt * k1, u_k)
        k3 = func(x + 0.5 * dt * k2, u_k)
        k4 = func(x + dt * k3, u_k)

        x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        xs.append(x)

    return torch.stack(xs, dim=1)


dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

t = dataset.t.to(device)

func = ControlledNeuralODEFunc().to(device)

optimizer = torch.optim.Adam(func.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

n_epochs = 1000

for epoch in range(n_epochs):
    total_loss = 0.0

    for batch in dataloader:
        x0_batch = batch["x0"].to(device)   # [batch_size, 2]
        u_batch = batch["u"].to(device)     # [batch_size, n_steps-1, 1]
        true_x = batch["x"].to(device)      # [batch_size, n_steps, 2]

        optimizer.zero_grad()

        pred_x = rk4_solve_controlled_node(
            func=func,
            x0=x0_batch,
            u=u_batch,
            t=t
        )

        loss = loss_fn(pred_x, true_x)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d} | Loss = {avg_loss:.8f}")


func.eval()

idx = 10

with torch.no_grad():
    sample = dataset[idx]

    x0_test = sample["x0"].unsqueeze(0).to(device)   # [1, 2]
    u_test = sample["u"].unsqueeze(0).to(device)     # [1, n_steps-1, 1]
    true_x = sample["x"].unsqueeze(0).to(device)     # [1, n_steps, 2]

    pred_x = rk4_solve_controlled_node(
        func=func,
        x0=x0_test,
        u=u_test,
        t=t
    )

true_x_cpu = true_x.cpu()
pred_x_cpu = pred_x.cpu()
u_cpu = u_test.cpu()
t_cpu = t.cpu()

plt.figure(figsize=(9, 5))
plt.plot(t_cpu.numpy(), true_x_cpu[0, :, 0].numpy(), label="True x1")
plt.plot(t_cpu.numpy(), pred_x_cpu[0, :, 0].numpy(), "--", label="NODE x1")
plt.plot(t_cpu.numpy(), true_x_cpu[0, :, 1].numpy(), label="True x2")
plt.plot(t_cpu.numpy(), pred_x_cpu[0, :, 1].numpy(), "--", label="NODE x2")
plt.xlabel("Time")
plt.ylabel("State")
plt.title("Controlled Harmonic Oscillator: True vs NODE Prediction")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(9, 3))
plt.plot(t_cpu[:-1].numpy(), u_cpu[0, :, 0].numpy(), label="Input u")
plt.xlabel("Time")
plt.ylabel("u(t)")
plt.title("Input used for prediction")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(6, 6))
plt.plot(true_x_cpu[0, :, 0].numpy(), true_x_cpu[0, :, 1].numpy(), label="True")
plt.plot(pred_x_cpu[0, :, 0].numpy(), pred_x_cpu[0, :, 1].numpy(), "--", label="NODE")
plt.xlabel("x1: position")
plt.ylabel("x2: velocity")
plt.title("Phase Portrait with Control Input")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()