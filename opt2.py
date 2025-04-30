import numpy as np
import matplotlib.pyplot as plt

# Constants
V = 250  # Cruise speed (m/s)
SFC = 0.6  # Specific Fuel Consumption (1/s)
payload_mass = 1000  # Payload mass (kg)

# Define the expanded Breguet range equation
def breguet_range(params):
    LD, S, AR, fuel_mass = params

    # Enforce physical bounds
    if not (5 <= LD <= 20) or not (10 <= S <= 200) or not (5 <= AR <= 15) or not (500 <= fuel_mass <= 10000):
        return -1e6  # Penalize invalid values

    W_empty = 20 * (S * AR) ** 0.6
    W_i = W_empty + fuel_mass + payload_mass
    W_f = W_empty + payload_mass

    if W_f >= W_i:
        return -1e6  # Penalize invalid weights

    R = (V / SFC) * LD * np.log(W_i / W_f)
    return R

# Gradient descent with projection to constraints
def gradient_descent_projected(initial_params, lr=1e-2, iterations=100):
    x = np.array(initial_params, dtype=np.float64)
    history = []

    for _ in range(iterations):
        grad = np.zeros_like(x)
        fx = -breguet_range(x)
        eps = 1e-4
        for j in range(len(x)):
            x_eps = np.copy(x)
            x_eps[j] += eps
            grad[j] = (-breguet_range(x_eps) - fx) / eps

        x -= lr * grad

        # Projection to constraints
        x[0] = np.clip(x[0], 5, 20)      # LD
        x[1] = np.clip(x[1], 10, 200)    # S
        x[2] = np.clip(x[2], 5, 15)      # AR
        x[3] = np.clip(x[3], 500, 10000) # fuel mass

        history.append(breguet_range(x))

    return x, history

# Run projected gradient descent
initial_params = [10.0, 50.0, 10.0, 2000.0]
opt_params_proj, history_proj = gradient_descent_projected(initial_params, lr=0.5, iterations=100)

# Plot the optimization history
plt.figure(figsize=(10, 5))
plt.plot(history_proj, label="Gradient Descent")
plt.xlabel("Iteration")
plt.ylabel("Range (meters)")
plt.title("Gradient Descent on Breguet Range")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

opt_params_proj, history_proj[-1]
