import random
import heapq
import numpy as np
import matplotlib.pyplot as plt

# Constants
payload_mass = 500
V = 250
c = 0.6

# Bounds and steps
bounds = {
    "AR": (6, 12),
    "S": (15, 30),
    "LD": (10, 20),
    "fuel_mass": (500, 2000)
}
step_sizes = {
    "AR": 1,
    "S": 1,
    "LD": 1,
    "fuel_mass": 1
}

# Objective function
def compute_range(AR, S, LD, fuel_mass):
    empty_mass = (20 * (S * AR)**0.6)
    return (V / c) * LD * np.log((empty_mass + fuel_mass + payload_mass) / (empty_mass + payload_mass))

def heuristic(_state):
    return 0

def astar(max_iters=5000):
    curr_iters = 0

    start = (
        random.uniform(bounds["AR"][0], bounds["AR"][1]),
        random.uniform(bounds["S"][0], bounds["S"][1]),
        random.uniform(bounds["LD"][0], bounds["LD"][1]),
        random.uniform(bounds["fuel_mass"][0], bounds["fuel_mass"][1])
    )

    visited = set()
    heap = []

    g_start = -compute_range(*start)
    h_start = heuristic(start)
    f_start = g_start + h_start
    heapq.heappush(heap, (f_start, g_start, start, "Start"))  # also track the last changed parameter

    best = start
    best_score = compute_range(*start)

    best_ranges = [best_score]
    best_params_over_time = [best]
    param_changes = ["Start"]  # Track changes (Start = initial random sample)

    while heap and curr_iters < max_iters:
        f_curr, g_curr, current, changed_param = heapq.heappop(heap)

        if current in visited:
            continue
        visited.add(current)

        current_range = -f_curr

        if current_range > best_score:
            best_score = current_range
            best = current

        # Record best so far
        best_ranges.append(best_score)
        best_params_over_time.append(best)
        param_changes.append(changed_param)

        # Expand neighbors
        for i, key in enumerate(["AR", "S", "LD", "fuel_mass"]):
            for delta in [-step_sizes[key], step_sizes[key]]:
                new = list(current)
                new[i] += delta

                if all(x >= 0 for x in new):
                    new_state = tuple(round(x, 2) for x in new)

                    if new_state not in visited:
                        g_new = -compute_range(*new_state)
                        h_new = heuristic(new_state)
                        f_new = g_new + h_new
                        heapq.heappush(heap, (f_new, g_new, new_state, key))  # record what changed

        curr_iters += 1

    return best, best_score, best_ranges, best_params_over_time, param_changes

# Run
best_params, max_range, best_ranges, best_params_over_time, param_changes = astar()

print("Best Parameters:", best_params)
print("Max Range:", max_range)

# Plotting
iterations = list(range(len(best_ranges)))
param_names = ["AR", "S", "LD", "fuel_mass"]
best_params_over_time = np.array(best_params_over_time)

# Max range plot
plt.figure(figsize=(12, 8))
plt.plot(iterations, best_ranges)
plt.xlabel("Iteration")
plt.ylabel("Max Range")
plt.title("Max Range Over Iterations")
plt.grid(True)
plt.show()

# Parameters over time
plt.figure(figsize=(12, 8))
for i, name in enumerate(param_names):
    plt.plot(iterations, best_params_over_time[:, i], label=name)
plt.xlabel("Iteration")
plt.ylabel("Parameter Value")
plt.yscale('log')
plt.title("Best Parameter Values Over Iterations")
plt.legend()
plt.grid(True)
plt.show()

# Plot what parameter was changed at each iteration
plt.figure(figsize=(12, 6))

param_to_num = {"Start": -1, "AR": 0, "S": 1, "LD": 2, "fuel_mass": 3}
nums = [param_to_num[p] for p in param_changes]

plt.scatter(iterations, nums, c=nums, cmap="Set1", marker='s')
plt.yticks([-1, 0, 1, 2, 3], ["Start", "AR", "S", "LD", "fuel_mass"])
plt.xlabel("Iteration")
plt.ylabel("Changed Parameter")
plt.title("Parameter Changed at Each Iteration")
plt.grid(True)
plt.show()
