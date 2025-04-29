import random
import heapq
import numpy as np
import matplotlib.pyplot as plt

# --- Constants --- #
payload_mass = 500  # Payload mass
V = 250             # Cruise velocity
c = 0.6             # Specific fuel consumption

# --- Parameter Initialization Bounds --- #
init_bounds = {
    "AR": (6, 12),              # Aspect Ratio
    "S": (15, 30),              # Wing Area
    "LD": (10, 20),             # Lift-to-Drag Ratio
    "fuel_mass": (500, 2000)    # Fuel Mass
}

# --- Parameter Global Bounds --- #
param_bounds = {
    "AR": (0, np.inf),
    "S": (0, np.inf),
    "LD": (0, np.inf),
    "fuel_mass": (0, np.inf)
}

# --- Step sizes for local search around a parameter --- #
step_sizes = {
    "AR": 1,
    "S": 1,
    "LD": 1,
    "fuel_mass": 1
}

def generate_initial_state():
    """
    Generates a random starting point within init_bounds.
    
    Returns:
        tuple: Random values for AR, S, LD, and fuel_mass within init_bounds.
    """
    return (
        random.uniform(*init_bounds["AR"]),
        random.uniform(*init_bounds["S"]),
        random.uniform(*init_bounds["LD"]),
        random.uniform(*init_bounds["fuel_mass"])
    )

def compute_range(AR, S, LD, fuel_mass):
    """
    Computes the aircraft's range based on design parameters.
    
    Args:
        AR (float): Aspect Ratio.
        S (float): Wing Area.
        LD (float): Lift-to-Drag Ratio.
        fuel_mass (float): Fuel Mass.
    
    Returns:
        float: Computed range using a modified Breguet range equation.
    """
    empty_mass = 20 * (S * AR)**0.6
    return (V / c) * LD * np.log((empty_mass + fuel_mass + payload_mass) / (empty_mass + payload_mass))

def heuristic(state):
    """
    Heuristic function for A*. Currently returns 0, making A* equivalent to Dijkstra's algorithm.
    
    Args:
        state (tuple): A tuple of parameter values.
    
    Returns:
        int: Heuristic cost (currently always 0).
    """
    return 0

def expand_neighbors(curr_params, visited, param_bounds):
    """
    Generates neighboring states by iterating each parameter by it's corresponding step size.
    
    Args:
        curr_params (tuple): Current state (AR, S, LD, fuel_mass).
        visited (set): Set of already visited states.
        param_bounds (dict): Dictionary specifying min and max bounds for each parameter.
    
    Returns:
        list: List of valid neighboring states with their corresponding f(n), state, and modified parameter.
    """
    neighbors = []
    for i, key in enumerate(["AR", "S", "LD", "fuel_mass"]):
        for delta in [-step_sizes[key], step_sizes[key]]:
            new = list(curr_params)
            new[i] += delta

            # Check if the updated parameter is within the valid range
            if param_bounds[key][0] <= new[i] <= param_bounds[key][1]:
                new_state = tuple(round(x, 2) for x in new)
                if new_state not in visited:
                    g_new = -compute_range(*new_state)
                    h_new = heuristic(new_state)
                    f_new = g_new + h_new
                    neighbors.append((f_new, new_state, key))
    return neighbors

def astar(max_iters=3000):
    """
    Performs A* search to optimize aircraft parameters for maximum range.
    
    Args:
        max_iters (int): Maximum number of iterations to run.
    
    Returns:
        tuple: Best parameters, best range, list of best ranges over time,
               list of parameter sets over time, and which parameter was changed each iteration.
    """
    curr_iters = 0
    start = generate_initial_state()
    visited = set()
    heap = []

    g_start = -compute_range(*start)
    h_start = heuristic(start)
    f_start = g_start + h_start
    heapq.heappush(heap, (f_start, start, "Start"))

    best_params = start
    best_range = compute_range(*start)

    best_ranges = [best_range]
    best_params_over_time = [best_params]
    param_changes = ["Start"]

    while heap and curr_iters < max_iters:
        f_curr, curr_params, changed_param = heapq.heappop(heap)

        if curr_params in visited:
            continue
        visited.add(curr_params)

        current_range = -f_curr

        if current_range > best_range:
            best_range = current_range
            best_params = curr_params

        best_ranges.append(best_range)
        best_params_over_time.append(best_params)
        param_changes.append(changed_param)

        neighbors = expand_neighbors(curr_params, visited, param_bounds)
        for neighbor in neighbors:
            heapq.heappush(heap, neighbor)

        curr_iters += 1

    return best_params, best_range, best_ranges, best_params_over_time, param_changes

# --- Run A* Search --- #
best_params, max_range, best_ranges, best_params_over_time, param_changes = astar()

print("Best Parameters:", best_params)
print("Max Range:", max_range)


# Shared visualization configuration
iterations = list(range(len(best_ranges)))
param_names = ["AR", "S", "LD", "fuel_mass"]
param_to_num = {"Start": -1, "AR": 0, "S": 1, "LD": 2, "fuel_mass": 3}
colors = {
    "AR": "tab:blue",
    "S": "tab:orange",
    "LD": "tab:green",
    "fuel_mass": "tab:red"
}
fig_folder_path = 'path/to/figures/a_star/'
best_params_over_time = np.array(best_params_over_time)

# --- Visualization: Max Range Over Iterations --- #
plt.figure()
plt.plot(iterations, best_ranges)
plt.xlabel("Iteration")
plt.ylabel("Max Range (m)")
plt.title("A* Optimization of Breguet Range")
plt.grid(True)
plt.tight_layout()
plt.savefig(fig_folder_path + 'range_plot.png')
plt.show()

# --- Visualization: Parameter Values Over Iterations --- #
plt.figure()
for name in param_names:
    idx = param_names.index(name)
    plt.plot(iterations, best_params_over_time[:, idx], label=name, color=colors[name])
plt.xlabel("Iteration")
plt.ylabel("Parameter Value")
plt.yscale('log')
plt.title("Parameter Values Over Iterations")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(fig_folder_path + 'param_values_plot.png')
plt.show()

# --- Visualization: Parameter Changed Each Iteration --- #
nums = [param_to_num[p] for p in param_changes]
scatter_colors = [colors[p] if p in colors else "gray" for p in param_changes]

plt.figure()
plt.scatter(iterations, nums, c=scatter_colors, marker='s')
plt.yticks(list(param_to_num.values()), list(param_to_num.keys()))
plt.xlabel("Iteration")
plt.ylabel("Changed Parameter")
plt.title("Parameter Changed at Each Iteration")
plt.grid(True)
plt.tight_layout()
plt.savefig(fig_folder_path + 'param_changes_plot.png')
plt.show()

# --- Visualization: Combined Plot --- #
plt.figure()
ax1 = plt.gca()

# Parameter value lines
for name in param_names:
    idx = param_names.index(name)
    ax1.plot(iterations, best_params_over_time[:, idx], label=name, color=colors[name])
ax1.set_ylabel("Parameter Value")
ax1.set_yscale('log')
ax1.grid(True)
ax1.legend(loc="upper left")

# Scatter of parameter changes
ax2 = ax1.twinx()
ax2.scatter(iterations, nums, c=scatter_colors, marker='s', alpha=0.6)
ax2.set_ylabel("Changed Parameter")
ax2.set_yticks(list(param_to_num.values()))
ax2.set_yticklabels(list(param_to_num.keys()))

plt.xlabel("Iteration")
plt.title("Parameter Values and Changes Over Iterations")
plt.tight_layout()
plt.savefig(fig_folder_path + 'param_combined_plot.png')
plt.show()
