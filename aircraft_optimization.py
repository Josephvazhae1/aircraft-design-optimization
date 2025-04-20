import numpy as np
import random
import matplotlib.pyplot as plt

def estimate_empty_mass(S, AR, k_struct=20):
    return k_struct * (S * AR)**0.6

def compute_weights(S, AR, fuel_mass, payload_mass):
    empty_mass = estimate_empty_mass(S, AR)
    W_i = empty_mass + fuel_mass + payload_mass
    W_f = empty_mass + payload_mass
    return W_i, W_f

def compute_range(LD, V, c, W_i, W_f):
    if LD <= 0 or W_i <= W_f:
        return 0
    return (V / c) * LD * np.log(W_i / W_f)


def generate_individual():
    return {
        'AR': random.uniform(6, 12),    #Aspect Ratio
        'S': random.uniform(15, 30),    #Wing Area
        'LD': random.uniform(10, 20),  # Lift/Drag estimate
        'fuel_mass': random.uniform(500, 2000),
        'payload_mass': 500,  # fixed
        'V': 250,             # m/s     #Cruise Velocity
        'c': 0.6              # SFC     #Specific Fuel COnsumption
    }

def evaluate(ind):
    W_i, W_f = compute_weights(ind['S'], ind['AR'], ind['fuel_mass'], ind['payload_mass'])
    return compute_range(ind['LD'], ind['V'], ind['c'], W_i, W_f)

def mutate(ind):
    for key in ['AR', 'S', 'LD', 'fuel_mass']:
        ind[key] *= random.uniform(0.95, 1.05)
    return ind

def crossover(p1, p2):
    return {
        key: random.choice([p1[key], p2[key]])
        for key in p1
    }

# # --- Live Plotting Function ---
# def live_plot_aircraft(design, generation):
#     plt.clf()
#     S, AR = design['S'], design['AR']
#     span = np.sqrt(AR * S)
#     chord = S / span
#     fuselage_length = span * 0.8
#     fuselage_width = chord * 0.2

#     ax = plt.gca()
#     ax.set_aspect('equal')
#     ax.set_xlim(-25, 25)
#     ax.set_ylim(-25, 25)
#     ax.set_title(f"Generation {generation+1} — S: {S:.1f}, AR: {AR:.1f}")

#     # Wing and fuselage
#     wing = patches.Rectangle((-chord/2, -span/2), chord, span, edgecolor='blue', facecolor='skyblue')
#     fuselage = patches.Rectangle((-fuselage_length/2, -fuselage_width/2), fuselage_length, fuselage_width, edgecolor='black', facecolor='gray')

#     ax.add_patch(wing)
#     ax.add_patch(fuselage)
#     plt.pause(0.8)

# # --- Genetic Algorithm ---
# def genetic_algorithm_live(gens=10, pop_size=20):
#     pop = [generate_individual() for _ in range(pop_size)]

#     plt.ion()  # Enable interactive plotting
#     fig = plt.figure()

#     for gen in range(gens):
#         scored = [(ind, evaluate(ind)) for ind in pop]
#         scored.sort(key=lambda x: x[1], reverse=True)
#         best = scored[0][0]

#         # Live plot current best
#         live_plot_aircraft(best, gen)

#         # Generate next population
#         new_pop = [best, scored[1][0]]
#         while len(new_pop) < pop_size:
#             p1, p2 = random.choices(scored[:10], k=2)
#             child = crossover(p1[0], p2[0])
#             new_pop.append(mutate(child))

#         pop = new_pop

#     plt.ioff()
#     plt.show()

# # --- Run It ---
# genetic_algorithm_live(gens=10)

def genetic_algorithm(gens=30, pop_size=20):
    pop = [generate_individual() for _ in range(pop_size)]
    best_fitnesses = []

    for gen in range(gens):
        scored = [(ind, evaluate(ind)) for ind in pop]
        scored.sort(key=lambda x: x[1], reverse=True)
        best = scored[0]
        best_fitnesses.append(best[1])
        print(f"Gen {gen+1}: Best range = {best[1]:.2f} m")

        new_pop = [best[0], scored[1][0]]  # elitism
        while len(new_pop) < pop_size:
            p1, p2 = random.choices(scored[:10], k=2)
            child = crossover(p1[0], p2[0])
            new_pop.append(mutate(child))

        pop = new_pop

    return best_fitnesses, best[0]


fitness_history, best_design = genetic_algorithm()

plt.plot(fitness_history)
plt.title("Genetic Algorithm Optimization of Breguet Range")
plt.xlabel("Generation")
plt.ylabel("Best Range (m)")
plt.grid(True)
plt.show()

print("\nBest Design Found:")
for k, v in best_design.items():
    print(f"{k}: {v:.2f}")
