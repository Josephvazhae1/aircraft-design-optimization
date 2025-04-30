import numpy as np
import matplotlib.pyplot as plt

# Fitness function (Breguet Range)
def fitness(individual):
    LD, S, AR, fuel_mass = individual
    payload_mass = 1000  # kg
    SFC = 0.6  # 1/hr
    V = 250  # m/s

    if not (10 <= LD <= 25 and 10 <= S <= 80 and 5 <= AR <= 20 and 0 <= fuel_mass <= 10000):
        return 0

    Wi = 20 * (S * AR)**0.6 + fuel_mass + payload_mass
    Wf = 20 * (S * AR)**0.6 + payload_mass
    if Wf >= Wi:
        return 0

    R = (V / SFC) * LD * np.log(Wi / Wf)
    return R

# Crossover
def crossover(p1, p2):
    alpha = np.random.rand()
    return alpha * p1 + (1 - alpha) * p2

# Mutation
def mutate(individual, mutation_rate=0.1):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] += np.random.normal(0, 0.05 * individual[i])
    return np.clip(individual, [10, 10, 5, 0], [25, 80, 20, 10000])

# Selection: Roulette Wheel
def roulette_wheel_selection(pop, fitnesses):
    total = np.sum(fitnesses)
    if total == 0:
        return pop[np.random.randint(len(pop), size=len(pop))]
    probabilities = fitnesses / total
    indices = np.random.choice(len(pop), size=len(pop), p=probabilities)
    return pop[indices]

# Selection: Tournament
def tournament_selection(pop, fitnesses, k=3):
    selected = []
    for _ in range(len(pop)):
        competitors = np.random.choice(len(pop), k)
        winner = competitors[np.argmax(fitnesses[competitors])]
        selected.append(pop[winner])
    return np.array(selected)

# Main GA function with plotting
def run_genetic_algorithm(
    fitness_func,
    pop_size=20,
    generations=50,
    selection_strategy='tournament',
    mutation_rate=0.1,
    elitism_count=2
):
    population = np.random.uniform([10, 10, 5, 0], [25, 80, 20, 10000], (pop_size, 4))
    best_fitnesses = []

    for gen in range(generations):
        fitnesses = np.array([fitness_func(ind) for ind in population])
        best_fitness = np.max(fitnesses)
        best_fitnesses.append(best_fitness)

        elite_indices = np.argsort(fitnesses)[-elitism_count:]
        elites = population[elite_indices]

        if selection_strategy == 'tournament':
            selected = tournament_selection(population, fitnesses)
        elif selection_strategy == 'roulette':
            selected = roulette_wheel_selection(population, fitnesses)
        else:
            raise ValueError("Unknown selection strategy.")

        next_gen = elites.tolist()
        while len(next_gen) < pop_size:
            p1, p2 = selected[np.random.randint(pop_size)], selected[np.random.randint(pop_size)]
            child = mutate(crossover(p1, p2), mutation_rate)
            next_gen.append(child)

        population = np.array(next_gen)

    # Plot range improvement over generations
    plt.plot(best_fitnesses, label='Best Range')
    plt.xlabel("Generation")
    plt.ylabel("Range (km)")
    plt.title("Breguet Range Optimization Over Time with Elitism")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return population, best_fitnesses

# Executing an example run
run_genetic_algorithm(fitness_func=fitness, selection_strategy='tournament')

