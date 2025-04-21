from aircraft_optimization import compute_weights, compute_range
import heapq
import random

def generate_initial_state():
    return {
        'AR': random.uniform(6, 12),
        'S': random.uniform(15, 30),
        'LD': random.uniform(10, 20),
        'fuel_mass': random.uniform(500, 2000),
        'payload_mass': 500,
        'V': 250,
        'c': 0.6
    }

def evaluate(state):
    W_i, W_f = compute_weights(state['S'], state['AR'], state['fuel_mass'], state['payload_mass'])
    return compute_range(state['LD'], state['V'], state['c'], W_i, W_f)

def heuristic(state):
    future = state.copy()
    future['LD'] = min(future['LD'] * 1.1, 20)
    future['fuel_mass'] = min(future['fuel_mass'] * 1.2, 2000)
    W_i, W_f = compute_weights(future['S'], future['AR'], future['fuel_mass'], future['payload_mass'])
    return -compute_range(future['LD'], future['V'], future['c'], W_i, W_f)

def neighbors(state):
    deltas = {
        'AR': 0.5,
        'S': 1.0,
        'LD': 0.5,
        'fuel_mass': 100
    }
    neighbor_states = []
    for key, delta in deltas.items():
        for direction in [-1, 1]:
            new_state = state.copy()
            new_state[key] = max(1, new_state[key] + direction * delta)
            neighbor_states.append(new_state)
    return neighbor_states

def a_star_search(max_iter=500):
    start = generate_initial_state()
    start_range = evaluate(start)
    frontier = [( -(start_range + heuristic(start)), start )]
    visited = set()
    best = (start_range, start)

    for _ in range(max_iter):
        if not frontier:
            break
        _, current = heapq.heappop(frontier)
        key = tuple(round(current[k], 2) for k in ['AR', 'S', 'LD', 'fuel_mass'])
        if key in visited:
            continue
        visited.add(key)

        curr_range = evaluate(current)
        if curr_range > best[0]:
            best = (curr_range, current)

        for neighbor in neighbors(current):
            n_key = tuple(round(neighbor[k], 2) for k in ['AR', 'S', 'LD', 'fuel_mass'])
            if n_key not in visited:
                cost = -evaluate(neighbor) + heuristic(neighbor)
                heapq.heappush(frontier, (cost, neighbor))

    return best

