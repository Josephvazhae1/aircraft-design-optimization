# Aircraft Design Optimization

Authors: [Joseph Vazhaeparampill](https://github.com/Josephvazhae1), [Daniel Quinteros](https://github.com/d-quinteros), 

## Introduction

This project implements three different algorithms to optimize intial aircraft design parameters. By testing gradient descent, A*, and genetic algorithms, we identify the best strategies to optimize the Brequet Range Equation, which predicts the maximum distance an aircraft can travel.

## Table of Contents

- [Libraries Used](#libraries-used)
- [Resources Used](#resources-used)
- [Project Structure](#project-structure)
- [Background of Algorithm](#background-of-algorithm)
- [How the Algorithm Works](#how-the-algorithm-works)

## Libraries Used
- NumPy: For numerical operations
- Matplotlib: For visualization of each iteration of the genetic algorithm
- Random: For initial individual generation

## Resources Used
- Holland, John H. “Genetic Algorithms.” Scientific American, vol. 267, no. 1, 1992, pp. 66–73. JSTOR, http://www.jstor.org/stable/24939139. Accessed 21 Apr. 2025.
- Spakovszky, Zoltan. "13.3 Aircraft Range: the Breguet Range Equation." Thermodynamics and Propulsion, https://web.mit.edu/16.unified/www/FALL/thermodynamics/notes/node98.html
- Chaudhry, Imran Ali. "Preliminary Aircraft Design Optimization Using Genetic Algorithms." Marsland Press, https://www.sciencepub.net/researcher/rsj130721/10_20117rsj130721_49_60.pdf
- Raymer, Daniel. "Aircraft Design: A Conceptual Approach." American Institute of Aeronautics and Astronautics, https://www.airloads.net/Downloads/Textbooks/Aircraft%20Design-A%20Conceptual%20Approach.pdf 

## Genetic Algorithms
### Background of Algorithm

Genetic Algorithms (GAs) are a class of metaheuristic, population-based optimization algorithms inspired by the principles of natural selection and evolutionary biology. Initially popularized by John Holland in the 1970s, GAs simulate the process of evolution by iteratively selecting, recombining, and mutating candidate solutions to evolve toward an optimal or near-optimal result.

Unlike gradient-based optimization techniques, GAs are not limited by differentiability or continuity of the objective function. This makes them especially useful for solving complex, high-dimensional, and non-convex problems where analytical gradients are difficult or impossible to compute. GAs are also highly flexible and can be adapted to a wide range of applications, including combinatorial optimization, machine learning hyperparameter tuning, scheduling, and circuit design.

A typical genetic algorithm begins with a randomly initialized population of chromosomes—encoded representations of possible solutions. Each individual is evaluated using a fitness function, and those with higher fitness are more likely to be selected for reproduction. New offspring are generated through crossover (recombination) of parent chromosomes and mutation, introducing variation into the population and helping avoid premature convergence. Over successive generations, the population evolves to exhibit increasingly fit individuals.

However, genetic algorithms are not without their limitations. Their performance can be sensitive to the choice of parameters such as mutation rate, crossover probability, population size, and selection strategy. Poor parameter tuning can lead to premature convergence on suboptimal solutions or slow convergence overall. Additionally, the reliance on stochastic operators can make the results non-deterministic, requiring multiple runs to gain confidence in the outcome. GAs also tend to require more computational resources compared to simpler methods, especially in problems with large search spaces or costly fitness evaluations.

Despite these challenges, GAs remain a widely used and powerful optimization technique, particularly when the search landscape is rugged or poorly understood. Their ability to maintain a diverse population of solutions and explore multiple regions of the search space in parallel makes them a robust choice for many real-world optimization problems.

### How the Algorithm Works

1. Initialize Population
Randomly generate a set of candidate aircraft designs (individuals).
Each individual has severa; genes: Aspect Ratio (AR), Wing Area (S), L/D ratio, Fuel Mass, etc.
We performed a kind of parameter sweep to determine the optimal population size. As can be seen from the graphs below, we see a significant improvement as the population size increase from 10 to twenty, but this advantage proves almost insignificant as the population size grows to 50 or 100, only adding extra runtime with minimal benefits in the convergence.
![pop10](https://github.com/user-attachments/assets/841abddd-f48e-419d-ba9d-1dec4f37e9e9)
![pop20](https://github.com/user-attachments/assets/742ecb3d-b768-4190-bce3-69b38dd6eec1)
![pop50](https://github.com/user-attachments/assets/f2e8b4fa-d87e-4532-a573-ec549c5f785b)
![pop100](https://github.com/user-attachments/assets/01214c10-5030-43c7-bb72-c03cf0aa4c39)

3. Evaluate Fitness
For each individual:
Compute the aircraft's initial and final weight using a mass model.
![image](https://github.com/user-attachments/assets/73085ff2-a3df-40e0-a2b8-cc4a8293bab7)

Use the Breguet range equation to calculate the fitness = estimated flight range.
![image](https://github.com/user-attachments/assets/997f8ab7-1296-4232-bb74-b9ac7e4bdde1)

![image](https://github.com/user-attachments/assets/c0aedada-0a25-4dfc-af5d-9fce8338b24c)

We will be optimizing the above colored parameters.

3. Choose the individuals to be the parents for the next generation.
We decided to test two different parent selection strategies.
The first parent selection strategy was roullete wheel. In roullete wheel selection, the probability of an individual being chosen as a parent is proportional to its fitness. As such, because roullete wheels ensure every individual has at least some chance of being selected, it does a good job of introducing variety into our genetic algorithm. The following shows the results of using our genetic algorithm with roullete wheel selection:

![roulette](https://github.com/user-attachments/assets/b5369349-86c3-4282-a469-5c8a9787279f)

The second parent selection mechanism we tested was the tournament selection. In tournament selection, a random selection of the population was taken and the fittest individuals from that selection were chosen to serve as parents. Tournament selection allows you to control the selective pressure by varying the size of the tournament. A larger tournament means competing with more fit individuals, introducing less variance than a smaller tournament. The following graph shows the results of using our genetic algorithm with tournament selection: 
![tournament](https://github.com/user-attachments/assets/699fe5e1-167a-48da-a3fa-dfa5b0f14996)

As you can see it converged much faster on the optimal value compared to roullete wheel selection.

5. Crossover: We implemented uniform crossover. This means that each gene is randomly inherits from one parent. For example, a childs aspect ratio can come from Parent 1, wing surface area from parent 2, lift/drag coefficient from parent 1 etc. This does a sufficient job of increasing diversity in our genetic algorithm.


6. Mutation: Slightly modify each child’s genes to maintain genetic diversity. We ulitized a mutation rate of 100%. We also restricted the mutation to scale by 10% at most. Because each gene was represented by a double instead of a bit, it made sense to restrict the mutation to allow for a better, more consistent convergence. The goal for this mutation strategy was to maintain genetic diversity and avoid local optima.
 
7. Create New Generation
Replace the old population with:
The elite individuals from the previous generation (e.g., best 2 designs).
The new children produced by crossover and mutation.

8. Repeat
Go back to Step 2 for a fixed number of generations, or until a convergence criterion is met (e.g., no improvement in best range).

10. Visualize/Analyze Results
Throughouts our gentic algorithm we tracked and/or visualized the following to better evaluate the performance of the genetic algorithme:
 - Evolution of the best design per generation.

 - Range improvements over time.

 - Final optimal aircraft geometry and performance.
   
The following graph visualizes the performance of our genetic algorithm: 
![tournament](https://github.com/user-attachments/assets/699fe5e1-167a-48da-a3fa-dfa5b0f14996)


## A* Algorithm
### Background of Algorithm

A* is a well-known pathfinding and graph traversal algorithm. [Include history].

At a high level, A* works by exploring paths in a graph using a cost function: f(n) = g(n) + h(n). Here, g(n) is the cost from the starting point to the current node n, while h(n) is a heuristic estimate of the remaining cost from n to the goal. By balancing these two, A* is able to prioritize nodes that are both promising and efficient to reach and is guaranteed to find the best (i.e. shortest) path.

Where A* really shines is in discrete spaces, like a grid map or a network of nodes, where the cost of moving from one node to another is clear and the heuristic function can guide the search toward the goal. Some common heuristics include things like Euclidean or Manhattan distance.

However, A* has its limitations. One major drawback is its memory usage: the algorithm stores all visited nodes and keeps track of the entire open set, which can become computationally expensive for large or dense graphs. Another key limitation, especially relevant to our use case, is that A* is not well-suited for continuous optimization problems. In continuous spaces, where parameters can take on an infinite number of values, defining a meaningful and efficient graph structure for A* to navigate becomes extremely challenging. Since A* relies on discrete states and predefined neighbors, it’s just not built for smooth landscapes or open-ended parameter spaces. Algorithms like gradient descent, evolutionary strategies, or Bayesian optimization tend to work better for that kind of task.

More importantly, defining a good heuristic in this context is difficult. In typical pathfinding, your heuristic might be something like the Euclidean distance to the goal. But when you’re trying to optimize an objective function, you don’t actually know what the “goal” value is. That makes it tough to estimate how “far” a candidate solution is from the best one. Without a meaningful heuristic, A* loses one of its main advantages and can start behaving more like a brute-force search.

So while A* is useful for pathfinding and decision-making in well-structured environments, it's not ideal for tasks like tuning continuous parameters to maximize an objective function.

### How the Algorithm Works

1. Initialization: We first randomly generate an initial aircraft configuration within specified bounds for each parameter:
- Aspect Ratio (AR): 5 to 20
- Wing Area (S): 10 to 80 m²
- Lift-to-Drag Ratio (L/D): 10 to 25
- Fuel Mass: 0 to 10,000 kg

This initial state serves as the starting node for the A* search algorithm. Typically, you'd also specify a goal node (or state), but in the context of our application, our "goal" is uknown since we are trying to find the optimal soultion.

2. Cost Function: Like previously mentioned, the component that makes A* so powerful is it's unique cost function:
![image](https://github.com/user-attachments/assets/3a24c49f-a6c1-472a-a9f2-3487b880a62f)
where:
- f(n): the estimate of the total cost from start node to goal node through node n
- g(n): actual cost from start node to node n
- h(n): estimated cost from node n to goal node

In the context of our application
- g(n) = the negative output of the Breguet range equation for the node n
![image](https://github.com/user-attachments/assets/997f8ab7-1296-4232-bb74-b9ac7e4bdde1)
- h(n) = 0; because we have no way to estimate the distance from our current node (current parameters values) to our goal node (optimal parameters values).


 
## Gradient Descent Algorithm
