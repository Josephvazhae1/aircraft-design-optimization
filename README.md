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

## Resources Used
- Holland, John H. “Genetic Algorithms.” Scientific American, vol. 267, no. 1, 1992, pp. 66–73. JSTOR, http://www.jstor.org/stable/24939139. Accessed 21 Apr. 2025.
- Spakovszky, Zoltan. "13.3 Aircraft Range: the Breguet Range Equation." Thermodynamics and Propulsion, https://web.mit.edu/16.unified/www/FALL/thermodynamics/notes/node98.html
- Chaudhry, Imran Ali. "Preliminary Aircraft Design Optimization Using Genetic Algorithms." Marsland Press, https://www.sciencepub.net/researcher/rsj130721/10_20117rsj130721_49_60.pdf
- Raymer, Daniel. "Aircraft Design: A Conceptual Approach." American Institute of Aeronautics and Astronautics, https://www.airloads.net/Downloads/Textbooks/Aircraft%20Design-A%20Conceptual%20Approach.pdf 

## Background of Algorithm

Genetic Algorithms (GAs) are a class of metaheuristic, population-based optimization algorithms inspired by the principles of natural selection and evolutionary biology. Initially popularized by John Holland in the 1970s, and later expanded upon by researchers such as David E. Goldberg, GAs simulate the process of evolution by iteratively selecting, recombining, and mutating candidate solutions to evolve toward an optimal or near-optimal result.

Unlike gradient-based optimization techniques, GAs are not limited by differentiability or continuity of the objective function. This makes them especially useful for solving complex, high-dimensional, multi-modal, and non-convex problems where analytical gradients are difficult or impossible to compute. GAs are also highly flexible and can be adapted to a wide range of applications, including combinatorial optimization, machine learning hyperparameter tuning, scheduling, and circuit design.

A typical genetic algorithm begins with a randomly initialized population of chromosomes—encoded representations of possible solutions. Each individual is evaluated using a fitness function, and those with higher fitness are more likely to be selected for reproduction. New offspring are generated through crossover (recombination) of parent chromosomes and mutation, introducing variation into the population and helping avoid premature convergence. Over successive generations, the population evolves to exhibit increasingly fit individuals.

However, genetic algorithms are not without their limitations. Their performance can be sensitive to the choice of parameters such as mutation rate, crossover probability, population size, and selection strategy. Poor parameter tuning can lead to premature convergence on suboptimal solutions or slow convergence overall. Additionally, the reliance on stochastic operators can make the results non-deterministic, requiring multiple runs to gain confidence in the outcome. GAs also tend to require more computational resources compared to simpler methods, especially in problems with large search spaces or costly fitness evaluations.

Despite these challenges, GAs remain a widely used and powerful optimization technique, particularly when the search landscape is rugged or poorly understood. Their ability to maintain a diverse population of solutions and explore multiple regions of the search space in parallel makes them a robust choice for many real-world optimization problems.

## How the Algorithm Works

1. Initialize Population
Randomly generate a set of candidate aircraft designs (individuals).

Each individual has severa; genes: Aspect Ratio (AR), Wing Area (S), L/D ratio, Fuel Mass, etc.

Function: generate_individual() - Randomly creates 20 individuals, each with genes:
  - AR (Aspect Ratio)
  - S (Wing Area)
  - LD (Lift-to-Drag Ratio)
  - fuel_mass
  - (Fixed payload mass, cruise speed V, and specific fuel consumption c)

2. Evaluate Fitness
For each individual:
Compute the aircraft's initial and final weight using a mass model.
![image](https://github.com/user-attachments/assets/73085ff2-a3df-40e0-a2b8-cc4a8293bab7)

Use the Breguet range equation to calculate the fitness = estimated flight range.

3. Select the Fittest
Sort the population by fitness (longer range = better).

Select the top performers (e.g., top 10%) to act as parents for the next generation.

4. Crossover (Mating)
Randomly pair parents to produce children.

Each child inherits a mix of traits (genes) from the parents.


5. Mutation
Slightly modify each child’s genes to maintain genetic diversity.


6. Create New Generation
Replace the old population with:
The elite individuals from the previous generation (e.g., best 2 designs).
The new children produced by crossover and mutation.

7. Repeat
Go back to Step 2 for a fixed number of generations, or until a convergence criterion is met (e.g., no improvement in best range).

8. Visualize/Analyze Results
Track or visualize:

 - Evolution of the best design per generation.

 - Range improvements over time.

 - Final optimal aircraft geometry and performance.
The following 
