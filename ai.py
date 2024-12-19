import random
import numpy as np
import matplotlib.pyplot as plt


def fitness(genome, equation):
    value = equation(genome)
    return np.inf if abs(value) == 0 else 1 / (abs(value) + 1e-6)


def select(population, n, fitness_fun, equation):
    return sorted(population, key=lambda genome: fitness_fun(genome, equation), reverse=True)[:n]


def select_parents(population):
    return random.choices(population, k=2)


def crossover(parents):
    child = [(p1 + p2) / 2 for p1, p2 in zip(parents[0], parents[1])]
    return child


def mutate(genome, mutation_rate=0.1):
    return [gene + random.uniform(-mutation_rate, mutation_rate) for gene in genome]


def generate_population(size, genome_size=1, value_range=(-100, 100)):
    return [[random.uniform(*value_range) for _ in range(genome_size)] for _ in range(size)]


def genetic_algorithm(equation, population_size=100, genome_size=1, generations=500, mutation_rate=0.1,
                      goal_fitness=1000):
    population = generate_population(population_size, genome_size)
    fitness_history=[]

    for generation in range(generations):
        top_population=select(population, population_size // 2, fitness, equation)
        best_genome=top_population[0]
        best_fitness =fitness(best_genome, equation)
        fitness_history.append(best_fitness)

        print(f"Generation {generation}, Best Fitness: {best_fitness:.4f}, Best Genome: {best_genome}")

        if best_fitness >= goal_fitness:
            print("Goal fitness achieved!")
            break
        new_population = []
        while len(new_population) < population_size:
            parents= select_parents(top_population)
            child =crossover(parents)
            child=mutate(child, mutation_rate)
            new_population.append(child)
        population = new_population

    plt.plot(fitness_history, label="Best Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness Over Generations")
    plt.legend()
    plt.show()
    return best_genome


def custom_equation(x):
    #return np.sin(10*np.pi*x[0])/(2*x[0])+(x[0]-1)**4 #Gramacy and LEE 2012
    #return (x[0]**2) #sphere
    #return -20 * np.exp(-0.2 * np.sqrt(x[0] ** 2)) - np.exp(np.cos(2 * np.pi * x[0])) + 20 + np.e  #Achley
    #return 100 * (x[0]**2 - x[0])**2 + (1 - x[0])**2 #Rosenbrock 1,0,0.3

    #Griewank
    sum_term = sum([xi**2 / 4000 for xi in x])
    prod_term = np.prod([np.cos(xi / np.sqrt(i+1)) for i, xi in enumerate(x)])
    return 1 + sum_term - prod_term


best_solution = genetic_algorithm(custom_equation, population_size=1000, generations=100, mutation_rate=0.5,goal_fitness=2000000)
print(f"Best solution found: {best_solution}, Fitness: {fitness(best_solution, custom_equation):.4f}")
