import numpy as np
import wandb

class PBIL:
    def __init__(self,
                 problem_dimension,
                 population_size,
                 learning_rate,
                 mutation_rate,
                 mutation_distortion_rate,
                 fitness_function,
                 max_iter,
                wandb_run,
                 **kwargs
                 ):
        self.problem_dimension = problem_dimension
        self.population_size = population_size
        self.learning_rate = learning_rate
        self.mutation_rate = mutation_rate
        self.mutation_distortion_rate = mutation_distortion_rate
        self.fitness_function = fitness_function
        self.max_iter = max_iter
        self.wandb_run = wandb_run

        self.probabilities = np.ones((self.problem_dimension,))*0.5
        self.population = self.random_population()

    def random_population(self):
        population = np.random.random((self.population_size, self.problem_dimension))
        population = (population < self.probabilities).astype('int')
        return population

    def compute_fitness(self, chromosomes: np.ndarray):
        '''Compute fitness of all chromosomes.'''
        fitness = np.apply_along_axis(self.fitness_function, 1, chromosomes)
        self.wandb_run.log({'fitness_max': fitness.max(), 'fitness_mean': fitness.mean(), 'fitness_min': fitness.min()})
        return fitness
    
    def update_probabilities(self, best_chromosome):
        '''Update probabilities.'''
        self.probabilities = self.probabilities*(1-self.learning_rate) + self.learning_rate*best_chromosome
        if np.random.rand() < self.mutation_rate:
            self.probabilities = self.probabilities*(1-self.mutation_distortion_rate) + self.mutation_distortion_rate*int(np.random.rand()<0.5)

    def evolve(self):
        for _ in range(self.max_iter):
            fitness = self.compute_fitness(self.population)
            best_chromosome = self.population[fitness.argmax()]
            self.update_probabilities(best_chromosome)
            self.population = self.random_population()
        self.compute_fitness(self.population)