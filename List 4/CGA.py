import numpy as np
import wandb

class CGA:
    '''Class implementing Compact Genetic Algorithm.'''
    def __init__(self,
                problem_dimension,
                learning_rate,
                fitness_function,
                max_iter,
                wandb_run,
                **kwargs
                ):
        self.problem_dimension = problem_dimension
        self.population_size = 2
        self.learning_rate = learning_rate
        self.fitness_function = fitness_function
        self.max_iter = max_iter
        self.wandb_run = wandb_run

        self.probabilities = np.ones((self.problem_dimension,))*0.5
        self.population = self.random_population()

    def random_population(self):
        '''Generate random population.'''
        population = np.random.random((self.population_size, self.problem_dimension))
        population = (population < self.probabilities).astype('int')
        return population    

    def compute_fitness(self, chromosomes: np.ndarray):
        '''Compute fitness of all chromosomes.'''
        fitness = np.apply_along_axis(self.fitness_function, 1, chromosomes)
        self.wandb_run.log({'fitness_max': fitness.max(), 'fitness_mean': fitness.mean(), 'fitness_min': fitness.min()})
        return fitness
    
    def update_probabilities(self, best_chromosome, worst_chromosome):
        '''Update probabilities.'''
        chromosome_diff = best_chromosome - worst_chromosome
        self.probabilities += self.learning_rate*chromosome_diff

    def evolve(self):
        '''Run CGA algorithm.'''
        for _ in range(self.max_iter):
            fitness = self.compute_fitness(self.population)
            best_chromosome = self.population[fitness.argmax()]
            worst_chromosome = self.population[fitness.argmin()]
            self.update_probabilities(best_chromosome, worst_chromosome)
            self.population = self.random_population()
        self.compute_fitness(self.population)
