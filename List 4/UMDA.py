import numpy as np
import wandb

class UMDA:
    '''Class implementing Univariate Marginal Distribution Algorithm.'''
    def __init__(self, top_k, population_size, max_iter, fitness_function, problem_dimension,
                wandb_run, **kwargs):
        self.top_k = top_k
        self.population_size = population_size
        self.max_iter = max_iter
        self.fitness_function = fitness_function
        self.problem_dimension = problem_dimension
        self.wandb_run = wandb_run
        
        self.probabilities = np.ones((self.problem_dimension,)) * .5
        self.population = self.random_population()
        
    def random_population(self):
        '''Generate random population.'''
        population = np.random.random((self.population_size, self.problem_dimension))
        population = (population < self.probabilities).astype('int')
        return population 
    
    def select_top_k(self, fitness):
        '''Select top k individuals from population based on fitness.'''
        return self.population[np.argsort(fitness)][-self.top_k:]
    
    def update_probabilities(self, top_chromosomes):
        '''Update probabilities based on top k individuals.'''
        self.probabilities = np.mean(top_chromosomes, axis=0)

    def compute_fitness(self, chromosomes: np.ndarray):
        '''Compute fitness of all chromosomes.'''
        fitness = np.apply_along_axis(self.fitness_function, 1, chromosomes)
        self.wandb_run.log({'fitness_max': fitness.max(), 'fitness_mean': fitness.mean(), 'fitness_min': fitness.min()})
        return fitness

    def evolve(self):
        '''Run UMDA algorithm.'''
        for _ in range(self.max_iter):
            fitness = self.compute_fitness(self.population)
            top_chromosomes = self.select_top_k(fitness)
            self.update_probabilities(top_chromosomes)
            self.population = self.random_population()
        self.compute_fitness(self.population)