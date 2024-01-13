import numpy as np
import wandb

class MIMIC:
    '''Class implementing MIMIC algorithm.'''
    
    def __init__(self,
                 problem_dimension,
                 initial_population_size,
                 descendant_population_size,
                 percentile,
                 fitness_function,
                 wandb_run,
                 max_iter = None,
                 premature_success = False,
                 **kwargs
                 ):
        self.problem_dimension = problem_dimension
        self.initial_population_size = initial_population_size
        self.percentile = percentile
        self.fitness_function = fitness_function
        self.max_iter = max_iter
        self.wandb_run = wandb_run
        self.descendant_population_size = descendant_population_size
        self.theta = None
        self.premature_success = premature_success

        self.conditional_probabilities = np.ones((self.problem_dimension, self.problem_dimension, 2, 2))*0.5

        population = np.random.random((self.initial_population_size, self.problem_dimension)) 
        self.population = (population < np.diagonal(self.conditional_probabilities[:, :, 1, 1])).astype('int')

    # Can be optimized: symmetric matrix
    def update_probabilities(self):
        '''Update probabilities.'''
        self.conditional_probabilities = np.zeros((self.problem_dimension, self.problem_dimension, 2, 2))
        for i in range(self.problem_dimension):
            for j in range(self.problem_dimension):
                for k in range(2):
                    for l in range(2):
                        self.conditional_probabilities[i, j, k, l] = np.sum((self.population[:, i] == k) & (self.population[:, j] == l))
        self.conditional_probabilities /= self.population.shape[0]

    def compute_fitness(self, chromosomes: np.ndarray):
        '''Compute fitness of all chromosomes.'''
        fitness = np.apply_along_axis(self.fitness_function, 1, chromosomes)
        self.wandb_run.log({'fitness_max': fitness.max(), 'fitness_mean': fitness.mean(), 'fitness_min': fitness.min()})
        return fitness
    
    def empirical_crossentropy(self, i: int, j: int):
        '''Compute empirical crossentropy.'''
        return -np.sum(self.conditional_probabilities[i, j]*np.log(self.conditional_probabilities[i, j]+1e-10))

    def pick_permutation(self):
        '''Pick permutation of variables.'''
        indices = [i for i in range(self.problem_dimension)]
        entropies = np.array([self.empirical_crossentropy(i, i) for i in range(self.problem_dimension)])
        permutation = [entropies.argmax(),]
        indices.remove(permutation[0])
        while len(indices) > 0:
            entropies = np.array([self.empirical_crossentropy(i, permutation[0]) for i in indices])
            max_id = indices[entropies.argmax()]
            permutation = [max_id,] + permutation
            indices.remove(max_id)

        self.permutation = permutation
        
    def sample_chromosome(self):
        '''Sample chromosome.'''
        chromosome = np.zeros((self.problem_dimension,)).astype('int')
        permutation_ids = self.permutation[::-1]
        chromosome[permutation_ids[0]] = np.random.rand() < self.conditional_probabilities[permutation_ids[0], permutation_ids[0], 1, 1]

        for i in range(1, len(permutation_ids)):
            chromosome[permutation_ids[i]] = np.random.rand() < (
                self.conditional_probabilities[i, permutation_ids[i-1], 1, chromosome[permutation_ids[i-1]]]
                / (self.conditional_probabilities[permutation_ids[i-1], permutation_ids[i-1], chromosome[permutation_ids[i-1]], chromosome[permutation_ids[i-1]]]+1e-10)
            )
        return chromosome

    def update_population_and_theta(self):
        population = np.vstack([self.sample_chromosome() for _ in range(self.descendant_population_size)])
        population = np.vstack([population, self.population])
        fitness = self.compute_fitness(population)
        self.theta = np.sort(fitness)[int((1 - self.percentile) * population.shape[0])]
        population = population[fitness >= self.theta]
        print(f'fitness_max: {fitness.max()}, fitness_mean: {fitness.mean()}, fitness_min: {fitness.min()} theta: {self.theta}, population_size: {population.shape[0]}')
        self.population = population
        return fitness.max() - fitness.min() == 0 or population.shape[0] > self.initial_population_size*10

    def evolve(self):
        '''Run MIMIC algorithm.'''
        if self.max_iter:
            for _ in range(self.max_iter):
                self.update_probabilities()
                self.pick_permutation()
                done = self.update_population_and_theta()
                if done and self.premature_success:
                    break
        else:
            done = False
            while not done:
                self.update_probabilities()
                self.pick_permutation()
                done = self.update_population_and_theta()
        self.compute_fitness(self.population)

