import numpy as np

class OneMax:
    
    def __call__(self, chromosome):
        '''
        Returns the numbers of ones.
        Maximum: chromosome consisting only of 1s
        '''
        return chromosome.sum()
    
    def __str__(self):
        return "OneMax"


class DeceptiveOneMax:
    
    def __call__(self, chromosome):
        '''
        Returns the numbers of ones or length plus one in the case of 0s only.
        Maximum: chromosome consisting only of 0s
        '''
        chromosome_sum = chromosome.sum()
        return chromosome.shape[0] + 1 if chromosome_sum == 0 else chromosome_sum
    
    def __str__(self):
        return "DeceptiveOneMax"


class KDeceptiveOneMax:
    
    def __init__(self, k):
        self.k = k

    def __call__(self, chromosome):
        '''
        Returns the sum of ones plus bonus for every section of length k that consists only of 0s.
        Maximum: chromosome consisting only of 0s
        '''
        chromosome_sum = 0
        for section in range(chromosome.shape[0]//self.k):
            part_sum = chromosome[section*self.k:(section+1)*self.k].sum()
            if part_sum == 0:
                chromosome_sum += self.k + 1
            else:
                chromosome_sum += part_sum
        return chromosome_sum

    def __str__(self):
        return f"KDeceptiveOneMax (k={self.k})"
