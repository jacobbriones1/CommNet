import numpy as np

def sample_distribution(distribution):    
    num_actions = len(list(distribution))
    return np.random.choice([i for i in range(num_actions)], p = distribution)

def sample_distributions(distribution_list):
    return [sample_distribution(distribution) for distribution in distribution_list]
