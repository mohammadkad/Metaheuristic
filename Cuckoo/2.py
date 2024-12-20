import numpy as np
from sklearn.cluster import KMeans

def cuckoo_search(func, dim, n, pa, lb, ub, max_iter, k):
    """
    Cuckoo Search algorithm with clustering.

    Parameters:
    func (function): Objective function to optimize.
    dim (int): Number of dimensions.
    n (int): Number of nests.
    pa (float): Probability of alien eggs.
    lb (float): Lower bound of the search space.
    ub (float): Upper bound of the search space.
    max_iter (int): Maximum number of iterations.
    k (int): Number of clusters.

    Returns:
    best_x (list): Best solution found.
    best_f (float): Best fitness value.
    """
    # Initialize nests
    nests = np.random.uniform(lb, ub, (n, dim))

    # Evaluate initial nests
    fitness = np.array([func(x) for x in nests])

    # Initialize best solution
    best_x = nests[np.argmin(fitness)]
    best_f = np.min(fitness)

    for _ in range(max_iter):
        # Cluster nests using k-means
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(nests)
        cluster_centers = kmeans.cluster_centers_

        # Generate new solutions using Lévy flights
        new_nests = np.array([levy_flight(x, dim, lb, ub) for x in cluster_centers])

        # Evaluate new solutions
        new_fitness = np.array([func(x) for x in new_nests])

        # Replace worst nests with new solutions
        worst_nests = np.argsort(fitness)[-int(pa * n):]
        nests[worst_nests] = new_nests[worst_nests]
        fitness[worst_nests] = new_fitness[worst_nests]

        # Update best solution
        if np.min(new_fitness) < best_f:
            best_x = new_nests[np.argmin(new_fitness)]
            best_f = np.min(new_fitness)

    return best_x, best_f

def levy_flight(x, dim, lb, ub):
    """
    Lévy flight.

    Parameters:
    x (list): Current solution.
    dim (int): Number of dimensions.
    lb (float): Lower bound of the search space.
    ub (float): Upper bound of the search space.

    Returns:
    new_x (list): New solution.
    """
    # Generate Lévy distribution
    sigma = 1.0
    u = np.random.normal(0, sigma, dim)
    v = np.random.normal(0, sigma, dim)
    s = u / np.abs(v) ** (1.0 / dim)

    # Update solution
    new_x = x + 0.01 * s * (ub - lb)

    # Ensure solution is within bounds
    new_x = np.clip(new_x, lb, ub)

    return new_x

# Example usage
def objective_function(x):
    return np.sum(x ** 2)

dim = 5
n = 10
pa = 0.25
lb = -5
ub = 5
max_iter = 100
k = 3

best_x, best_f = cuckoo_search(objective_function, dim, n, pa, lb, ub, max_iter, k)
print("Best solution:", best_x)
print("Best fitness:", best_f)
