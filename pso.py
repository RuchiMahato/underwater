import numpy as np

# Particle class definition
class Particle:
    def __init__(self, func, dim, vmin, vmax, seed):
        np.random.seed(seed)
        self.velocity = np.zeros(dim)
        self.position = np.random.uniform(vmin, vmax, dim)
        self.best_part_pos = np.copy(self.position)

        self.fitness = func(self.position)
        self.best_part_fitness = self.fitness

# PSO algorithm
def pso(func, max_iter, num_particles, dim, vmin, vmax, params):
    wmax = params["wmax"]  # Max inertia
    wmin = params["wmin"]  # Min inertia
    c1 = params["c1"]      # Cognitive coefficient
    c2 = params["c2"]      # Social coefficient

    swarm = [Particle(func, dim, vmin, vmax, i) for i in range(num_particles)]

    best_swarm_pos = np.zeros(dim)
    best_swarm_fitness = np.inf

    for particle in swarm:
        if particle.fitness < best_swarm_fitness:
            best_swarm_fitness = particle.fitness
            best_swarm_pos = np.copy(particle.position)

    for it in range(max_iter):
        if it % 5 == 0:
            print(f"Iteration = {it}, Best Fitness = {best_swarm_fitness:.6f}")

        w = wmax - ((wmax - wmin) / max_iter) * it

        for particle in swarm:
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            cognitive = c1 * r1 * (particle.best_part_pos - particle.position)
            social = c2 * r2 * (best_swarm_pos - particle.position)

            particle.velocity = w * particle.velocity + cognitive + social
            particle.position += particle.velocity

            particle.position = np.clip(particle.position, vmin, vmax)

            particle.fitness = func(particle.position)

            if particle.fitness < particle.best_part_fitness:
                particle.best_part_fitness = particle.fitness
                particle.best_part_pos = np.copy(particle.position)

            if particle.fitness < best_swarm_fitness:
                best_swarm_fitness = particle.fitness
                best_swarm_pos = np.copy(particle.position)

    return {"position": best_swarm_pos, "cost": best_swarm_fitness}
