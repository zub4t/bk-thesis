import numpy as np
class ParticleFilter:
    def __init__(self, initial_particles, process_noise_std, measurement_noise_std):
        self.particles = initial_particles  # Nx2 array for N particles.
        self.process_noise_std = process_noise_std
        self.measurement_noise_std = measurement_noise_std

    def predict(self):
        # Add process noise, assuming a constant velocity model.
        noise = np.random.randn(*self.particles.shape) * self.process_noise_std
        self.particles += noise

    def update(self, measurement):
        # Compute weights based on how close each particle is to the measurement.
        distances = np.linalg.norm(self.particles - measurement, axis=1)
        weights = np.exp(-distances**2 / (2 * self.measurement_noise_std**2))
        weights += 1e-10  # avoid division by zero
        weights /= np.sum(weights)

        # Resample particles based on weights.
        indices = np.random.choice(np.arange(len(self.particles)), size=len(self.particles), p=weights)
        self.particles = self.particles[indices]

    def estimate(self):
        # Compute the weighted mean of the particles.
        return np.mean(self.particles, axis=0)
