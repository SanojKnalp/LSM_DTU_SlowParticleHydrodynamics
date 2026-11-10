import numpy as np
import Particle
from mpi4py import MPI

# our grid class.
# currently work in progress
class Grid:
    def __init__(self, lengths):
        self.buffer = np.array([lengths, [], []], dtype=object);
        self.bounding_box = self.buffer[0];
        self.list_of_owned_particles = self.buffer[1];
        self.list_of_ghost_particles = self.buffer[2];
        


    def repartition_grid(self):
        return 0;

    def request_ghost_cells(self):
        return 0;

    def compute_density(self, particle):
        return 0;

