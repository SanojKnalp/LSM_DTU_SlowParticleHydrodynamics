import numpy as np
import Particle
from mpi4py import MPI

# our grid class.
# currently work in progress
class Grid:
    def __init__(self, lengths):
        self.bounding_box = lengths;
        self.list_of_owned_cells = 0;
        self.list_of_ghost_cells = 0;


    def repartition_grid(self):
        return 0;

    def request_ghost_cells(self):



