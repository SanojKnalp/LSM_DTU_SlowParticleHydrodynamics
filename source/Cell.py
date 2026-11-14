import numpy as np
import Particle
# cell class we will use to store our particles
# we need to know if it is a boundary cell or not
# to potentially resolve collisions as this is the cell's task
# Also WIP
class Cell:
    def __init__(self, lower_left, length, global_length_x, global_length_y):
        # 0:1 : x- and y-index of the cell
        # 2: length of cell (cubed, so only one necessary)
        # 3:4 : global length in x and y
        # 5: precomputed identifier if cell is at boundary
        # 6: array_of_particles
        self.buffer = np.empty(7, dtype=object); 

        self.lower_left = self.buffer[0:1];
        self.lower_left = lower_left; # we have cubed cells, so we will use the lower point and we have completely defined the cell

        self.cell_length = self.buffer[2];
        self.cell_length = length;

        self.global_x = self.buffer[3];
        self.global_y = self.buffer[4];
        self.global_x = global_length_x
        self.global_y = global_length_y

        self.is_at_boundary = self.buffer[5];
        self.is_at_boundary = self.boundary_check();

        self.particles = self.buffer[6];
        self.particles = []; 

    def boundary_check(self):
        if self.lower_left[0] == 0 or self.lower_left[1] == 0:
            return True;
        elif self.lower_left[0]*self.cell_length + self.cell_length >= self.global_x or self.lower_left[1]*self.cell_length + self.cell_length >= self.global_y:
            return True;
        else:
            return False;

    # fine
    def resolve_collision(self, particle):
        if self.is_at_boundary:
            if ((particle.position[0]-particle.radius) <= 0) or ((particle.position[0] + particle.radius) >= (self.lower_left[0]+1)*self.cell_length):
                particle.velocity[0] = -particle.velocity[0]
            if ((particle.position[1]-particle.radius) <= 0) or ((particle.position[1] + particle.radius) >= (self.lower_left[1]+1)*self.cell_length):
                particle.velocity[1] = -particle.velocity[1]

    # if a particle has moved outside of the cell domain,
    # we let grid know and the grid takes care of the 
    # moving process. The details of the function are up for debate
    def tell_grid_particle_has_moved():
        return 0;

    # add particle to the the array of particles
    def add_particle(self):
        return 0;
    def get_global_index(self):
        return self.lower_left;
