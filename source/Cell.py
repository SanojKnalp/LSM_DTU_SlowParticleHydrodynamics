import numpy as np
from Particle import Particle
# cell class we will use to store our particles
# we need to know if it is a boundary cell or not
# to potentially resolve collisions as this is the cell's task
# Also WIP
class Cell:
    def __init__(self, lower_left, length, global_length_x, global_length_y, mpi_rank, particles = []):
        self.lower_left = lower_left; # we have cubed cells, so we will use the lower point and we have completely defined the cell

        self.cell_length = length;

        self.global_x = global_length_x
        self.global_y = global_length_y

        self.is_at_boundary = self.boundary_check();

        self.particles = particles;

        self.rank_it_owns = mpi_rank;

    def boundary_check(self):
        if self.lower_left[0] == 0 or self.lower_left[1] == 0:
            return True;
        elif self.lower_left[0]*self.cell_length + self.cell_length >= self.global_x or self.lower_left[1]*self.cell_length + self.cell_length >= self.global_y:
            return True;
        else:
            return False;

    def particle_outside_cell(self, particle):
        if ((particle.position[0]) < self.lower_left[0]*self.cell_length) or ((particle.position[0]) > (self.lower_left[0]+1)*self.cell_length):
            return True
        if ((particle.position[1]) < self.lower_left[1]*self.cell_length) or ((particle.position[1]) > (self.lower_left[1]+1)*self.cell_length):
            return True
        return False;

    # fine
    def resolve_collision(self, particle):
        if self.is_at_boundary:
            if ((particle.position[0]-particle.radius) < 0):
                particle.velocity[0] = -particle.velocity[0]*0.99; # I wanted a bit of damping
                particle.position[0] = particle.radius;
            elif ((particle.position[0] + particle.radius) >= self.global_x):
                particle.velocity[0] = -particle.velocity[0]*0.99;
                particle.position[0] = self.global_x - particle.radius;
            if ((particle.position[1]-particle.radius) <= 0):
                particle.velocity[1] = -particle.velocity[1]*0.99;
                particle.position[1] = particle.radius;
            elif ((particle.position[1] + particle.radius) >= self.global_y):
                particle.velocity[1] = -particle.velocity[1]*0.99;
                particle.position[1] = self.global_y - particle.radius;

    
    def get_global_index(self):
        return self.lower_left;

    def pack_into_buffer(self):
        return np.array([self.lower_left, self.cell_length, self.global_x, self.global_y, self.particles, self.rank_it_owns], dtype=object);
