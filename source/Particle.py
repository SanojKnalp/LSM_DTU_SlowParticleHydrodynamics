import numpy as np

# our base particle class
# The design of the class should not be changed
# as packing everything into one buffer makes it easy
# to send them with MPI
# We should consider also storing the density, viscosity and pressure (if it makes sense)
# maybe also the particle id or something?
class Particle:
    def __init__(self, pos, vel):
        self.buffer = np.zeros(8, dtype=np.float64);
        self.velocity = self.buffer[0:1];
        self.velocity = vel;
        self.position = self.buffer[2:3];
        self.position = pos;
        self.mass = self.buffer[4];
        self.density = self.buffer[5];
        self.pressure = self.buffer[6];
        self.viscosity = self.buffer[7]; # the viscosity is constant for 

    def reset_dynamic_values(self):
        self.density = 0;
        self.pressure = 0;