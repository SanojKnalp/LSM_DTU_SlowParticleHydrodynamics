import numpy as np

# our base particle class
# The design of the class should not be changed
# as packing everything into one buffer makes it easy
# to send them with MPI
# We should consider also storing the density, viscosity and pressure (if it makes sense)
# maybe also the particle id or something?
class Particle:
    def __init__(self, pos, vel):
        self.buffer = np.empty(6, dtype=np.float64);
        self.velocity = self.buffer[:3];
        self.velocity = vel;
        self.position = self.buffer[3:];
        self.position = pos;

    

