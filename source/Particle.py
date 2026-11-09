import numpy as np

# our base particle class
# The design of the class should not be changed
# as packing everything into one buffer makes it easy
# to send them with MPI
class Particle:
    def __init__(self, pos, vel):
        self.buffer = np.empty(6, dtype=np.float64);
        self.velocity = self.buffer[:3];
        self.velocity = vel;
        self.position = self.buffer[3:];
        self.position = pos;

    

