import numpy as np

# our base particle class
# The design of the class should not be changed
# as packing everything into one buffer makes it easy
# to send them with MPI
# We should consider also storing the density, viscosity and pressure (if it makes sense)
# maybe also the particle id or something?
class Particle:
    def __init__(self, pos, vel, radius):
        self.buffer = np.zeros(11, dtype=np.float64);
        self.position = self.buffer[0:1];
        self.position = pos;
        self.velocity = self.buffer[2:3];
        self.velocity = vel;
        self.acceleration = self.buffer[4:5];
        self.mass = self.buffer[6];
        self.density = self.buffer[7];
        self.pressure = self.buffer[8];
        self.viscosity = self.buffer[9]; # the viscosity is constant for 
        self.radius = self.buffer[10];
        self.radius = radius;

    def reset_dynamic_values(self):
        self.density = 0;
        self.pressure = 0;
        self.acceleration = np.array([0.0, 0.0]);
    def reset_acceleration(self):
        self.acceleration = np.array([0.0, 0.0]);