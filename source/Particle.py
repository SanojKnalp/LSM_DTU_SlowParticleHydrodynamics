import numpy as np

# our base particle class
# The design of the class should not be changed
# as packing everything into one buffer makes it easy
# to send them with MPI
# We should consider also storing the density, viscosity and pressure (if it makes sense)
# maybe also the particle id or something?
class Particle:
    def __init__(self, pos, vel, viscosity = 0, mass = 1):
        self.buffer = np.zeros(10, dtype=np.float64);
        self.position = self.buffer[:2];
        self.position = pos;
        self.velocity = self.buffer[2:4];
        self.velocity = vel;
        self.acceleration = self.buffer[4:6];
        self.mass = self.buffer[6];
        self.mass = mass;
        self.density = self.buffer[7];
        self.pressure = self.buffer[8];
        self.viscosity = self.buffer[9]; # the viscosity is constant for 
        self.viscosity = viscosity;

    def reset_dynamic_values(self):
        self.density = 0;
        self.pressure = 0;
        self.acceleration = np.array([0.0, 0.0]);
    def reset_acceleration(self):
        self.acceleration = np.array([0.0, 0.0]);
'''
pos = np.array([0.0, 1.0]);
vel = np.array([2.0, 3.0]);
particle = Particle(pos, vel);

print(particle.position);
print(particle.velocity);
print(particle.acceleration);
print(particle.mass);
print(particle.density);
print(particle.pressure);
print(particle.viscosity);
print(particle.buffer);
'''