import Particle

# our time-integrator class 
# we implement an explicit kick and drift scheme here.
# the reason why it is a class and not a standalone function
# is because we only need to pass the time step-size once 
class VelocityVerlet:
    def __init__(self, dt, simulation_time):
        self.dt = dt;
        self.T = simulation_time;
        self.current_time = 0;

    # initial draft of our time-stepper. The acceleration part
    # needs to be reworked later when we have it in place
    # we should definitely pass a function instead.

    # we split them into three steps because we need to transfer particles
    # between cells after the second step

    # kick
    def first_kick(self, particle, acceleration):
        particle.velocity = particle.velocity + (self.dt/2)*acceleration;
    # drift
    def drift(self, particle):
        particle.position = particle.position + (self.dt)*particle.velocity;
    #kick
    def second_kick(self, particle, acceleration_next):
        particle.velocity = particle.velocity + (self.dt/2)*acceleration_next;
        self.current_time += self.dt;

    def reached_final_time(self):
        if (self.current_time < self.T):
            return False;
        return True;
        