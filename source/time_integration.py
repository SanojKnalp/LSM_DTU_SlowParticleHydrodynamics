import Particle

# our time-integrator class 
# we implement an explicit kick and drift scheme here.
# the reason why it is a class and not a standalone function
# is because we only need to pass the time step-size once 
class VelocityVerlet:
    def __init__(self, dt):
        self.dt = dt;

    # initial draft of our time-stepper. The acceleration part
    # needs to be reworked later when we have it in place
    # we should definitely pass a function instead
    def advance_time_step(self, particle, acceleration, acceleration_next):
        # kick
        particle.velocity = particle.velocity + (self.dt/2)*acceleration;
        # drift
        particle.position = particle.position + (self.dt)*particle.velocity;
        #kick
        particle.velocity = particle.velocity + (self.dt/2)*acceleration_next;
        