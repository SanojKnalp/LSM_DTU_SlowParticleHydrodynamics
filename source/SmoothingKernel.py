import numpy as np

# even though there are multiple different kernels out there
# we just went with this one kernel as it is the easiest
# to use because the smoothing radius and our cell size
# have a 1:1 correlation without scaling

# this function implements the smoothing kernel.
# it's smoothing radius is by default set to 1.
# If we make it smaller, it will be less blurry

class SmoothingKernel:
    def __init__(self, smoothing_radius):
        self.smoothing_radius = smoothing_radius;
        self.normalization_constant = 5/(np.pi*((self.smoothing_radius)**2))
    def evaluate(self, distance):
        R = distance/self.smoothing_radius;
        return (1+3*R)*((1-R)**3)*self.normalization_constant;

    # same thing here, but for the derivative
    def evaluate_gradient(self, distance):
        return -self.normalization_constant*(12*distance*(distance-self.smoothing_radius)**2)/(self.smoothing_radius**4);

