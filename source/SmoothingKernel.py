import numpy as np

# even though there are multiple different kernels out there
# we just went with this one kernel as it is the easiest
# to use because the smoothing radius and our cell size
# have a 1:1 correlation without scaling

# this function implements the smoothing kernel.
# class with functions to implement the gradient

class SmoothingKernel:
    def __init__(self, smoothing_radius):
        self.smoothing_radius = smoothing_radius;
        self.normalization_constant = 105/(16*np.pi*((self.smoothing_radius)**3))
    def evaluate(self, distance):
        R = distance/self.smoothing_radius;
        return (1+3*R)*((1-R)**3)*self.normalization_constant;

    # same thing here, but for the derivative
    def evaluate_gradient(self, distance):
        R = distance/self.smoothing_radius;
        return -self.normalization_constant*(12*distance*(R)**2)/(self.smoothing_radius**4);

