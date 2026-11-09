import numpy as np

# even though there are multiple different kernels out there
# we just went with this one kernel as it is the easiest
# to use because the smoothing radius and our cell size
# have a 1:1 correlation without scaling

# this function implements the smoothing kernel.
# it's smoothing radius is by default set to 1.
# If we make it smaller, it will be less blurry
def smoothing_kernel(distance, smoothing_radius = 1.0):
    R = distance/smoothing_radius;
    normalization_constant = 105/(16*np.pi*smoothing_radius**3);
    return (1+3*R)*((1-R)**3)*normalization_constant;

# same thing here, but for the derivative
def smoothing_kernel_derivative(distance, smoothing_radius=1.0):
    R = distance/smoothing_radius;
    normalization_constant = 105/(16*np.pi*smoothing_radius**3);
    return -normalization_constant*(12*distance*(distance-smoothing_radius)**2)/(smoothing_radius**4);

