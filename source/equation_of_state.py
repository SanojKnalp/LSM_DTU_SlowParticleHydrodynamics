
# the equation of state used here
# We will be simulating water, so a reference density is set
# the density is the density of the respective particle
# the speed of sound is not an arbitrary choice and
# should be set w.r.t. some length scales

class EoS:
    def __init__(self, speed_of_sound, reference_density):
        self.speed_of_sound = speed_of_sound;
        self.reference_density = reference_density;

    # we should maybe pass the particle instead of the 
    # density value if we decide to store the pressure as a value of the
    # particle
    def evaluate_pressure(self,density):
        return (self.speed_of_sound**2)*(density - self.reference_density);

