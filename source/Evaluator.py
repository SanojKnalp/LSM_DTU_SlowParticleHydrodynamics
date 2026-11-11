import numpy as np
from Particle import Particle
from equation_of_state import EoS
from SmoothingKernel import SmoothingKernel


class Particle2ParticleEvaluator:
    def __init__(self, speed_of_sound, reference_density, smoothing_kernel, body_force):
        self.eos = EoS(speed_of_sound, reference_density);
        self.kernel = SmoothingKernel(smoothing_kernel);
        self.b = body_force;

    def evaluate_density_and_pressure(self, particle1, particle2):
        dist = np.linalg.norm(particle1.position - particle2.position);
        particle1.density += self.kernel.evaluate(distance=dist)*particle1.mass;
        self.eos.evaluate_pressure(particle1);


    def evaluate_p_tilde(self, particle1, particle2):
        return ((particle2.density)*(particle1.pressure) + (particle1.density)*(particle2.pressure))/(particle1.density + particle2.density);

    def evaluate_eta_tilde(self, particle1, particle2):
        return 2*particle1.viscosity*particle2.viscosity/(particle1.viscosity + particle2.viscosity);

    def evaluate_volume_term(self, particle1, particle2):
        return (particle1.mass/particle1.density)**2 + (particle2.mass/particle2.density)**2;

    def evaluate_acceleration_term_pressure(self, particle1, particle2):
        if(particle1.position == particle2.position):
            return;
        p_tilde = self.evaluate_p_tilde(particle1, particle2);
        r_ij = particle1.position - particle2.position;
        r_ij_norm = np.linalg.norm(r_ij);
        e_ij = r_ij/r_ij_norm;
        W_ij_r_ij = self.kernel.evaluate_gradient(r_ij_norm);
        volume_term = self.evaluate_volume_term(particle1, particle2);
        particle1.acceleration +=  (1/(particle1.mass))*volume_term*(- p_tilde*W_ij_r_ij*e_ij );

    def evaluate_acceleration_term_viscosity(self, particle1, particle2):
        eta_tilde = self.evaluate_eta_tilde(particle1, particle2);
        volume_term = self.evaluate_volume_term(particle1, particle2);
        r_ij = particle1.position - particle2.position;
        r_ij_norm = np.linalg.norm(r_ij);
        u_ij = particle1.velocity - particle2.velocity;
        W_ij_r_ij = self.kernel.evaluate_gradient(r_ij_norm);
        particle1.acceleration += (1/(particle1.mass))*volume_term*((eta_tilde/(r_ij_norm))*W_ij_r_ij*u_ij);

    def evaluate_acceleration_term_body_force(self, particle1):
        particle1.acceleration += self.b;