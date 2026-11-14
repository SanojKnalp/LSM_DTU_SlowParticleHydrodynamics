from Evaluator import Particle2ParticleEvaluator
from time_integration import VelocityVerlet
from grid import Grid
from Particle import Particle
import numpy as np

class Simulator:
    def __init__(self, speed_of_sound, reference_density, smoothing_length, body_force, time_step_size, final_time, grid_lengths, subdivisions):
        self.p2p_eval = Particle2ParticleEvaluator(speed_of_sound, reference_density, smoothing_length, body_force);
        self.verlet = VelocityVerlet(time_step_size, final_time);
        self.grid = Grid(grid_lengths, subdivisions);
        self.particle_radius = smoothing_length;

    # we need separate loops as we need to precompute the densities, because we need them a lot
    def cell_loop_density_and_pressure(self):
            for cell in self.grid.matrix_of_cells.flat:
                cell_and_neighbours = self.grid.get_neighbouring_cells(cell);
                for particle in cell.particles: # this needs to be adjusted as soon as I have the cell class
                    for local_cell in cell_and_neighbours.flat:
                        for local_particle in local_cell.particles: # adjust as well, because we need access to the particle list
                            self.p2p_eval.evaluate_density_and_pressure(particle1=particle, particle2=local_particle);
    
    # the next is to evaluate the acceleration
    def cell_loop_acceleration(self):
        # we loop over all the cells we own
        for cell in self.grid.matrix_of_cells.flat:
            if (self.grid.is_locally_owned(cell)):
                # we extract the cell and its neighbours
                cell_and_neighbours = self.grid.get_neighbouring_cells(cell);
                # for every particle, we then loop over only the cell itself 
                # and the neighbouring cells
                for local_cell in cell_and_neighbours.flat: 
                    for particle in cell.particles:
                        # then we get the particles of the neighbouring cell 
                        for local_particle in local_cell.particles:
                            # now we can evaluate the pressure and viscosity contributions from each particle.
                            self.p2p_eval.evaluate_acceleration_term_pressure(particle1=particle, particle2=local_particle);
                            self.p2p_eval.evaluate_acceleration_term_viscosity(particle1=particle, particle2=local_particle);
                        # the last step is to add the body force term to that specific particle
                        self.p2p_eval.evaluate_acceleration_term_body_force(particle1=particle);
    
    # we do a separate time-loop over the cells. 
    # we cannot do it in cell_loop_acceleration, as it would overwrite the position
    # and we need them at the exact same time-step 
    def cell_loop_time_integration_first_kick_and_drift(self):
        # we loop over all the cells we own
        for cell in self.grid.matrix_of_cells.flat:
            # we loop over the particles in that cell
            for particle in cell.particles:
                # we execute the first kick and drift
                self.verlet.first_kick(particle=particle);
                self.verlet.drift(particle=particle);
                # now because the positions have changed, we need to reset the pressure,
                # density and acceleration, otherwise we would always add more and more
                # acceleration until we blow up
                particle.reset_dynamic_values();

    def cell_loop_time_integration_second_kick(self):
        # we loop over all the cells we own
        for cell in self.grid.matrix_of_cells.flat:
            # we loop over the particles in that cell
            for particle in cell.particles:
                self.verlet.second_kick(particle=particle);
                particle.reset_acceleration();

    def spawn_particles(self):
        dx = self.grid.bounding_box[0]/11;
        for i in range(5):
            pos = np.array([0.1+dx*i, 1]);
            vel = np.zeros(2);
            particle = Particle(pos, vel);
            self.grid.insert_particle(particle);

    def print_particle_position(self):
        # we loop over all the cells we own
        for cell in self.grid.matrix_of_cells.flat:
            # we loop over the particles in that cell
            print(cell.particles);
        for cell in self.grid.matrix_of_cells.flat:
            for particle in cell.particles:
                print(particle.position);
    def run(self):

        self.spawn_particles();
        self.print_particle_position();
        self.cell_loop_density_and_pressure();
        self.cell_loop_acceleration();
        
        while (True):
            self.cell_loop_time_integration_first_kick_and_drift();
            # position and velocity have changed, so we need to run the full cell loop
            # and recompute basically everything
            # we recompute the density and then immediately the pressure
            self.cell_loop_density_and_pressure();
            # now the acceleration
            self.cell_loop_acceleration();
            # we do the second kick
            self.cell_loop_time_integration_second_kick();
            # now if we reached the final time-step, we break.
            if(self.verlet.reached_final_time()):
                break;
            # we do not need to recompute the densities, as they have not changed. Same for pressure. 
            # This is because the position did not change. 
            # we only need to overwrite the acceleration
            self.cell_loop_acceleration();

        self.print_particle_position();

        
sim = Simulator(speed_of_sound=1.0, reference_density=1, smoothing_length=0.01, body_force=np.zeros(2), time_step_size=1e-4, final_time=1, grid_lengths=np.array([5, 5]), subdivisions=np.array([5,5]));

sim.run();