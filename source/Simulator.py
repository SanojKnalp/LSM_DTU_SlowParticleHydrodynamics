from Evaluator import Particle2ParticleEvaluator
from time_integration import VelocityVerlet
from grid import Grid
from Particle import Particle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import vtk

class Simulator:
    def __init__(self, speed_of_sound, reference_density, smoothing_length, body_force, time_step_size, final_time, grid_lengths, subdivisions):
        self.p2p_eval = Particle2ParticleEvaluator(speed_of_sound, reference_density, smoothing_length, body_force);
        self.verlet = VelocityVerlet(time_step_size, final_time);
        self.grid = Grid(grid_lengths, subdivisions, smoothing_length);
        self.particle_radius = smoothing_length;

    # we need separate loops as we need to precompute the densities, because we need them a lot
    def cell_loop_density_and_pressure(self):
            for cell in self.grid.matrix_of_cells.flat:
                cell_and_neighbours = self.grid.get_neighbouring_cells(cell);
                for particle in cell.particles: # this needs to be adjusted as soon as I have the cell class
                    for local_cell in cell_and_neighbours.flat:
                        for local_particle in local_cell.particles: # adjust as well, because we need access to the particle list
                            self.p2p_eval.evaluate_density(particle1=particle, particle2=local_particle);
                    self.p2p_eval.eos.evaluate_pressure(particle=particle);
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
        self.verlet.advance_in_time();
    def spawn_particles(self):
        for i in range(50):
            pos1 = np.array([np.random.uniform(0, self.grid.bounding_box[0]), np.random.uniform(0, self.grid.bounding_box[1])]);
            vel = np.zeros(2);
            particle1 = Particle(pos1, vel, self.particle_radius);
            self.grid.insert_particle(particle1);

    def print_particle_position(self):
        # we loop over all the cells we own
        for cell in self.grid.matrix_of_cells.flat:
            # we loop over the particles in that cell
            print(cell.particles);
        i = 0;
        for cell in self.grid.matrix_of_cells.flat:
            for particle in cell.particles:
                print('pos',particle.position);
                i+=1;
        print(i);

    def write_points_to_vtk(self):
        vtk_points = vtk.vtkPoints();
        for cell in self.grid.matrix_of_cells.flat:
            for particle in cell.particles:
                vtk_points.InsertNextPoint(particle.position[0], particle.position[1], 0.0);
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)
        writer = vtk.vtkPolyDataWriter()
        file_name = f"positions_t={self.verlet.current_time}.vtk"
        writer.SetFileName(file_name);
        writer.SetInputData(polydata)
        writer.Write()


    def correct_boundary_and_moved(self):
        for cell in self.grid.matrix_of_cells.flat:
            for i in range(len(cell.particles)-1, 0, -1):
                print(cell.particles[i].position);
                cell.resolve_collision(cell.particles[i]);
                print(cell.particles[i].position);
                if (cell.particle_outside_cell(cell.particles[i])):
                    self.grid.move_particle_to_correct_cell(cell=cell, particle_index=i);


    def run(self):

        self.spawn_particles();
        
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
            #self.print_particle_position();
            # we do the second kick
            self.cell_loop_time_integration_second_kick();
            # now if we reached the final time-step, we break.
            #self.write_points_to_vtk();
            if(self.verlet.reached_final_time()):
                break;

            self.correct_boundary_and_moved();
            
            # we do not need to recompute the densities, as they have not changed. Same for pressure. 
            # This is because the position did not change. 
            # we only need to overwrite the acceleration
            self.cell_loop_acceleration();

        
sim = Simulator(speed_of_sound=1, reference_density=30, smoothing_length=0.2, body_force=np.array([0, -9.81]), time_step_size=1e-4, final_time=1e0, grid_lengths=np.array([5, 5]), subdivisions=np.array([4,4]));

sim.run();