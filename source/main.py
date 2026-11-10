generate_local_grid()
spawn_particles()

compute_densities();
compute_pressure();
compute_viscosity();

compute_acceleration();
first_kick();
drift();
move_particles();
# with new position
compute_densities();
compute_pressure();
compute_viscosity();

compute_acceleration();
second_kick()



compute_viscosity();
compute_acceleration();
first_kick();
drift();
# with new position
compute_densities();
compute_pressure();
compute_viscosity();
compute_acceleration();
second_kick();



