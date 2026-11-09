
# cell class we will use to store our particles
# we need to know if it is a boundary cell or not
# to potentially resolve collisions as this is the cell's task
# Also WIP
class Cell:
    def __init__(self, lower_left, length):
        self.lower_left = lower_left; # we have cubed cells, so we will use the lower point and we have completely defined the cell
        self.length     = length;
        self.is_at_boundary = 0;
        self.array_of_particles = 0; # this one will store our particles

    def resolve_collision(self, particle):
        return 0;

    # if a particle has moved outside of the cell domain,
    # we let grid know and the grid takes care of the 
    # moving process. The details of the function are up for debate
    def tell_grid_particle_has_moved():
        return 0;

    # add particle to the the array of particles
    def add_particle(self):
        return 0;



