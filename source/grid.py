import numpy as np
from Particle import Particle
from Cell import Cell
from mpi4py import MPI

# our grid class.
# currently work in progress
class Grid:
    # length contains three entries:
    # length in x, length in y and cell length
    def __init__(self, lengths,  subdivisions):
        self.bounding_box = lengths;
        self.matrix_of_cells = np.empty((subdivisions[0], subdivisions[1]), dtype=object);
        self.bool_matrix_ghost = np.empty((subdivisions[0],subdivisions[1]), dtype=bool);
        self.cell_length = lengths[0]/subdivisions[0];
        self.max_ids = subdivisions-1;

        self.setup_cells();

    def get_neighbouring_cells(self, cell):
        id = cell.lower_left;
        if (id[0] == 0):    
            if (id[1] == 0):
                return self.matrix_of_cells[:2, :2]; # works
            elif (id[1] == self.max_ids[1]):
                return self.matrix_of_cells[0:2, (id[1]-1):]; # works
            else:
                return self.matrix_of_cells[0:2, (id[1]-1):(id[1]+2)]; # works
    
        elif (id[0] == self.max_ids[0]):
            if (id[1] == 0):
                return self.matrix_of_cells[(id[0]-1):, 0:2]; # works
            elif (id[1] == self.max_ids[1]):
                return self.matrix_of_cells[(id[0]-1):, (id[1]-1):]; # works
            else:
                return self.matrix_of_cells[(id[0]-1):, (id[1]-1):(id[1]+2)]; #works
    
        else:
            if(id[1] == 0):
                return self.matrix_of_cells[(id[0]-1):(id[0]+2), 0:2]; #works
            elif (id[1] == self.max_ids[1]):
                return self.matrix_of_cells[(id[0]-1):(id[0]+2), (id[1]-1):]; #works
            else:
                return self.matrix_of_cells[(id[0]-1):(id[0]+2), (id[1]-1):(id[1]+2)]; # works
    

    def request_ghost_cells(self):
        return 0;

    # tests if a cell is locally_owned
    # important for parallel handling of cells
    def is_locally_owned(self, cell):
        id = cell.lower_left;
        return self.bool_matrix_ghost[id[0], id[1]];


    def setup_cells(self):
        for i in range (self.matrix_of_cells.shape[0]):
            for j in range (self.matrix_of_cells.shape[1]):
                # we have not implemented any parallelization yet, so all are owned
                self.bool_matrix_ghost[i,j] = True;
                # we now compute the input quantities for the cells.
                self.matrix_of_cells[i,j] = Cell(np.array([i,j]) ,self.cell_length, self.bounding_box[0], self.bounding_box[1]);
