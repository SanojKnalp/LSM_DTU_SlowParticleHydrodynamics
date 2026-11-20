import numpy as np
from Particle import Particle
from Cell import Cell
from mpi4py import MPI

# our grid class.
# currently work in progress
class Grid:
    # length contains three entries:
    # length in x, length in y and cell length
    def __init__(self, lengths,  subdivisions, communicator, locally_owned_global_indices):
        self.communicator = communicator;
        self.global_max_ids = subdivisions-1; 
        self.bounding_box = lengths;
        self.local_owned_indices = locally_owned_global_indices;
        self.cell_length = lengths[0]/subdivisions[0];

        # track the number of extensions into every direction
        self.left_extend = False;
        self.right_extend = False;
        self.top_extend = False;
        self.bottom_extend = False;
        self.x_extensions = self.x_extend();
        
        


        self.matrix_of_cells = np.empty((len(self.local_owned_indices[0])+self.x_extensions, len(self.local_owned_indices[1])), dtype=object);
        self.bool_matrix_ghost = np.empty((len(self.local_owned_indices[0])+self.x_extensions, len(self.local_owned_indices[1])), dtype=bool);

        # we want to store the particles we send to each rank into one buffer.
        # then we also want to pack them up into a buffer
        # the first index is the rank, the second index is the particle buffer
        self.particles_to_send = np.empty((self.communicator.Get_size(), 2));
        

        self.setup_owned_cells();

    #adjusted to account for the local matrices
    def get_neighbouring_cells(self, cell):
        id = cell.lower_left;
        if (id[0] == 0):    
            if (id[1] == 0):
                return self.matrix_of_cells[:2, :2]; # works
            elif (id[1] == self.global_max_ids[1]):
                return self.matrix_of_cells[0:2, (self.transform_global_index_to_local_index_y(id[1])-1):]; # works
            else:
                return self.matrix_of_cells[0:2, (self.transform_global_index_to_local_index_y(id[1])-1):(self.transform_global_index_to_local_index_y(id[1])+2)]; # works
    
        elif (id[0] == self.global_max_ids[0]):
            if (id[1] == 0):
                return self.matrix_of_cells[(self.transform_global_index_to_local_index_x(id[0])-1):, 0:2]; # works
            elif (id[1] == self.global_max_ids[1]):
                return self.matrix_of_cells[(self.transform_global_index_to_local_index_x(id[0])-1):, (self.transform_global_index_to_local_index_x(id[0])-1):]; # works
            else:
                return self.matrix_of_cells[(self.transform_global_index_to_local_index_x(id[0])-1):, (self.transform_global_index_to_local_index_y(id[1])-1):(self.transform_global_index_to_local_index_y(id[1])+2)]; #works
    
        else:
            if(id[1] == 0):
                return self.matrix_of_cells[(self.transform_global_index_to_local_index_x(id[0])-1):(self.transform_global_index_to_local_index_x(id[0])+2), 0:2]; #works
            elif (id[1] == self.global_max_ids[1]):
                return self.matrix_of_cells[(self.transform_global_index_to_local_index_x(id[0])-1):(self.transform_global_index_to_local_index_x(id[0])+2), (self.transform_global_index_to_local_index_y(id[1])-1):]; #works
            else:
                return self.matrix_of_cells[(self.transform_global_index_to_local_index_x(id[0])-1):(self.transform_global_index_to_local_index_x(id[0])+2), (self.transform_global_index_to_local_index_y(id[1])-1):(self.transform_global_index_to_local_index_y(id[1])+2)]; # works
    

    # this way pf requesting ghost cells does only work if we partition with slabs, so caution.
    # Explanation: We know that we store the entries from left to right w.r.t. the mpi rank, so we can do that.
    def send_and_request_ghost_cells_slabs(self):
        # if we extend to the left, we simply get the ghost cells from rank n-1 if we are rank n.
        # The same goes for the sending part. We always send first with non-blocking communication and then also receive with non-blocking communication
        # so first, we need to know which rank we are.
        rank = self.communicator.Get_rank();

        # two tags we define to 
        send_to_left_tag = 5;
        send_to_right_tag = 6;

        requests = [];

        if (self.left_extend):
            # first, we need to copy all the cells and pack them into a buffer.
            # this is what we do here. We first create a numby array we want to fill
            left_cells_to_send = np.empty(self.global_max_ids[1]+1, dtype=object);
            # then we fill the array
            # we fill it with the first index, because the zeroth index is the ghost layer
            for i in range(self.global_max_ids[1]+1):
                left_cells_to_send[i] = self.matrix_of_cells[1, i].pack_into_buffer();


            # unfortunately, we kinda have to use the small letter send, but it's better than unpacking everything and reformatting.
            # we also do non-blocking communication so that we can go straight into the receiving phase
            self.communicator.isend(left_cells_to_send, dest=rank-1, tag = send_to_left_tag);


            left_cells_to_receive = np.empty(self.global_max_ids[1]+1, dtype=object);
            # now we straight receive the other cells. Attention: these cells are received from the right
            requests.append(self.communicator.irecv(left_cells_to_receive, source = rank-1, tag = send_to_right_tag));

        # now everything here is basically the same as for the left one
        if (self.right_extend):
            right_cells_to_send = np.empty(self.global_max_ids[1]+1, dtype=object);
            # we have to found out the length of the matrix, so the number of columns
            max_right_index = self.matrix_of_cells.shape[0];
            for i in range(self.global_max_ids[1]+1):
                # we reduce the index by 2, one for simple 0-indexing and the second one for the 
                # ghost layer
                right_cells_to_send[i] = self.matrix_of_cells[max_right_index-2, i].pack_into_buffer();
            
            self.communicator.isend(right_cells_to_send, dest=rank+1, tag = send_to_right_tag);

            right_cells_to_receive = np.empty(self.global_max_ids[1]+1, dtype=object);
            requests.append(self.communicator.irecv(right_cells_to_receive, source = rank+1, tag = send_to_left_tag));

        # now we wait until we received all the ghost cell information
        MPI.Request.Waitall(requests);

        if(self.left_extend):
            # we loop over all the cells
            for i in range (len(left_cells_to_receive)):
                # then we pick the cell
                cell = left_cells_to_receive[i];
                # and we create the new cell.
                self.matrix_of_cells[0, i] = Cell(lower_left=cell[0], length=cell[1], global_length_x=cell[2], global_length_y=cell[3], mpi_rank=cell[5], particles=cell[4]);

        if (self.right_extend):
            for i in range(len(right_cells_to_receive)):
                cell = right_cells_to_receive[i];

                self.matrix_of_cells[-1, i] = Cell(lower_left=cell[0], length=cell[1], global_length_x=cell[2], global_length_y=cell[3], mpi_rank=cell[5], particles=cell[4]);



    # adjust indices
    def insert_particle(self, particle):
        x_index = int(np.floor(particle.position[0]/self.cell_length));
        y_index = int(np.floor(particle.position[1]/self.cell_length));
        if (not self.bool_matrix_ghost[x_index, y_index]):
            self.pack_up_particle(particle=particle, rank_t0_send_to=self.matrix_of_cells[x_index, y_index].rank_it_owns);
        self.matrix_of_cells[x_index, y_index].particles.append(particle);

    def move_particle_to_correct_cell(self, cell, particle_index):
        # we create an instance of the moved particle
        moved_particle = cell.particles[particle_index];
        # we delete the old particle
        del cell.particles[particle_index];
        # now we figure out the new position
        self.insert_particle(moved_particle);

    # Look at self.particles_to_send and then store it here
    # then do one giant send
    def pack_up_particle(self, particle, rank_to_send_to):
        self.particles_to_send[rank_to_send_to, ]
        
    def send_particles(self):
        return 0;

    # tests if a cell is locally_owned
    # important for parallel handling of cells
    def is_locally_owned(self, cell):
        id = cell.lower_left;
        return self.bool_matrix_ghost[self.transform_global_index_to_local_index_x(id[0]), self.transform_global_index_to_local_index_y(id[1])];


    def setup_owned_cells(self):
        for i in range (self.matrix_of_cells.shape[0]):
            for j in range (self.matrix_of_cells.shape[1]):
                # we have not implemented any parallelization yet, so all are owned
                self.bool_matrix_ghost[i,j] = True;
                if( i == 0 and self.left_extend):
                    self.bool_matrix_ghost[i,j]= False;
                elif (i == self.matrix_of_cells.shape[0]-1 and self.right_extend):
                    self.bool_matrix_ghost[i,j] = False;
                elif (j == 0 and self.bottom_extend):
                    self.bool_matrix_ghost[i,j] = False;
                elif (j == self.matrix_of_cells.shape[0]-1 and self.top_extend):
                    self.bool_matrix_ghost[i,j] = False;
                
                if (self.bool_matrix_ghost[i,j]):
                # we now compute the input quantities for the cells.
                # this needs to be a global index for the cell though
                    self.matrix_of_cells[i,j] = Cell(np.array([self.transform_local_index_to_global_index_x(i),self.transform_local_index_to_global_index_y(j)]) ,self.cell_length, self.bounding_box[0], self.bounding_box[1], self.communicator.Get_rank());


    def x_extend(self):
        extensions = 0;
        if self.local_owned_indices[0,0] != 0:
            extensions+=1;
            self.left_extend = True;
        if self.local_owned_indices[0,1] != self.global_max_ids[0]:
            extensions+=1;
            self.right_extend = True;
        return extensions;

    def y_extend(self):
        extensions = 0;
        if self.local_owned_indices[1,0] != 0:
            extensions+=1;
            self.bottom_extend = True;
        if self.local_owned_indices[1,1] != self.global_max_ids[1]:
            extensions+=1;
            self.top_extend = True;
        return extensions;

    # These account now properly for ghost layers
    def transform_local_index_to_global_index_x(self, local_x_index):
        val = self.local_owned_indices[0, 0] + local_x_index;
        if (self.left_extend):
            val -=1;
        return val;

    def transform_local_index_to_global_index_y(self, local_y_index):
        val =  self.local_owned_indices[1, 0] + local_y_index;
        if (self.bottom_extend):
            val -= 1;
        return val;

    def transform_global_index_to_local_index_x(self, global_x_index):
        val = global_x_index-self.local_owned_indices[0,0];
        if (self.left_extend):
            val +=1;
        return val;

    def transform_global_index_to_local_index_y(self, global_y_index):
        val = global_y_index-self.local_owned_indices[1,0];
        if (self.bottom_extend):
            val +=1;
        return val;