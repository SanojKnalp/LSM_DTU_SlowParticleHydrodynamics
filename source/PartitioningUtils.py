import numpy as np
import math
from mpi4py import MPI

# this function computes the initial grid partitioning for rectangles
def create_initial_partitioning_rectangles(rank, size, number_of_cells_x, number_of_cells_y):
    # since we want our pieces to be as equal as possible, we need to com
    grid_aspect_ratio = number_of_cells_x/number_of_cells_y;

    # using this grid aspect ratio we can now compute the square metric
    # this works by taking the square root of the size of the communicator 
    # multiplied by the aspect ratio and then rounding UP
    # s_metric is then the number of columns we are going to have
    s_metric = math.ceil(np.sqrt(size*grid_aspect_ratio));

    # we obtain the correctly splitted grid in the following notation [first_index0 last_index first_index1 last_index1 ...]
    x_splitted = split_line_grid_into_n_pieces(number_of_parts=s_metric, number_of_cells_in_direction=number_of_cells_x);

    # we compute how many ranks we still need to somehow get rid of
    remaining_ranks = size - s_metric;
    # compute some metrics
    y_split_min = int(np.floor(remaining_ranks/s_metric));
    y_split_max = math.ceil(remaining_ranks/s_metric);
    num_more_splits = np.mod(remaining_ranks, s_metric);
    distributed_splits = distribute_splits(y_split_min=y_split_min, y_split_max=y_split_max, num_more_splits=num_more_splits, num_columns=s_metric);


    y_splits = split_y(number_of_cells_y, distributed_splits);

    return compute_locally_owned_cells_from_rank_and_splits(rank, x_splitted, y_splits, s_metric);



# this function computes the correct splitting. 
# reminder: locally_owned_cells[0] = [0,1,2,3]  and locally_owned_cells=[2,3,4]; for example
def compute_locally_owned_cells_from_rank_and_splits(rank, x_splitted, y_splitted, number_of_columns):
    
    number_of_cells_already_distributed = 0;
    for i in range(number_of_columns):
        second_loop_bound = int(len(y_splitted[i])/2);
        if (number_of_cells_already_distributed + (second_loop_bound-1) < rank):
            number_of_cells_already_distributed+=second_loop_bound;
            continue;
        else:
            start_index_x = x_splitted[2*i];
            end_index_x = x_splitted[2*i+1];
            x_indices = np.linspace(start_index_x, end_index_x, end_index_x-start_index_x+1).astype(int);

            y_index = rank - number_of_cells_already_distributed;
            start_index_y = y_splitted[i][2*y_index];
            end_index_y = y_splitted[i][2*y_index+1];
            y_indices = np.linspace(start_index_y, end_index_y, end_index_y-start_index_y+1).astype(int);
            locally_owned_cells = [x_indices, y_indices];
            break;


    return locally_owned_cells;


def split_y(number_of_cells_y, uneven_splits):

    splits = [];

    for i in range(len(uneven_splits)):
        y_split_i = split_line_grid_into_n_pieces(uneven_splits[i], number_of_cells_y);
        splits.append(y_split_i);

    return splits;


# this function computes how many rectangles each column has
def distribute_splits(y_split_min, y_split_max, num_more_splits, num_columns):
    split = np.ones(num_columns, dtype=np.int32);

    split += y_split_max;

    num_less_splits = num_columns - num_more_splits;
    halfed_less_splits = int(np.floor(num_less_splits/2));
    for i in range(halfed_less_splits):
        split[i] = y_split_min+1;
        split[-1-i] = y_split_min+1;

    if (np.mod(num_less_splits, 2) != 0):
        split[halfed_less_splits] = y_split_min;

    return split;


# this function here now takes the number of parts and then computes an array with the respective start and end indices packed together
def split_line_grid_into_n_pieces(number_of_parts, number_of_cells_in_direction):
    split = np.zeros(number_of_parts*2, dtype=np.int32);

    perfect_division = int(np.floor(number_of_cells_in_direction/number_of_parts));

    remainder = number_of_cells_in_direction-perfect_division*number_of_parts;

    if (remainder == 0):
        current_index = 0;
        for i in range(number_of_parts):
            split[2*i] = current_index;
            split[2*i+1] = current_index + (perfect_division-1);
            current_index+= perfect_division;

    elif (remainder != 0):
        extensions = np.zeros(number_of_parts, dtype=np.int32);
        splitted_remainder = int(np.floor(remainder/2));

        for i in range(splitted_remainder):
            extensions[i] += 1;
            extensions[-1-i] +=1;

        if (np.mod(remainder,2) != 0):
            extensions[splitted_remainder] +=1;
            
        current_index = 0;

        for j in range(number_of_parts):
            if (j == 0):
                split[0] = 0;
            else:
                split[2*j] = current_index;

            split[2*j+1] = current_index + (perfect_division-1) + extensions[j];

            current_index+=perfect_division + extensions[j];


    
    return split;







    

print(create_initial_partitioning_rectangles(rank=11, size=22, number_of_cells_x=12, number_of_cells_y=8));


