import numpy as np
import heapq
import matplotlib.pyplot as plt

# main A* algorithm
def a_star(grid: np.ndarray, start_pos: tuple[int, int], goal_pos: tuple[int, int]) -> list[tuple[int,int]]:
    '''
    Use the A* path planning algorithm to return a list of nodes given
    an occupancy grid of obstacles, a start node, and a goal node.

    Keyword arguments:
    grid -- 2D numpy array occupancy grid, 0 = free 1 = obstacle; note (row, col) = (y, x)
    start -- integer pair coordinates (x,y) of start node on occupancy grid
    goal -- integer pair coordinates (x,y) of goal node on occupancy grid

    Returns:
    Ordered list of node coordinates (x,y) on computed path from start to goal
    '''

    # create start node
    start_node = create_node(
        position = start_pos,
        g = 0.0,
        h = euclidean_distance(start_pos, goal_pos)
    )

    # initialize lists
    open_list = [(start_node['f'], start_pos)] # priority queue of nodes to explore - positions ordered by estimated cost
    open_dict = {start_pos: start_node} # dict of full node information, key is position
    closed_set = set() # set of positions already reached, starts empty

    # main loop
    # while there are nodes to explore, check the one with the lowest estimated cost
    while open_list:
        # use heapq to pop lowest cost node from list
        _, current_pos = heapq.heappop(open_list)
        current_node = open_dict[current_pos]

        # check for goal
        if current_pos == goal_pos:
            return construct_path(current_node)

        # mark node as reached
        closed_set.add(current_pos)

        # explore neighbors
        for neighbor_pos in get_neighbors(grid=grid, position=current_pos):
            # skip if already reached
            if neighbor_pos in closed_set:
                continue

            # calculate path cost 
            g_new = current_node['g'] + euclidean_distance(current_pos, neighbor_pos)

            # create or update adjacent node
            if neighbor_pos not in open_dict:
                # node not in queue, create it
                neighbor_node = create_node(
                    position = neighbor_pos,
                    g = g_new,
                    h = euclidean_distance(neighbor_pos, goal_pos),
                    parent = current_node
                )

                # add to queue
                heapq.heappush(open_list, (neighbor_node['f'], neighbor_pos))
                open_dict[neighbor_pos] = neighbor_node

            elif g_new < open_dict[neighbor_pos]['g']:
                # better path to existing node, update path and total cost
                neighbor_node = open_dict[neighbor_pos]
                neighbor_node['g'] = g_new
                neighbor_node['f'] = g_new + neighbor_node['h']
                neighbor_node['parent'] = current_node

    # if no path is found, return empty list
    return []

# helpers
def create_node(position: tuple[int, int], g=float('inf'), h=0.0, parent=None) -> dict:
    '''
    Create a node with the given properties

    Keyword arguments:
    position -- integer pair coordinates (x,y) of node position on occupancy grid
    g -- path cost from start; defaults to infinity
    h -- Euclidean distance to goal heuristic; defaults to zero
    parent -- parent node on exploration tree; defaults to none

    Returns:
    Dictionary of node properties
    '''
    return {
        'position': position,
        'g': g,
        'h': h,
        'f': g+h,
        'parent': parent
    }

def euclidean_distance(pos_1: tuple[int, int], pos_2: tuple[int, int]) -> float:
    '''
    Calculate euclidean distance between two 2D coordinates
    '''

    x1, y1 = pos_1
    x2, y2 = pos_2
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def construct_path(goal_node: dict):
    '''
    Returns path to goal_node from start as list of positions (x,y)
    '''

    # path starts empty, tracing back from goal node
    path = []
    current_node = goal_node

    # iterate through parents until there aren't any
    while current_node is not None:
        path.append(current_node['position'])
        current_node = current_node['parent']

    # path is from goal to start, so reverse and return
    return path[::-1]

def get_neighbors(grid: np.ndarray, position: tuple[int, int]) -> list[tuple[int, int]]:
    '''
    Checks occupancy grid and returns unoccupied, unreached adjacent nodes

    Keyword Arguments:
    grid -- grid -- 2D numpy array occupancy grid, 0 = free 1 = obstacle; note (row, col) = (y, x)
    position -- integer pair coordinates (x,y) of node to extend

    Returns:
    List of integer pair coordinates (x,y) of valid neighbors
    '''

    x, y = position
    rows, cols = grid.shape

    # all possible adjacent positions
    adj_pos = [
        (x+1, y), (x-1, y), # E, W
        (x, y+1), (x, y-1), # N, S
        (x+1, y+1), (x-1, y+1), # NE, NW
        (x+1, y-1), (x-1, y-1), # SE, SW
    ]

    return [
        (nx, ny) for nx, ny in adj_pos
        if 0 <=nx < cols and 0 <= ny < rows # within grid
        and grid[ny, nx] == 0 # location unoccupied
    ]

def show_path(grid: np.ndarray, path: list[tuple[int, int]]):
    '''
    Plot occupancy grid of obstacles and planned path, with start and goal positiions marked
    note grid (row, col) = (y, x)
    '''

    # create figure and show grid
    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap='binary', origin='lower')

    # plot path and markers
    if path:
        path_arr = np.array(path)
        plt.plot(path_arr[:,0], path_arr[:,1], 'b-', linewidth=3, label='Path')
        plt.plot(path_arr[0,0], path_arr[0,1], 'go', markersize=15, label='Start')
        plt.plot(path_arr[-1,0], path_arr[-1,1], 'ro', markersize=15, label='Goal')

    plt.grid(True)
    plt.legend(fontsize=12)
    plt.title('A* Path Planning')
    plt.show()

# test and visualize
# create empty grid
grid = np.zeros((20,30))

# add obstacles
grid[10, 5:25] = 1 # E-W wall

# start and goal positions
start_pos = (4, 2)
goal_pos = (28, 18)

# find path
path = a_star(grid=grid, start_pos=start_pos, goal_pos=goal_pos)

# display if found
if path:
    show_path(grid=grid, path=path)
else:
    print("No path found.")
