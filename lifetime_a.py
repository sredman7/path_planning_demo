import numpy as np
import heapq
import matplotlib.pyplot as plt

## Lifetime Planning A*
# structure: Graph contains all vertices keyed by their position and start/goal information. Grid is an occupancy grid of obstacles. 
# Queue is a heap list of (key, position) tuple pairs.

def calculate_key(vertex: dict, goal_pos: tuple[int, int]) -> tuple[float, float]:
    '''
    Calculates the key of a vertex on the graph.

    Keyword Arguments:
    vertex -- the vertex whose key is being calculated
    start_pos -- the start position of the graph, (x,y)

    Returns:
    Key pair for priority queue
    '''
    return min(vertex['g'], vertex['rhs']) + heuristic(vertex['position'], goal_pos), min(vertex['g'], vertex['rhs'])

def heuristic(pos1: tuple[int, int], pos2: tuple[int, int]) -> float:
    '''
    Calculates the heuristic between two positions on the graph, in this case
    the Euclidean distance.
    '''
    x1, y1 = pos1
    x2, y2 = pos2
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def update_vertex(vertex: dict, graph: dict, queue: list):
    '''
    Updates the rhs and key of a vertex
    '''
    if vertex['position'] != graph['start_pos']:
        # update look-ahead distance estimate
        rhs_min = float('inf')
        for neighbor_pos in vertex['neighbors']:
            neighbor = graph[neighbor_pos]
            rhs_min = min(rhs_min, neighbor['g'] + heuristic(neighbor['position'], vertex['position']))
        vertex['rhs'] = rhs_min

    # remove from queue if present
    in_queue = [item for item in queue if vertex['position'] in item] # weird fanagling to grab vertex with matching position from key queue
    if in_queue != []:
        queue.remove(in_queue[0])

    # add back to queue if inconsistent
    if vertex['g'] != vertex['rhs']:
        vertex['key'] = calculate_key(vertex=vertex, goal_pos=graph['goal_pos'])
        heapq.heappush(queue, (vertex['key'], vertex['position']))
        
def compute_shortest_path(queue: list, graph: dict) -> list[tuple[int, int]]:
    '''
    Computes and returns shortest path between start and goal.
    '''
    # loop while goal is inconsistent or there's a more efficient path
    while queue[0][0] < calculate_key(vertex=graph[graph['goal_pos']], goal_pos=graph['goal_pos']) or \
        graph[graph['goal_pos']]['g'] != graph[graph['goal_pos']]['rhs']:
        
        _, current_pos = heapq.heappop(queue)
        
        # if overconsistent, make consistent and update neighbors
        if graph[current_pos]['g'] > graph[current_pos]['rhs']:
            graph[current_pos]['g'] = graph[current_pos]['rhs']
            for neighbor_pos in graph[current_pos]['neighbors']:
                update_vertex(
                    vertex = graph[neighbor_pos], 
                    graph = graph, 
                    queue = queue
                )
        
        # otherwise, mark inf cost and update self and neighbors
        else:
            graph[current_pos]['g'] = float('inf')
            for neighbor_pos in graph[current_pos]['neighbors']:
                update_vertex(
                    vertex = graph[neighbor_pos], 
                    graph = graph, 
                    queue = queue
                )
            update_vertex(
                vertex = graph[current_pos],
                graph = graph,
                queue = queue
            )
    
    # trace and return path
    if graph[graph['goal_pos']]['g'] == float('inf'): return [] # empty if no path found
    
    else:
        current_pos = graph['goal_pos']
        path = [current_pos]

        while current_pos != graph['start_pos']:
            cost_min = float('inf')
            pos_min = current_pos

            # find neighbor with lowest path cost
            for neighbor_pos in graph[current_pos]['neighbors']:
                if graph[neighbor_pos]['g'] + heuristic(neighbor_pos, current_pos) < cost_min:
                    cost_min = graph[neighbor_pos]['g'] + heuristic(neighbor_pos, current_pos)
                    pos_min = neighbor_pos
            
            # append to path and iterate
            path.append(pos_min)
            current_pos = pos_min

        return path
    
def get_neighbors(grid: np.ndarray, position: tuple[int, int]) -> list[tuple[int, int]]:
    '''
    Checks occupancy grid and returns unoccupied adjacent nodes

    Keyword Arguments:
    grid -- 2D numpy array occupancy grid, 0 = free 1 = obstacle; note (row, col) = (y, x)
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

def create_vertex(position: tuple[int, int], g = float('inf'), rhs = float('inf'), neighbors = None) -> dict:
    '''
    Create a vertex with the given properties
    '''
    return {
        'position': position,
        'g': g,
        'rhs': rhs,
        'key': (float('inf'), float('inf')),
        'neighbors': neighbors
    }

def create_graph(grid: np.ndarray, start_pos: tuple[int, int], goal_pos: tuple[int, int]) -> dict:
    '''
    Creates the graph containing all vertices of the state space
    '''
    rows, cols = grid.shape
    graph = {}

    # populate graph with default vertices
    for nx in range(cols):
        for ny in range(rows):
            graph[(nx, ny)] = create_vertex(
                position = (nx, ny),
                neighbors = get_neighbors(grid=grid, position=(nx, ny))
            )

    # remember start and goal positions
    graph['start_pos'] = start_pos
    graph['goal_pos'] = goal_pos

    return graph

def show_path(grid: np.ndarray, path: list[tuple[int, int]]):
    '''
    Plot occupancy grid of obstacles and planned path, with start and goal positiions marked
    note grid (row, col) = (y, x)
    '''

    # create figure and show grid
    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap='binary')

    # plot path and markers
    if path:
        path_arr = np.array(path)
        plt.plot(path_arr[:,0], path_arr[:,1], 'b-', linewidth=3, label='Path')
        plt.plot(path_arr[0,0], path_arr[0,1], 'go', markersize=15, label='Start')
        plt.plot(path_arr[-1,0], path_arr[-1,1], 'ro', markersize=15, label='Goal')

    plt.grid(True)
    plt.legend(fontsize=12)
    plt.title('LPA* Path Planning')
    plt.show()

def modify_grid(grid: np.ndarray, coords: list[tuple[int, int]], val=1):
    '''
    Given an occupancy grid and list of coordinates (x, y), set the occupancy of each vertex

    Keyword Arguments:
    grid -- occupancy grid (row, col) = (y, x); 0 = empty, 1 = obstacle
    coords -- list of grid coordinates (x, y) to modify
    val -- value assigned to modified grid vertices, defaults to 1 (occupied)
    '''
    
    rows, cols = grid.shape
    for (x, y) in coords:
        if 0 <= x < cols and 0 <= y < rows:
            grid[y, x] = val
        else:
            print(f"obstacle at {x}, {y} out of bounds")

def update_edge_costs(graph: dict, grid: np.ndarray, queue: list, obs: list[tuple[int, int]]):
    '''
    Given a graph of vertices, an occupancy grid, a priority queue, and new obstacles, 
    update the relevant edge costs and vertices.

    Keyword Arguments:
    graph -- graph containing vertex information
    grid -- occupancy grid
    queue -- priority queue
    obs -- list of new obstacles
    '''

    rows, cols = grid.shape # grid bounds
    to_update = [] # empty list of vertices to update

    # add obstacle and neighbors to update list if not already there and within bounds
    for (x, y) in obs:
        for nx in range(x-1, x+2):
            for ny in range(y-1, y+2):
                if (nx, ny) not in to_update and 0<=nx<cols and 0<=ny<rows: 
                    to_update.append((nx, ny))

    # for each changed vertex, update neighbors then update vertex
    for (x, y) in to_update:
        graph[(x, y)]['neighbors'] = get_neighbors(grid=grid, position=(x,y))
        update_vertex(vertex=graph[(x,y)], graph=graph, queue=queue)

## initialization
# obstacle grid
grid = np.zeros((20 ,30))

# horizontal wall
obs = []
for x in range(5,25): obs.append((x, 10))
modify_grid(grid=grid, coords=obs)


# start and goal positions
start_pos = (4, 2)
goal_pos = (28, 18)

# graph of all vertices
graph = create_graph(
    grid = grid,
    start_pos = start_pos,
    goal_pos = goal_pos
)

# set start vertex rhs to zero
graph[graph['start_pos']]['rhs'] = 0

# create queue and add start vertex
queue = [(calculate_key(graph[graph['start_pos']], graph['goal_pos']), graph['start_pos'])]

## main path planning loop

# forever
    # compute_shortest_path
    # wait for changes in edge costs
    # for all edges with changed costs
        # update edge cost
        # update vertex


## testing and visualization
# find base path
path = compute_shortest_path(queue=queue, graph=graph)

# display if found
if path:
    show_path(grid=grid, path=path)
else:
    print("No path found.")

# add new obstacle
obs = []
for y in range(5,10): obs.append((15, y))
modify_grid(grid=grid, coords=obs)

# update edge costs and vertices
update_edge_costs(graph=graph, grid=grid, queue=queue, obs=obs)

# recompute and display path
path = compute_shortest_path(queue=queue, graph=graph)

if path:
    show_path(grid=grid, path=path)
else:
    print("No path found.")