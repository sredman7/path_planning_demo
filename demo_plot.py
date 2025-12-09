from d_lite import create_wall, create_box, dstar, gen_start_goal
import matplotlib.pyplot as plt
import numpy as np

## parameters
# start and goal positions
start_default = (4, 2)
goal_default = (28, 18)
bounds = (20, 30) # grid shape (rows, cols)

perception_range = 5 # robot perception range

## obstacle configurations
# cross + box
obs_cross = []
create_wall(lrange=(5,25), pos=10, obs=obs_cross) # horizontal wall
create_wall(lrange=(5,19), pos=15, obs=obs_cross, vertical=True) # vertical wall
create_box(xrange=(18,20), yrange=(5,7), obs=obs_cross) # box

# scattered boxes
obs_scatter = []
create_box(xrange=(5,7), yrange=(1,3), obs=obs_scatter)
create_box(xrange=(8,11), yrange=(5,6), obs=obs_scatter)
create_box(xrange=(11,12), yrange=(10,14), obs=obs_scatter)
create_box(xrange=(4,7), yrange=(12,13), obs=obs_scatter)
create_box(xrange=(18,21), yrange=(15,16), obs=obs_scatter)
create_box(xrange=(16,18), yrange=(5,10), obs=obs_scatter)
create_box(xrange=(24,26), yrange=(10,18), obs=obs_scatter)
create_box(xrange=(20,26), yrange=(0,4), obs=obs_scatter)
create_box(xrange=(1,4), yrange=(6,10), obs=obs_scatter)

# maze
obs_maze = []
# horizontal walls
create_wall(lrange=(10,17), pos=17, obs=obs_maze)
create_wall(lrange=(21,27), pos=19, obs=obs_maze)
create_wall(lrange=(23,30), pos=17, obs=obs_maze)
create_wall(lrange=(5,28), pos=15, obs=obs_maze)
create_wall(lrange=(5,9), pos=12, obs=obs_maze)
create_wall(lrange=(11,21), pos=11, obs=obs_maze)
create_wall(lrange=(16,21), pos=13, obs=obs_maze)
create_wall(lrange=(25,30), pos=13, obs=obs_maze)
create_wall(lrange=(3,9), pos=8, obs=obs_maze)
create_wall(lrange=(1,11), pos=6, obs=obs_maze)
create_wall(lrange=(13,19), pos=9, obs=obs_maze)
create_wall(lrange=(13,17), pos=2, obs=obs_maze)
create_wall(lrange=(13,21), pos=5, obs=obs_maze)
create_wall(lrange=(4,7), pos=4, obs=obs_maze)
create_wall(lrange=(21,25), pos=1, obs=obs_maze)
# vertical walls
create_wall(lrange=(6,18), pos=1, obs=obs_maze, vertical=True)
create_wall(lrange=(1,6), pos=2, obs=obs_maze, vertical=True)
create_wall(lrange=(8,17), pos=3, obs=obs_maze, vertical=True)
create_wall(lrange=(0,3), pos=4, obs=obs_maze, vertical=True)
create_wall(lrange=(12,15), pos=5, obs=obs_maze, vertical=True)
create_wall(lrange=(10,12), pos=6, obs=obs_maze, vertical=True)
create_wall(lrange=(15,18), pos=8, obs=obs_maze, vertical=True)
create_wall(lrange=(0,6), pos=9, obs=obs_maze, vertical=True)
create_wall(lrange=(8,10), pos=9, obs=obs_maze, vertical=True)
create_wall(lrange=(17,20), pos=10, obs=obs_maze, vertical=True)
create_wall(lrange=(0,3), pos=11, obs=obs_maze, vertical=True)
create_wall(lrange=(6,11), pos=11, obs=obs_maze, vertical=True)
create_wall(lrange=(11,13), pos=12, obs=obs_maze, vertical=True)
create_wall(lrange=(5,9), pos=13, obs=obs_maze, vertical=True)
create_wall(lrange=(11,13), pos=14, obs=obs_maze, vertical=True)
create_wall(lrange=(2,7), pos=17, obs=obs_maze, vertical=True)
create_wall(lrange=(7,9), pos=19, obs=obs_maze, vertical=True)
create_wall(lrange=(17,20), pos=19, obs=obs_maze, vertical=True)
create_wall(lrange=(5,11), pos=21, obs=obs_maze, vertical=True)
create_wall(lrange=(15,19), pos=21, obs=obs_maze, vertical=True)
create_wall(lrange=(4,13), pos=23, obs=obs_maze, vertical=True)
create_wall(lrange=(1,13), pos=25, obs=obs_maze, vertical=True)

## pick start and goal locations
# default
start_cross, goal_cross = start_default, goal_default
start_scatter, goal_scatter = start_default, goal_default
start_maze, goal_maze = (0,20), (15, 6)

# random
#start_cross, goal_cross = gen_start_goal(bounds=bounds, obs=obs_cross)
#start_scatter, goal_scatter = gen_start_goal(bounds=bounds, obs=obs_scatter)
#start_maze, goal_maze = gen_start_goal(bounds=(21,31), obs=obs_maze)

## testing
graph_cross,_,true_cross = dstar(
    start_pos = start_cross,
    goal_pos = goal_cross,
    bounds = bounds,
    obs = obs_cross,
    perception_range = perception_range
)

graph_scatter,_,_ = dstar(
    start_pos = start_scatter,
    goal_pos = goal_scatter,
    bounds = bounds,
    obs = obs_scatter,
    perception_range = perception_range
)

graph_maze,_,_ = dstar(
    start_pos=start_maze,
    goal_pos=goal_maze,
    bounds=(21,31),
    obs=obs_maze,
    perception_range=3
)

# show steps
paths = graph_cross['path_history']
fig, axs = plt.subplots(int(np.ceil(len(paths)/3)),3)

i = 0
for path in paths.values():
    axs.flat[i].imshow(true_cross, cmap='binary', origin='lower')
    
    arr = np.array(path)
    axs.flat[i].plot(arr[:,0], arr[:,1], 'b-')
    axs.flat[i].plot(arr[0,0], arr[0,1], 'go')
    axs.flat[i].plot(arr[-1,0], arr[-1,1], 'ro')

    i = i+1

axs[-1,-1].axis('off')

plt.show()