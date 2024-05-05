import numpy as np
import copy
from utils import *
import torch
from typing import List

def reconstruct_path(parent_grid, goal):
    path = []
    current = goal
    while current!=[0,0]:
        path.append(current)
        x=list(parent_grid[current[0],current[1]].astype(int))
        current = x
    return path[::-1] # Return reversed path


def plot_path(path: List[List], map: np.ndarray):
    for [x,y] in path:
        map[x,y]=50
    return map


def euclid_distance(a: List, b: List):
    return np.linalg.norm(np.array(a)- np.array(b))

def normed_euclid_distance(state: List, goal: List = [0,31], start: List = [31,0]):
    max_dist=euclid_distance(start, goal)
    state_dist=euclid_distance(state, goal)
    return 1- state_dist/max_dist

def chessboard_dist(a: List, b: List):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def greedy_vanilla(grid, model=None):
    '''
    Classic greedy search with euclid distance as heuristic.
    '''
    return greedy_heur(grid)


def greedy_heur(grid, model=None):
    '''
    Greedy search variant using model prediction as heuristic.
    '''
    end = False
    maxdim=grid.shape[0]-1
    open = [[0, 0, maxdim,0]]
    open_pos= [[maxdim, 0]]
    closed=[]

    goal = [0,maxdim]
    grid2=copy.deepcopy(grid)
    parent_grid=np.zeros((grid.shape[0],grid.shape[0],2))
    i=0
    while end == False:
        open.sort()
        open.reverse()
        
        [heur, steps, x,y]= open.pop()
        open_pos = [pos for pos in open_pos if pos != [x,y]]
        closed.append([x,y])
        i=i+1
        grid2[x][y]=100
        s=[x,y]
        for move in delta:
            if check_move(s,move,grid2):
                n2=[s[0] + move[0],s[1] + move[1]]

                if model:
                    input=torch.from_numpy(prepareInputImage(grid,n2)).float().unsqueeze(1).to(next(model.parameters()).device)
                    c=model(input)
                else:
                    c=euclid_distance(n2,goal)
                
                if n2 not in closed and [n2[0],n2[1]] not in open_pos:
                    parent_grid[n2[0],n2[1],:]=s
                    open.append([c, steps+1, n2[0],n2[1]])
                    open_pos.append([n2[0],n2[1]])
                

                if s[0] + move[0] == goal[0] and s[1] + move[1] == goal[1]:
                    end = True
                    path=reconstruct_path(parent_grid, [0, maxdim])
                    # single_path_vis=plot_path(path, grid2)
                    return grid2, i, len(open), len(path)-1
                
    path=reconstruct_path(parent_grid, [0, maxdim])
    return grid2, i, len(open), len(path)-1



def SLOPEr(grid, model, threshold=0.9, ctg_model=None):
    '''
    SLOPE search with threshold pruning and recursive calling. Threshold recursively decreases by 0.1.
    Converges to classic greedy.
    '''
    end = False
    maxdim=grid.shape[0]-1
    open = [[0, 0, maxdim,0]]
    open_pos= [(maxdim, 0)]
    closed=set()

    goal = [0,maxdim]
    vis_grid=copy.deepcopy(grid)
    parent_grid=np.zeros((grid.shape[0],grid.shape[0],2))
    i=0
    while end == False:
        open.sort()
        open.reverse()
        [value, me, x,y]= open.pop()
        open_pos = [pos for pos in open_pos if pos != (x,y)]
        closed.add((x,y))
        i=i+1
        vis_grid[x][y]=100

        s=(x,y)
        for move in delta:
            if check_move(s,move,grid):
                n2=(s[0] + move[0],s[1] + move[1])

                if isinstance(model, np.ndarray):   #support for using gt optimal notations
                    model_eval=model[n2[0],n2[1]]
                else:
                    input = torch.from_numpy(prepareInputImage(grid,n2)).float().unsqueeze(1).to(next(model.parameters()).device)
                    model_eval=model(input)
                
                if ctg_model:   #support for using neural ctg
                    input = torch.from_numpy(prepareInputImage(grid,n2)).float().unsqueeze(1).to(next(ctg_model.parameters()).device)
                    c=ctg_model(input)
                else:
                    c=euclid_distance(n2,goal)
                
                if n2 not in closed and (n2[0],n2[1]) not in open_pos:
                    parent_grid[n2[0],n2[1],:]=s
                    if model_eval>threshold:
                        open.append([c, model_eval, n2[0],n2[1]])
                        open_pos.append((n2[0],n2[1]))
                

                if s[0] + move[0] == goal[0] and s[1] + move[1] == goal[1]:
                    end = True
                    path=reconstruct_path(parent_grid, [0, maxdim])
                    reconstruct_path_vis = plot_path(path, vis_grid)
                    return reconstruct_path_vis, i, len(open), len(path)-1
        
        if len(open) == 0 and end!=True:
            return SLOPEr(grid, model, threshold-0.1, ctg_model)
    
    path=reconstruct_path(parent_grid, [0, maxdim])
    reconstruct_path_vis=plot_path(path, vis_grid)
    return reconstruct_path_vis, i, len(open), len(path)-1




def SLOPE(grid, model, threshold=0.5, ctg_model=None):
    '''
    Greedy search variant, prunes only optimal elements under a threshold and adds to backup list.
    Above threshold nodes are added to open list.
    '''
    end = False
    maxdim=grid.shape[0]-1
    open = [[0, 0, maxdim, 0]]
    open_pos= [(maxdim, 0)]
    closed=set()
    backup=[]
    backup_pos=[]

    goal = [0,maxdim]
    vis_grid=copy.deepcopy(grid)
    i=0
    parent_grid=np.zeros((grid.shape[0],grid.shape[0],2))

    while end == False:
        open.sort()
        open.reverse()
        [value, me, x,y]= open.pop()
        open_pos = [pos for pos in open_pos if pos != (x,y)]
        closed.add((x,y))
        i=i+1
        vis_grid[x][y]=100
        s=(x,y)
        for move in delta:
            if check_move(s,move,grid):
                n2=(s[0] + move[0],s[1] + move[1])
                
                if isinstance(model, np.ndarray):   #support for using gt optimal notations
                    model_eval=model[n2[0],n2[1]]
                else:
                    input = torch.from_numpy(prepareInputImage(grid,n2)).float().unsqueeze(1).to(next(model.parameters()).device)
                    model_eval=model(input)
                
                if ctg_model:   #support for using neural ctg
                    input = torch.from_numpy(prepareInputImage(grid,n2)).float().unsqueeze(1).to(next(ctg_model.parameters()).device)
                    c=ctg_model(input)
                else:
                    c=euclid_distance(n2,goal)
                
                if n2 not in closed and (n2[0],n2[1]) not in open_pos:
                    parent_grid[n2[0],n2[1],:] = s
                    if model_eval>threshold:
                        open.append([c, model_eval, n2[0],n2[1]])
                        open_pos.append((n2[0],n2[1]))
                    else:
                        backup.append([c, model_eval, n2[0],n2[1]])
                        backup_pos.append((n2[0],n2[1]))

                if s[0] + move[0] == goal[0] and s[1] + move[1] == goal[1]:
                    end = True
                    path = reconstruct_path(parent_grid, [0, maxdim])
                    reconstruct_path_vis = plot_path(path, vis_grid)
                    return reconstruct_path_vis, i, len(open), len(path)-1
        
        if len(open) == 0 and end!=True:
            print('Backing up...')
            open=backup
            open_pos=backup_pos
            backup=[]
            backup_pos=[]
            threshold/=2

    path=reconstruct_path(parent_grid, [0, maxdim])
    reconstruct_path_vis=plot_path(path, vis_grid)
    return reconstruct_path_vis, i, len(open), len(path)-1