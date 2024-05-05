import numpy as np
from utils import *
import cv2
import json
import os
from pathlib import Path
import time
from multiprocessing import Pool

def vis_map_regions(opt_wave, unopt_wave, grid):
    for i in range(len(unopt_wave)):
        grid[unopt_wave[i][0],unopt_wave[i][1]]=100

    for i in range(len(opt_wave)):
        grid[opt_wave[i][0],opt_wave[i][1]]=200
    
    return grid


def find_opt_region(grid, val_grid):
    '''
    Find optimal path by expanding neighbours whose cost-to-go is lower than current.
    '''
    start=(grid.shape[1]-1,0)

    open_curr_wave = []
    open_curr_wave.append((start[0], start[1]))
    closed=set()
    out_img=np.copy(grid)
    opt_path_list=[]
    while len(open_curr_wave)>0:
        open_next_wave=[]
        while len(open_curr_wave)>0:
            (x,y)=open_curr_wave.pop()
            s=(x,y)
            closed.add(s)
            out_img[s[0],s[1]]=200
            opt_path_list.append(s)

            for move in delta:
                if check_move(s,move,grid):
                    s2=(s[0] + move[0],s[1] + move[1])

                    if s2 not in closed and val_grid[s2[0],s2[1]] < val_grid[s[0],s[1]]:
                        open_next_wave.append(s2)
                        closed.add(s2)

           
        open_curr_wave=open_next_wave
    # visualise_map(out_img)

    return list(opt_path_list)



def find_surrounding_regions(grid, steps, opt_wave, max_waves):
    open_curr_wave=opt_wave.copy()
    closed=set(opt_wave.copy())
    
    waves={}
    i=0
    wave_width=1
    #Find n neighbour waves of states
    while len(open_curr_wave)>0:
        open_next_wave=[]
        key=int(i/wave_width)

        if key>max_waves:
            break

        while len(open_curr_wave)>0:
            s=open_curr_wave.pop()
            closed.add(s)

            for move in delta:
                s2=(s[0] + move[0],s[1] + move[1])
                if check_move(s,move,grid) and s2 not in closed:
                    open_next_wave.append(s2)
                    closed.add(s2)
        
        i+=1
        if key not in waves:
            waves[key]=open_next_wave.copy()
        else:
            waves[key].extend(open_next_wave.copy())
            
        open_curr_wave=open_next_wave
    
    # Rest of map
    closed=[list(t) for t in closed]
    all_space=(np.array(np.where(steps!=-1)).T).tolist()
    rom = [x for x in all_space if x not in closed]
    waves['rom']=rom

    return waves


def find_obstacle_region(grid, visualise=False):
    open_curr_wave = [[int(x), int(y)] for x, y in zip(*np.where(grid == 0))]
    save_open_wave=open_curr_wave.copy()
    closed=[]
    open_next_wave=[]
    rest_of_map=[]

    i=0
    while i<2:
        open_next_wave=[]
        while len(open_curr_wave)>0:
            s=open_curr_wave.pop()
            closed.append(s)

            for move in delta:
                s2=[s[0] + move[0],s[1] + move[1]]
                if check_in_map(s,move,grid) and s2 not in save_open_wave:
                    if s2 not in closed:
                        open_next_wave.append(s2)
                        rest_of_map.append(s2)
                        closed.append(s2)
        i+=1
        open_curr_wave=open_next_wave

    if visualise:
        grid=vis_map_regions(rest_of_map, [], grid)
        grid=cv2.resize(grid, (256,256), interpolation = cv2.INTER_AREA)
        visualise_map(grid)

    return rest_of_map



def run_region_generation(map_type_path, num_maps, max_waves=3):
    '''
    Based on cost-to-go data for all space from generate_data, find optimal path and n neighbour waves.
    Outputs json with opt_path states, n-1 waves of states distanced from opt_path, and rest of map (rom).
    '''
    
    img_paths=map_type_path + '/input/'
    step_paths=map_type_path + '/ctg_data/'
    save_reg_paths=map_type_path + '/opt_regions/'
    os.makedirs(save_reg_paths, exist_ok=True)

    for i in range(num_maps):
        img_path=img_paths+str(i)+'.png'
        step_path=step_paths+str(i)+'.npy'

        img=cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
        steps=np.load(step_path).astype(float)
        opt_path=find_opt_region(img, steps)
        waves=find_surrounding_regions(img, opt_path, max_waves)

        full_map={}
        full_map['opt_path']=opt_path
        for key in waves.keys():
            full_map[key]=waves[key]


        with open(save_reg_paths+str(i)+'.json', 'w') as fp:
            json.dump(full_map, fp)


def region_generation_map(data):
    '''
    Based on cost-to-go data for all space from generate_data, find optimal path region and n neighbour waves.
    Outputs json with opt_path states, n-1 waves of states distanced from opt_path, and rest of map (rom).
    '''
    map_type_path=data[0]
    i=data[1]
    max_waves=3
    img_paths=map_type_path + '/input/'
    step_paths=map_type_path + '/ctg_data/'
    save_reg_paths=map_type_path + '/opt_regions/'
    os.makedirs(save_reg_paths, exist_ok=True)

    img_path=img_paths+str(i)+'.png'
    step_path=step_paths+str(i)+'.npy'

    img=cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    steps=np.load(step_path).astype(float)
    opt_path=find_opt_region(img, steps)
    waves=find_surrounding_regions(img, steps, opt_path, max_waves)

    full_map={}
    full_map['opt_path']=opt_path
    for key in waves.keys():
        full_map[key]=waves[key]


    with open(save_reg_paths+str(i)+'.json', 'w') as fp:
        json.dump(full_map, fp)


def run_region_generation_parallel(map_type_path, num_maps):
    num_maps=range(0,num_maps)
    paths=[map_type_path]*len(num_maps)
    data=list(zip(paths, num_maps))
    with Pool(os.cpu_count() - 1) as pool:
        pool.map(region_generation_map, data)


def generate_percent_wave(json_file, save_vals_paths, mapdim=201):
    x=-np.ones((mapdim,mapdim)).astype(float)

    with open(json_file, 'r') as fp:
        data = json.load(fp)
    
    eps_vals=np.linspace(0,1, len(data.keys()))

    for e, key in zip(eps_vals, data.keys()):
        for s in data[key]:
            x[s[0],s[1]]=1-e

    
    savepath=Path(save_vals_paths) / Path(json_file).parts[-1]

    np.save(str(savepath)[:-4]+'npy', x)
    


def generate_percentage_waves(map_type_path, num_maps):
    '''
    Generate npy outputs that for all states have 0-1 value of how close to optimal path the state is.
    0- Fully unoptimal, 1- fully optimal.
    '''
    json_inputs_path=map_type_path + '/opt_regions/'
    save_vals_paths=map_type_path + '/opt_regions_norm/'
    os.makedirs(save_vals_paths, exist_ok=True)

    json_paths = [os.path.join(str(json_inputs_path), f"{i}.json") for i in range(num_maps)]

    for json_file in json_paths:
        generate_percent_wave(json_file, save_vals_paths)




if __name__ == "__main__":
    map_type_path=get_root_path()/'data/single_bugtrap'
    t0=time.time()
    run_region_generation_parallel(str(map_type_path), 500)
    t1=time.time()
    print(t1-t0)
    generate_percentage_waves(str(map_type_path), 500)
    print(time.time()-t1)