import os
import numpy as np
from multiprocessing import Pool
from utils import *
import cv2
from pathlib import Path


def explore_map(grid, start= None, end= None):
    '''
    Generates optimal cost-to-go values for all explorable states.
    '''

    if start is None:
        start=(0, grid.shape[1]-1)

    if end is None:
        end=(grid.shape[0]-1, 0)
        

    move_cost=1

    open_curr_wave = set()
    open_curr_wave.add((0,start[0], start[1]))
    closed=set()

    exit_grid=-1*np.ones_like(grid)

    while len(open_curr_wave)>0:
        open_next_wave=set()
        while len(open_curr_wave)>0:
            (g,x,y)=open_curr_wave.pop()
            s=(x,y)
            closed.add(s)

            for move in delta:
                s2=(s[0] + move[0],s[1] + move[1])
                if check_move(s,move,grid) and s2 not in closed:
                    element=(g+move_cost, s[0] + move[0],s[1] + move[1])
                    open_next_wave.add(element)
                    closed.add(s2)
                    exit_grid[element[1],element[2]]=element[0]
            
        open_curr_wave=open_next_wave


    return exit_grid
                

# Function to generate data for a single image
def generate_data_for_image(image_path):
    try:
        X = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
        y = explore_map(X)
        a= Path(image_path).parent.parent/ 'ctg_data' / (str(Path(image_path).parts[-1][:-3])+'npy')
        np.save(a, y)
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None


def generate_data(dataset_path, num_samples):
    image_paths = [os.path.join(str(dataset_path), f"{i}.png") for i in num_samples]
    os.makedirs(dataset_path.parent / 'ctg_data', exist_ok=True)
    
    with Pool(os.cpu_count() - 1) as pool:
        pool.map(generate_data_for_image, image_paths)




if __name__ == "__main__":
    bm_path=get_root_path()/'data/shifting_gaps/input'
    x=range(0,500)
    generate_data(bm_path, x)