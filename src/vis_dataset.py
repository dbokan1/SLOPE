import cv2
import numpy as np
from pathlib import Path
import os
from utils import *

# script for visualizing optimality regions

num_samples=100
dataset_name='alternating_gaps'
x= get_root_path()/'data'/dataset_name
input = x / 'input'

image_paths = [Path(os.path.join(str(input), f"{i}.png")) for i in range(num_samples)]

for exp in image_paths:
    img=cv2.cvtColor(cv2.imread(str(exp)), cv2.COLOR_BGR2GRAY)
    img=img/255

    percent_path=exp.parent.parent / 'opt_regions_norm' / (str(exp.parts[-1])[:-3]+'npy')
    percent_data=np.load(percent_path, allow_pickle=True)
    x_arr,y_arr=np.where(percent_data!=-1)

    for x, y in zip(x_arr,y_arr):
        img[x,y]=percent_data[x,y]
    
    visualise_map(img)