from pathlib import Path
import numpy as np
from multiprocessing import Pool
import os
import cv2
import json
import random
from utils import *

delta = [[-1, 0], # go up
         [ 0,-1], # go left
         [ 1, 0], # go down
         [ 0, 1], # go right
         [ 1, 1], # down right diagonal
         [ 1,-1], # down left diagonal
         [-1, 1], # up right diagonal
         [-1,-1],] # up left diagonal


def visualise_map(image):
    """
    prop_id: Window property to edit such as cv2.WINDOW_NORMAL, cv2.WINDOW_KEEPRATIO, cv2.WINDOW_FULLSCREEN, etc.
    prop_value: New value of the window property such as cv2.WND_PROP_FULLSCREEN, cv2.WND_PROP_AUTOSIZE, cv2.WND_PROP_ASPECT_RATIO, etc.
    """
    cv2.namedWindow("foo", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("foo", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("foo", image)
    cv2.waitKey()
    cv2.destroyWindow("foo")


def load_one_point(data):
    img=cv2.cvtColor(cv2.imread(str(data[3])), cv2.COLOR_BGR2GRAY)
    img=prepareInputImage(img, [data[0],data[1]])
    return [img, data[2]]


def load_ctg_map(input_path):
    
    out_path=input_path.parent.parent / 'ctg_data' / (str(input_path.parts[-1])[:-3]+'npy')
    out_data=np.load(out_path, allow_pickle=True)
    x_coords, y_coords = np.where(out_data!=-1)
    data_to_process = [(x, y, out_data[x, y], input_path) for x, y in zip(x_coords, y_coords)]

    img_dist_list=[]
    for i in range(len(data_to_process)):
        img_dist_list.append(load_one_point(data_to_process[i]))

    return img_dist_list


def load_ctg_data(dataset_path: Path, num_samples):
    input=dataset_path / 'input'

    image_paths = [Path(os.path.join(str(input), f"{i}.png")) for i in range(num_samples)]

    num_tasks= 8

    with Pool(processes=num_tasks) as pool:
        data=pool.map(load_ctg_map, image_paths)

    out_data_flattened=[]
    for data_array in data:
        out_data_flattened.extend(data_array)
    
    X = [item[0] for item in out_data_flattened]
    y = [item[1] for item in out_data_flattened]
    X = np.array(X)
    y = np.array(y)
    print("Loaded ", X.shape[0], ' data points')

    return X, y


####################################################################################################################

def load_one_map(input_path):
    
    wave_path=input_path.parent.parent / 'opt_regions' / (str(input_path.parts[-1])[:-3]+'json')
    percent_path=input_path.parent.parent / 'opt_regions_norm' / (str(input_path.parts[-1])[:-3]+'npy')

    percent_data=np.load(percent_path, allow_pickle=True)
    with open(wave_path, 'r') as fp:
        json_data = json.load(fp)
    
    full_point_list=[]

    for key in json_data.keys():
        if key=='rom':
            idx=int(len(json_data[key]) * 0.5)
            full_point_list+=random.sample(json_data[key], idx)
        else:
            full_point_list+=json_data[key]
    
    
    data_to_process = [(x, y, percent_data[x, y], input_path) for [x, y] in full_point_list]

    img_dist_list=[]
    for i in range(len(data_to_process)):
        img_dist_list.append(load_one_point(data_to_process[i]))

    return img_dist_list


def load_data(dataset_path: Path, num_samples):
    input=dataset_path / 'input'

    image_paths = [Path(os.path.join(str(input), f"{i}.png")) for i in range(num_samples)]
    num_tasks= 8

    with Pool(processes=num_tasks) as pool:
        data=pool.map(load_one_map, image_paths)

    out_data_flattened=[]
    for data_array in data:
        out_data_flattened.extend(data_array)
    
    X = [item[0] for item in out_data_flattened]
    y = [item[1] for item in out_data_flattened]
    X = np.array(X)
    y = np.array(y)
    print("Loaded ", X.shape[0], ' data points')

    return X, y

#####################################################################################


def load_one_point_weights(data):
    img=cv2.cvtColor(cv2.imread(str(data[3])), cv2.COLOR_BGR2GRAY)
    weight=1
    for move in delta:
        if check_in_map([data[0],data[1]],move,img) and check_if_black([data[0],data[1]],move,img):
            weight=1.5 #attempt to weigh close to obstacle points more, ignore
            break
    img=prepareInputImage(img, [data[0],data[1]], transform=False)
    return [img, data[2], weight]


def load_one_map_weights(input_path):
    balance=input_path[1]
    input_path=input_path[0]
    wave_path=input_path.parent.parent / 'opt_regions' / (str(input_path.parts[-1])[:-3]+'json')
    percent_path=input_path.parent.parent / 'opt_regions_norm' / (str(input_path.parts[-1])[:-3]+'npy')

    percent_data=np.load(percent_path, allow_pickle=True)
    with open(wave_path, 'r') as fp:
        json_data = json.load(fp)
    
    full_point_list=[]

    opt_len=len(json_data['opt_path'])
    for key in json_data.keys():
        # balancing certain datasets manually
        # if key=='rom':
        #     idx=int(len(json_data[key]) *0.25)
        #     random.seed(42)
        #     full_point_list+=random.sample(json_data[key], idx)
        # else:
        if balance:
            full_point_list+=random.sample(json_data[key], min(opt_len, len(json_data[key])))
        else:
            if key=='rom':
                idx=int(len(json_data[key]) *0.5)
                random.seed(42)
                full_point_list+=random.sample(json_data[key], idx)
            else:
                full_point_list+=json_data[key]
    
    
    data_to_process = [(x, y, percent_data[x, y], input_path) for [x, y] in full_point_list]

    img_dist_list=[]
    for i in range(len(data_to_process)):
        img_dist_list.append(load_one_point_weights(data_to_process[i]))

    return img_dist_list


def load_weighted_data(dataset_path: Path, num_samples, balance):
    input=dataset_path / 'input'

    image_paths = [[Path(os.path.join(str(input), f"{i}.png")), balance] for i in range(num_samples)]

    num_tasks= 8

    with Pool(processes=num_tasks) as pool:
        data=pool.map(load_one_map_weights, image_paths)

    out_data_flattened=[]
    for data_array in data:
        out_data_flattened.extend(data_array)
    
    X = [item[0] for item in out_data_flattened]
    y = [item[1] for item in out_data_flattened]
    w = [item[2] for item in out_data_flattened]
    X = np.array(X)
    y = np.array(y)
    w = np.array(w) #ignore

    print("Loaded ", X.shape[0], ' data points')

    return X, y, w