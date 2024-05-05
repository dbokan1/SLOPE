import numpy as np
import cv2
from pathlib import Path

delta = [[-1, 0], # go up
         [ 0,-1], # go left
         [ 1, 0], # go down
         [ 0, 1], # go right
         [ 1, 1], # down right diagonal
         [ 1,-1], # down left diagonal
         [-1, 1], # up right diagonal
         [-1,-1],] # up left diagonal


def check_in_map(s,move,grid):
    return (s[0] + move[0] < len(grid) and s[1] + move[1] < len(grid[0]) and s[0] + move[0] >= 0 and s[1] + move[1] >= 0)

def check_if_black(s,move,grid):
    return grid[s[0] + move[0]][s[1] + move[1]]==0

def check_move(s, move, grid):
    return (check_in_map(s,move,grid) and not check_if_black(s,move,grid))


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


def prepareInputImage(grid,s, transform=False):
    '''
    Takes in input map and current state, draws in the current and goal state, inverts colors.
    Adds simple transforms to input if transform flag is set.
    '''
    out = np.copy(grid).astype(float)
    out= 255 - out
    out[s[0],s[1]]=200
    out[0,-1]=100
    out= out / 255.
    # input transformations
    if transform:
        out=np.rot90(out, k=np.random.randint(-4, 5), axes=(0, 1))
        flip_mod=np.random.randint(0, 10)
        if flip_mod==0:
            out=np.flipud(out)
        if flip_mod==1:
            out=np.fliplr(out)
    
    out=out.reshape(1,32,32)
    return out

def get_root_path():
    return Path(__file__).parent.parent.resolve()
