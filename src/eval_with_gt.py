from model import CostToGoModel
from utils import *
import cv2
from search_algs import greedy_heur, SLOPE
import torch


def compare_heurs_gt(dataset_type, map_range, ctg_model=None):
    opt_sum=0
    model_sum1=0
    open_sum1=0
    rec_path1=0

    model_sum2=0
    open_sum2=0
    rec_path2=0

    model_sum3=0
    open_sum3=0
    rec_path3=0

    eval1=greedy_heur
    eval2=SLOPE

    steps1=[]
    steps2=[]
    steps3=[]

    print('Running eval...')

    for iter_map in map_range:
        imgpath=get_root_path() / 'data' / dataset_type / 'input' / (str(iter_map)+'.png')
        img=cv2.cvtColor(cv2.imread(str(imgpath)), cv2.COLOR_BGR2GRAY)
        dist_path=get_root_path() / 'data' / dataset_type / 'ctg_data' / (str(iter_map)+'.npy')
        dist=np.load(dist_path)
        dist_val=dist[-1, 0]

        percent_path=imgpath.parent.parent / 'opt_regions_norm' / (str(imgpath.parts[-1])[:-3]+'npy')
        percent_data=np.load(percent_path, allow_pickle=True)

        out1, out_steps1, open_size1, rp1=eval1(img, ctg_model)
        model_sum1+=out_steps1
        open_sum1+=open_size1
        rec_path1+=rp1
        steps1.append(out_steps1)

        out2, out_steps2, open_size2, rp2=eval2(img, percent_data, 0.9)
        model_sum2+=out_steps2
        open_sum2+=open_size2
        rec_path2+=rp2
        steps2.append(out_steps2)

        out3, out_steps3, open_size3, rp3 = eval2(img, percent_data, 0.9, ctg_model=ctg_model)
        model_sum3+=out_steps3
        open_sum3+=open_size3
        rec_path3+=rp3
        steps3.append(out_steps3)

        opt_sum+=dist_val
    
        
    print('Total heur: ', model_sum1, rec_path1, open_sum1)
    print('Total prune: ', model_sum2, rec_path2, open_sum2)
    print('Total prune mix: ', model_sum3, rec_path3, open_sum3)

    res_map={'heur': [model_sum1, open_sum1],
             'prune': [model_sum2, open_sum2],
             'prune_heur': [model_sum3, open_sum3],
             }
    return res_map




if __name__ == "__main__":
    dataset_type='single_bugtrap'

    ctg_model=CostToGoModel()
    ctg_model.load_state_dict(torch.load(get_root_path() / 'weights' / dataset_type / 'CTG/trained_model.pth'))
    ctg_model.eval()
    compare_heurs_gt(dataset_type, range(400,500), ctg_model=ctg_model)