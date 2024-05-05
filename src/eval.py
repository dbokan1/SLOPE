from model import OptPathModel, CostToGoModel
from utils import *
import cv2
from search_algs import SLOPE, SLOPEr, greedy_heur, greedy_vanilla
import torch
import sys
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize, suppress=True)

def get_algorithm(alg_type: str):
    if alg_type=='heur':
        return greedy_heur
    if alg_type=='pruning':
        return SLOPE
    if alg_type=='recursive':
        return SLOPEr
    if alg_type=='vanilla':
        return greedy_vanilla


def box_plot(dataset_type, x1, x2, x3, x4, x5):
    plt.figure(figsize=(5, 4))

    bp = plt.boxplot([x1, x2, x3, x4, x5], 
                    positions=[1, 2, 3, 4, 5],
                    widths=0.6,
                    patch_artist=True,
                    medianprops=dict(color="black"))
    
    colors = ['gold', 'skyblue', 'skyblue', 'lightgreen', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.ylabel('Expansions', fontsize=14)
    plt.title('Single bugtrap')
    plt.xticks([1, 2, 3, 4, 5], ['CTG', 'SLOPE', 'SLOPE*', 'SLOPEr', 'SLOPEr*'])

    plt.grid(True, linestyle='--', alpha=0.7)
    # plt.show()
    plt.savefig(str(get_root_path()/ 'data'/ dataset_type /'boxplot.png'))



def compare_heurs(model, dataset_type, map_range, ctg_model=None, verbose=False):
    opt_sum=0
    model_sum1=0
    open_sum1=0
    sing_sum1=0

    model_sum2=0
    open_sum2=0
    sing_sum2=0

    model_sum3=0
    open_sum3=0
    sing_sum3=0

    model_sum4=0
    open_sum4=0
    sing_sum4=0

    model_sum5=0
    open_sum5=0
    sing_sum5=0

    eval1=greedy_heur
    eval2=SLOPE
    eval3=SLOPEr

    steps1=[]
    steps2=[]
    steps3=[]
    steps4=[]
    steps5=[]

    print('Running eval...')

    for iter_map in map_range:
        imgpath=get_root_path() / 'data' / dataset_type / 'input' / (str(iter_map)+'.png')
        img=cv2.cvtColor(cv2.imread(str(imgpath)), cv2.COLOR_BGR2GRAY)
        dist_path=get_root_path() / 'data' / dataset_type / 'ctg_data' / (str(iter_map)+'.npy')
        dist=np.load(dist_path)
        dist_val=dist[-1, 0]

        out1, out_steps1, open_size1, sing1 = eval1(img, ctg_model)
        model_sum1+=out_steps1
        open_sum1+=open_size1
        sing_sum1+=sing1
        steps1.append(out_steps1)

        out2, out_steps2, open_size2, sing2=eval2(img, model, 0.67)
        model_sum2+=out_steps2
        open_sum2+=open_size2
        sing_sum2+=sing2
        steps2.append(out_steps2)

        out3, out_steps3, open_size3, sing3=eval2(img, model,0.6, ctg_model=ctg_model)
        model_sum3+=out_steps3
        open_sum3+=open_size3
        sing_sum3+=sing3
        steps3.append(out_steps3)

        out4, out_steps4, open_size4, sing4=eval3(img, model)
        model_sum4+=out_steps4
        open_sum4+=open_size4
        sing_sum4+=sing4
        steps4.append(out_steps4)

        out5, out_steps5, open_size5, sing5=eval3(img, model, ctg_model=ctg_model)
        model_sum5+=out_steps5
        open_sum5+=open_size5
        sing_sum5+=sing5
        steps5.append(out_steps5)

        opt_sum+=dist_val

        if verbose:
            print('Steps/open, heur:', model_sum1, open_sum1, ', prune:',model_sum2, open_sum2, ', mix:',model_sum3, open_sum3)
            print('-'*20)
            visualise_map(out1)
            visualise_map(out2)
            visualise_map(out3)
            visualise_map(out4)
            visualise_map(out5)
        
    print('Total heur: ', model_sum1, sing_sum1, open_sum1)
    print('Total prune: ', model_sum2, sing_sum2,open_sum2)
    print('Total prune mix: ', model_sum3, sing_sum3,open_sum3)
    print('Total rec: ', model_sum4, sing_sum4,open_sum4)
    print('Total rec mix: ', model_sum5, sing_sum5,open_sum5)

    res_map={'heur': [model_sum1, open_sum1],
             'prune': [model_sum2, open_sum2],
             'prune_heur': [model_sum3, open_sum3],
             'rec': [model_sum4, open_sum4],
             'rec_heur': [model_sum5, open_sum5],
             }
    
    box_plot(dataset_type, steps1, steps2, steps3, steps4, steps5)
    return res_map
    


def run_eval(model, dataset_type, map_range, ctg_model=None, threshold=0.5, alg_type='pruning', verbose=False):
    opt_sum=0
    model_sum=0
    open_sum=0
    reconstruct_path_sum=0
    eval_alg=get_algorithm(alg_type)

    print('Running eval...')
    steps_taken=[]
    for iter_map in map_range:
        imgpath=get_root_path() / 'data' / dataset_type / 'input' / (str(iter_map)+'.png')
        img=cv2.cvtColor(cv2.imread(str(imgpath)), cv2.COLOR_BGR2GRAY)
        dist_path=get_root_path() / 'data' / dataset_type / 'ctg_data' / (str(iter_map)+'.npy')
        dist=np.load(dist_path)

        dist_val=dist[-1, 0]
        out, out_steps, open_size, reconstruct_path_len=eval_alg(img, model, threshold, ctg_model)
        steps_taken.append(out_steps)
        model_sum+=out_steps
        opt_sum+=dist_val
        open_sum+=open_size
        reconstruct_path_sum+=reconstruct_path_len

        if verbose:
            print('Model steps: ', out_steps,', optimal steps: ', dist_val,'open list: ',open_size)
            print('-'*20)
            visualise_map(out)

    print('Total optimal: ', opt_sum)
    print('Total model: ', model_sum)
    print('Total open: ', open_sum)
    print('Total reconstruct path: ', reconstruct_path_sum)
    return model_sum



if __name__ == "__main__":
    dataset_type='multiple_bugtraps'

    model = OptPathModel()
    model.load_state_dict(torch.load(get_root_path() / 'weights' / dataset_type / 'OptPathModel/best_model.pth'))
    model.eval()

    ctg_model=CostToGoModel()
    ctg_model.load_state_dict(torch.load(get_root_path() / 'weights' / dataset_type / 'CTG/trained_model.pth'))
    ctg_model.eval()

    run_eval(model, dataset_type, range(400,500), threshold=0.8, alg_type='pruning', verbose=False)
    # compare_heurs(model, dataset_type, range(400,500), ctg_model=ctg_model, verbose=False)