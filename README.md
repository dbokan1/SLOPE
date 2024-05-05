# SLOPE- Search with Learned Optimal Pruning-based Expansion

This repository contains the Pytorch implementation of 'SLOPE: Search with Learned Optimal Pruning-based Expansion'.

## Setup
```shell
conda create -n slope python=3.8
conda activate slope
pip install -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
Both the training data and pretrained weights can be downloaded [here](https://drive.google.com/drive/folders/1HXeTwASV0fNsTHdKfpG_a7KREPXKWy54?usp=sharing).
For training or evaluation, the directory structure should be as such:
```
SLOPE/
│
├── src/
├── data/
└── weights/
```

## Usage
Generating optimal cost-to-go data, stored in `data/map_type/ctg_data`, is done with `python generate_data.py`; manually set dataset type and sample number in the script. Similarly, optimal regions are found with `python generate_regions.py`; pre-generated cost-to-go data is required.

To run training, set dataset type, number of epochs and batch size in `train.py`. Options for data balancing are located in loading functions from `load_data.py`.

Evaluation of trained models is done on maps indexed 400 to 500 by running `python eval.py`. Model with a singular search algorithm can be run with the `run_eval` function, while benchmarking both SLOPE algorithms with and without neural cost-to-go is done with `compare_heurs` function. Note pruning thresholds should be manually set.
