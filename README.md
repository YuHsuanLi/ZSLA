# ZSLA
This project is the author implementation of [Make an Omelette with Breaking Eggs:Zero-Shot Learning for Novel Attribute Synthesis](https://arxiv.org/abs/2111.14182)

# Installation
## clone the project
Clone this project by git clone
```bash
git clone https://github.com/evaProjectARICL/ZSLA.git
```

## enviroment setting
First install anaconda and then create a enviroment, whose name is ZSLA (set in the first line of yaml file)
```bash
conda env create -f environment.yml
```
Activate the environment ZSLA
```bash
conda activate ZSLA
```

## run
Go to the ZSLA folder
```bash
cd ZSLA
```
### run ZSLA
Their are two stages in ZSLA, the first stage is to obtain the seen attribute directors and the second stage is to train intersection/union models and use them to syntheize the unseen attribute directors.
Here we use CUB dataset as example:
#### stage 1
If want to run ɑ-CLEVR dataset, add --data_type=alpha-CLEVR and change the cpt_weight from 0.2 to 1 (--cpt_weight=1)
train: 
```bash
python main.py --device=cuda:5 --stage=1 --is_normalized --is_abs --use_location_info --cpt_weight=0.2 --umc_start=0  --output_PATH=./stage1_cub
```
test: 
```bash
python main.py --test --device=cuda:5 --stage=1  --is_normalized --is_abs --attributes_PATH=./stage1_cub/classifier_10000.pth
```
#### stage 2
train: 
```bash
python main.py --device=cuda:5 --stage=2 --is_normalized --is_abs --attributes_PATH=./stage1_cub/classifier_10000.pth --output_PATH=./stage2_cub
```

test: 
```bash
python main.py --test --device=cuda:5 --stage=2 --gate_PATH=./stage2_cub/model_200000.pth
```
### run baseline
#### A-ESZSL
train: 
```bash
python main.py --baseline --method=A-ESZSL --device=cuda:5 --use_location_info --output_PATH=./A_ESZSL_cub 
```
test: 
```bash
python main.py --test --baseline --method=A-ESZSL --device=cuda:5 --use_location_info --baseline_PATH=./A_ESZSL_cub/model_5000.pth
```
#### A-LAGO
train: 
```bash
python main.py --baseline --method=A-LAGO --device=cuda:5 --use_location_info --output_PATH=/eva_data/hdd4/yu_hsuan_li/logic_kernel/output/A_LAGO_cub 
```
test: 
```bash
python main.py --test --baseline --method=A-LAGO --device=cuda:5 --use_location_info --baseline_PATH=/eva_data/hdd4/yu_hsuan_li/logic_kernel/output/A_LAGO_cub/model_5000.pth
```

