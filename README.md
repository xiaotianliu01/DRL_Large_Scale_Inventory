# DRL_Large_Scale_Inventory
Official codes for ["Deep Reinforcement Learning for Large-Scale Inventory Management"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4490327)

## Create Environment
The codes are implementable on both Windows and Linux with Python 3.6.5 and pytorch.

## Prepare Your Data
1. Follow the example data under the folder .\data\ to form two csv files of your data.
2. Replace the example files under the folder .\data\ with your csv files.

   *Note. The example data is not the data used to generate the results in the paper ["Deep Reinforcement Learning for Large-Scale Inventory Management"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4490327). Specifically, the example demand data for all SKUs is generated with Poisson(20) and the example vlt data for all SKUs is generated with Poisson(3).

## How To Run
***You can start the DL training on the single echelon supply chain model with***
``` Bash
python dl_train.py --seed 1 --env-name "single_echelon" --log-dir "./logs_dl/" --save-dir './models_dl/dl_single_echelon.pt' --sku-mini-batch 25 --num-processes 4 --train-demand-data-pth "./data/df_sales.csv" --train-vlt-data-pth "./data/df_vlt.csv"
```
Here
```
--seed : random seed
--env-name : name for supply chain model, ''single_echelon'' or ''multi_echelon''
--log-dir : path for training logs
--save-dir : path for saved models
--sku-mini-batch : number of SKUs that are processed in a batch
--num-processes : number of processes to parrelly simulate supply chain models for multiple SKUs
--train-demand-data-pth : path for demand data
--train-vlt-data-pth : path for vlt data
```
  
  Note. Params *sku-mini-batch* and *num-processes* come from the parrel deisign in the codes which are meant to improve the code effeciency with a large volume of SKUs. You have to make sure that **sku-mini-batch*num-processes>=number of total SKUs in your dataset**. Otherwise, data of some of your SKUs may not be loaded and used during training and inference.

***You can start the RL training on the single echelon supply chain model with loading the saved DL model with***
``` Bash
python rl_train.py --seed 1 --env-name "single_echelon" --log-dir "./logs_rl/" --save-dir './models_rl/rl_single_echelon.pt' --resume "./models_dl/dl_single_echelon.pt" --sku-mini-batch 25 --num-processes 4 --train-demand-data-pth "./data/df_sales.csv" --train-vlt-data-pth "./data/df_vlt.csv"
```
Here
```
--resume : path of the DL model to be loaded
```
***You can start the RL training on the single echelon supply chain model from scratch with***
``` Bash
python rl_train.py --seed 1 --env-name "single_echelon" --log-dir "./logs_rl/" --save-dir './models_rl/rl_single_echelon.pt' --sku-mini-batch 25 --num-processes 4 --train-demand-data-pth "./data/df_sales.csv" --train-vlt-data-pth "./data/df_vlt.csv"
```
***You can start the RL training on the multi echelon supply chain model from scratch with***
``` Bash
python rl_train.py --seed 1 --env-name "multi_echelon" --log-dir "./logs_rl/" --save-dir './models_rl/rl_multi_echelon.pt' --sku-mini-batch 25 --num-processes 4 --train-demand-data-pth "./data/df_sales.csv" --train-vlt-data-pth "./data/df_vlt.csv"
```
***You can test the saved model with***
``` Bash
python test.py --env-name "single_echelon" --load-dir './models_rl/rl_single_echelon.pt' --test_pics_save_dir './test_pics/' --seed 1
```
Here
```
--load-dir : model to be evaluated
--test_pics_save_dir : path used to save evaluation figures, do not specify this argument if you do not want to genenrate figures
```
