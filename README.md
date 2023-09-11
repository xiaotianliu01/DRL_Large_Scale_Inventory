# DRL_Large_Scale_Inventory
Official codes for ["Deep Reinforcement Learning for Large-Scale Inventory Management"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4490327)

## Create Environment
The codes are implementable on both Windows and Linux with Python 3.6.5 and pytorch.

## How To Run
When your environment is ready, you can start the DL training on with
``` Bash
python dl_train.py --seed 1 --env-name "single_echelon" --log-dir "./logs_dl/" --save-dir './models_dl/dl_single_echelon.pt' --sku-mini-batch 40 --num-processes 1 --train-demand-data-pth "./data/df_sales.csv" --train-vlt-data-pth "./data/df_vlt.csv"
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
Note params *sku-mini-batch* and *num-processes* come from the parrel deisign in the codes which are meant to improve the code effeciency with a large volume of SKUs. You have to make sure that **sku-mini-batch*num-processes>=number of total SKUs in your dataset**. Otherwise, some of your SKUs may not be used during training.
