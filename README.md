# GNN_DDI
---
This repository provides a framework of two stages that combines several drug features to predict DDI associated events, using attributed heterogeneous networks and multiple deep neural networks. Firstly, it generates embedding for attributed heterogeneous networks using a GNN model,  next it uses multiple deep neural networks for DDI event prediction. We implement this model based on tensorflow, which enables this model to be trained with GPUs. 
For additional details, read the published article for this work through this [link](https://doi.org/10.1038/s41598-022-19999-4).

## Environments
---
- Python 3.6.12 :: Anaconda.
- Tensorflow 2.1
- keras 2.3
- numpy 1.19
- pandas 1.1.3
- sqlite3 3.8.6
- tqdm 4.63
- matplotlib 3.3

## Usage
---
The data used in this project is derived from the study of [Deng et al.](https://doi.org/10.1093/bioinformatics/btaa501). You can access the data from [GitHub link](https://github.com/YifanDengWHU/DDIMDL).
When using this code, you need to clone this repo and load all the files in the folder into your running environment first. Then, you should enter the root directory and run the following code:
```
    python prepare.py
    python generate_embbeding.py
    python concat.py
    python train.py
```
where the file prepare.py used to prepare data and generate the attributed heterogeneous networks, and generate_embbeding.py is to generate representation vectors of the drugs by using GNN model as representation learning method for attributed heterogeneous networks, and concat.py used to concatenate the embedding vectors, and train.py uses the representation vectors to train this model and predict the DDI and their types.
