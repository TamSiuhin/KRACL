# Contrastive Learning with Graph Context Modeling for Sparse Knowledge Graph Completion

Original implementation for KRACL framework proposed by paper **Contrastive Learning with Graph Context Modeling for Sparse Knowledge Graph Completion**.

## Installation
### Requirements

```python
torch==1.8.0
torch-cluser==1.6.0
torch-scatter==2.0.9
torch-sparse==0.6.121
torch-spline-conv==1.2.1
pytorch-lightning==1.6.1
```

## Run experiment

Please input the following command to reproduce the reported results on FB15k-237 dataset!

```python
CUDA_VISIBLE_DEVICES=0 python noaug_FB_addnoise.py --train_path /path/to/traindata/folder/ --test_path /path/to/testdata/folder/ --ent_num 14541 --rel_num 474 --cl_epochs 1500
```