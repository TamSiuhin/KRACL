# KRACL: Contrastive Learning with Graph Context Modeling for Sparse Knowledge Graph Completion

Original implementation for KRACL framework proposed by paper **KRACL: Contrastive Learning with Graph Context Modeling for Sparse Knowledge Graph Completion**.

Knowledge Graph Embeddings (KGE) aim to map entities and relations to low dimensional spaces and have become the \textit{de-facto} standard for knowledge graph completion. Most existing KGE methods suffer from the sparsity challenge, where it is harder to predict entities that appear less frequently in knowledge graphs. In this work, we propose a novel framework KRACL to alleviate the widespread sparsity in KGs with graph context and contrastive learning. Firstly, we propose the Knowledge Relational Attention Network (KRAT) to leverage the graph context by simultaneously projecting neighboring triples to different latent spaces and jointly aggregating messages with the attention mechanism. KRAT is capable of capturing the subtle semantic information and importance of different context triples as well as leveraging multi-hop information in knowledge graphs. Secondly, we propose the knowledge contrastive loss by combining the contrastive loss with cross entropy loss, which introduces more negative samples and thus enriches the feedback to sparse entities. Our experiments demonstrate that KRACL achieves superior results across various standard knowledge graph benchmarks, especially on WN18RR and NELL-995 which have large numbers of low in-degree entities. Extensive experiments also bear out KRACL's effectiveness in handling sparse knowledge graphs and robustness against noisy triples.

**WORK IN PROGRESS**

## Installation

```python
torch==1.8.0
torch-cluser==1.6.0
torch-scatter==2.0.9
torch-sparse==0.6.12
torch-spline-conv==1.2.1
pytorch-lightning==1.6.1
```

## Run experiment

Please run the following command to reproduce the reported results on FB15k-237 dataset

```python
CUDA_VISIBLE_DEVICES=0 python noaug_FB_addnoise.py --train_path /path/to/traindata/folder/ --test_path /path/to/testdata/folder/ --ent_num 14541 --rel_num 474 --cl_epochs 1500
```
