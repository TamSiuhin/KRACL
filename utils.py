import torch
import numpy as np
import random
import torch.nn as nn
from torch_geometric.utils import subgraph, k_hop_subgraph, sort_edge_index, dropout_adj, add_self_loops, degree, remove_self_loops, to_undirected
from torch.nn import functional as F
from torch_sparse import SparseTensor, coalesce, spspmm
from torch_geometric.transforms import GDC, TwoHop
from torch_scatter import scatter
from torch import Tensor
import torch
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F

class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets:torch.Tensor, n_classes:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                    device=targets.device) \
                .fill_(smoothing /(n_classes-1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
            self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss

class Compose():
    def __init__(self, augmenter_list):
        self.augmentors = augmenter_list
    
    def augment(self, x, edge_index, edge_type, rel_emb, edge_batch_idx=None):
        for aug in self.augmentors:
            x, edge_index, edge_type, rel_emb = aug.augment(x=x, edge_index=edge_index, edge_type=edge_type, rel_emb=rel_emb, edge_batch_idx=edge_batch_idx)

        return x, edge_index, edge_type, rel_emb


class Random_Choice():
    def __init__(self, augmenters, num_choice):
        assert num_choice <= len(augmenters) 
        self.augmenters = augmenters
        self.num_choice = num_choice

    def augment(self, x, edge_index, edge_type, rel_emb, edge_batch_idx=None):
        num_augmentors = len(self.augmenters)
        perm = torch.randperm(num_augmentors)
        idx = perm[:self.num_choice]
        
        for i in idx:
            aug = self.augmenters[i]
            x, edge_index, edge_type, rel_emb = aug.augment(x=x, edge_index=edge_index, edge_type=edge_type, rel_emb=rel_emb, edge_batch_idx=edge_batch_idx)
        return x, edge_index, edge_type, rel_emb


# Mask a fraction of dimension of embedding
class Mask_Emb():
    def __init__(self, p):
        super(Mask_Emb, self).__init__()
        self.p = p

    def augment(self, x, edge_index, edge_type, rel_emb=None, edge_batch_idx=None):
        device = x.device
        drop_mask = torch.empty((x.size(1),), dtype=torch.float32).uniform_(0, 1) < self.p
        drop_mask = drop_mask.to(device)
        x = x.clone()
        x[:, drop_mask] = 0

        return x,  edge_index, edge_type, rel_emb


# Dropout for entity or edge embedding
class Dropout_Emb():
    def __init__(self, p):
        self.p = p
        self.dropout = torch.nn.Dropout(p)

    def augment(self, x, edge_index, edge_type, rel_emb=None, edge_batch_idx=None):
        drop_emb = self.dropout(x)

        if rel_emb != None:
            drop_edge = self.dropout(rel_emb)
        return drop_emb, edge_index, edge_type, drop_edge


# randomly drop edge
class Random_Drop_Edge():
    def __init__(self, p):
        self.p = p

    def augment(self, x, edge_index, edge_type, rel_emb=None, edge_batch_idx=None):
        edge_index, edge_type = dropout_adj(edge_index, edge_type, self.p)
        return x, edge_index, edge_type, rel_emb


# subsampling for a subgraph
class SubGraph():
    def __init__(self, percent=0.8):
        self.p = percent
        
    def augment(self, x, edge_index, edge_type, rel_emb=None, edge_batch_idx=None):
        """return subgraph edge_index, edge_type respectively"""
        mask = torch.rand(edge_index.size(1)) < self.p
        select_node = edge_index[:, mask].flatten()
        edge_index, edge_type = subgraph(subset=select_node, edge_index=edge_index, edge_attr=edge_type)

        return x, edge_index, edge_type, rel_emb


# subsampling a k-hop subgraph
class K_hop_SubGraph():
    def __init__(self, k, p):
        self.k = k
        self.p = p

    def augment(self, x, edge_index, edge_type, rel_emb=None, edge_batch_idx=None):
        
        mask = torch.rand(edge_index.size(1)) < self.p
        select_node = edge_index[:, mask].flatten()
        _, sub_edge_idx, _ , mask =k_hop_subgraph(node_idx=select_node, edge_index=edge_index, num_hops=self.k)
        sub_edge_type = edge_type[mask]

        return x, sub_edge_idx, sub_edge_type, rel_emb


# subgraph induced by randomwalk
class RandomWalk_SubGraph():
    def __init__(self, p, len):
        self.len = len
        self.p = p

    def augment(self, x, edge_index, edge_type, rel_emb=None, edge_batch_idx=None):
        
        edge_index, edge_type = sort_edge_index(edge_index, edge_type)
        mask = torch.rand(edge_index.size(1)) < self.p

        row, col = edge_index
        adj = SparseTensor(row=row, col=col).to(edge_index.device)
        
        start = edge_index[:, mask].flatten()

        node_idx = adj.random_walk(start, self.len).view(-1).to(edge_index.device)

        edge_index, edge_type = subgraph(node_idx, edge_index, edge_type)
        return x, edge_index, edge_type, rel_emb

# randomly add edge
class Random_Add_Edge():
    def __init__(self, p, num_node=40943):
        self.p = p
        self.num_node = num_node

    def augment(self, x, edge_index, edge_type, rel_emb=None, edge_batch_idx=None):
        num_edges = edge_index.size(1)
        num_nodes = self.num_node
        num_add = int(num_edges * self.p)
        num_edge_type = edge_type.max().item() + 1

        new_edge_index = torch.randint(0, num_nodes - 1, size=(2, num_add)).to(edge_index.device)
        new_edge_type = torch.randint(0, num_edge_type-1, size=(num_add,)).to(edge_index.device)
        
        edge_index = torch.cat([edge_index, new_edge_index], dim=1)
        edge_type = torch.cat([edge_type, new_edge_type], dim=0)

        edge_index, edge_type = sort_edge_index(edge_index, edge_type)

        edge_index, edge_type = edge_index_coalesce(edge_index=edge_index, edge_attr=edge_type, num_nodes=num_nodes+num_add, is_sorted=True, reduce="mean")
        return x, edge_index, edge_type.long(), rel_emb

# replace nodes neighbors with their 2-hop neighbors
class Local_Restruct():
    def __init__(self, p, num_node):
        self.p = p
        self.num_nodes = num_node

    def augment(self, x, edge_index, edge_type, rel_emb=None, edge_batch_idx=None):
        
        edge_index, edge_type = sort_edge_index(edge_index, edge_type)
        mask = torch.rand(edge_index.size(1)) < self.p
        add_edge_index = edge_index[:, mask]
        add_edge_type = edge_type[mask]
        N = self.num_nodes
        
        remain_edge_index = edge_index[:, ~mask]
        remain_edge_type = edge_type[~mask]
        
        value = add_edge_index.new_ones((add_edge_index.size(1), ), dtype=torch.float)
        # add_index, value = sort_edge_index(add_edge_index, value)
        
        add_index, value = coalesce(add_edge_index, value, N, N, op="min")
        add_index, value = spspmm(add_index, value, add_index, value, N, N, N)
        value.fill_(0)
        add_index, value = remove_self_loops(add_index, value)
        edge_index = torch.cat([remain_edge_index, add_index], dim=1)
        
        value = value.view(-1, *[1 for _ in range(add_edge_type.dim() - 1)])
        value = value.expand(-1, *list(add_edge_type.size())[1:])
        
        edge_type = torch.cat([remain_edge_type, value], dim=0)
        edge_index, edge_type = sort_edge_index(edge_index, edge_type)
        edge_index, edge_type = edge_index_coalesce(edge_index=edge_index, edge_attr=edge_type, num_nodes=self.num_nodes, is_sorted=True, reduce="max")
        # torch.cuda.empty_cache()
        return x, edge_index, edge_type.long(), rel_emb

# add 2-hop neghbors
class Add_2hop_Neighbors():
    def __init__(self, p, num_node):
        self.p = p
        self.num_node = num_node

    def augment(self, x, edge_index, edge_type, rel_emb=None, edge_batch_idx=None):
        edge_index, edge_type = sort_edge_index(edge_index, edge_type)
        mask = torch.rand(edge_index.size(1)) < self.p
        add_edge_index = edge_index[:, mask]
        add_edge_type = edge_type[mask]
        
        del mask
        N = edge_index.max() + 1
        value = add_edge_index.new_ones((add_edge_index.size(1), ), dtype=torch.float)
        # add_index, value = sort_edge_index(add_edge_index, value)
        

        add_index, value = coalesce(add_edge_index, value, N, N, op="min")
        add_index, value = spspmm(add_index, value, add_index, value, N, N, N)
        value.fill_(0)
        add_index, value = remove_self_loops(add_index, value)
        edge_index = torch.cat([edge_index, add_index], dim=1)
        del add_edge_index
        del add_index

        value = value.view(-1, *[1 for _ in range(add_edge_type.dim() - 1)])
        value = value.expand(-1, *list(add_edge_type.size())[1:])
        
        edge_type = torch.cat([edge_type, value], dim=0)
        del value, add_edge_type

        edge_index, edge_type = sort_edge_index(edge_index, edge_type)
        edge_index, edge_type = edge_index_coalesce(edge_index=edge_index, edge_attr=edge_type, num_nodes=self.num_node, is_sorted=True, reduce="max")
        # torch.cuda.empty_cache()
        return x, edge_index, edge_type.long(), rel_emb


# local path ablation
class Path_Ablation():
    def __init__(self, p, num_hop):
        self.p = p
        self.num_hop = num_hop

    def augment(self, x, edge_index, edge_type, rel_emb=None, edge_batch_idx=None):
        num_ent = edge_index.max()+1
        mask = torch.rand(edge_index.size(1)) < self.p
        select_edge_index = edge_index[:, mask]
        select_edge_type = edge_type[mask]
    
        for _ in range(self.num_hop - 1):
            N = select_edge_index.max() + 1
            value = select_edge_index.new_ones((select_edge_index.size(1), ), dtype=torch.float)
            select_edge_index, value = spspmm(select_edge_index, value, select_edge_index, value, N, N, N, coalesced=True)
            value.fill_(0)
            select_edge_index, value = remove_self_loops(select_edge_index, value)

        combined = torch.cat((select_edge_index, edge_index))
        uniques, inverse_index, counts = combined.unique(return_counts=True, dim=0, return_inverse=True)
        edge_index = uniques[counts == 1]
        edge_type = edge_type[inverse_index[: select_edge_index.size(1)]]

        return x, edge_index, edge_type, rel_emb

    def augment_v1(self, x, edge_index, edge_type, rel_emb=None, edge_batch_idx=None):
        edge_num = edge_index.size(1)
        num_ent = edge_index.max()+1
        edge_idx = torch.randperm(edge_num)[0: int(edge_num * self.p)].cuda()
        node_idx = edge_index[0, edge_idx]
        src, dst = edge_index

        node_mask = dst.new_empty(num_ent, dtype=torch.bool)
        edge_mask = dst.new_empty(dst.size(0), dtype=torch.bool)
        
        node_subsets = [node_idx]
        edge_type_subsets = [edge_type[edge_idx]]
        edge_subsets = [edge_index[:, edge_idx]]

        for _ in range(self.num_hop):
            node_mask.fill_(False)
            node_mask[node_subsets[-1]] = True
            torch.index_select(node_mask, 0, src, out=edge_mask)
            node_subsets.append(dst[edge_mask])
            edge_type_subsets.append(edge_type[edge_mask])
            edge_subsets.append(edge_index[:, edge_mask])

        src_mask = dst.new_empty(dst.size(0), dtype=torch.bool)
        dst_mask = dst.new_empty(dst.size(0), dtype=torch.bool)
        
        # initialize
        src_mask.fill_(False)
        dst_mask.fill_(False)
        
        begin = node_subsets[0]
        end = node_subsets[-1]
        
        node_mask.fill_(False)
        node_mask[begin] = True
        torch.index_select(node_mask, 0, src, out=src_mask)
        
        node_mask.fill_(False)
        node_mask[end] = True
        torch.index_select(node_mask, 0, dst, out=dst_mask)
        
        edge_mask = src_mask & dst_mask

        edge_index = edge_index[:, ~edge_mask]
        edge_type = edge_type[~edge_mask]

        # src_mask[begin] = True
        # dst_mask[end] = True
        
        # edge_mask = src
        # delete_src = []
        # delete_dst = []
        # for i in end:
        #     in_neighbor, out_neighbor = find_neghbors(i, edge_index)
        #     for j in np.intersect1d(in_neighbor, begin):
        #         delete_src.append(j)
        #         delete_dst.append(i)

        # src = np.setdiff1d(src, torch.tensor(delete_src))
        # dst = np.setdiff1d(dst, torch.tensor(delete_dst))
        # edge_index = torch.stack((src, dst), dim=0)
        
        return x, edge_index, edge_type, rel_emb

# remove all reverse triples in KG
class Remove_Inverse_Triples():
    def __init__(self, p):
        self.p = p

    def augment(self, x, edge_index, edge_type, rel_emb=None, edge_batch_idx=None):
        edge_num = edge_index.size(1)//2
        mask = torch.rand(edge_index.size(1)) < self.p

        select_edge_index = edge_index[:, mask]
        select_edge_type = edge_type[mask]

        remain_edge_index = edge_index[:, ~mask]
        remain_edge_type = edge_type[~mask]

        edge_index = torch.cat((remain_edge_index, select_edge_index), dim=1)
        edge_type = torch.cat((remain_edge_type, select_edge_type), dim=0)

        return x, edge_index, edge_type, rel_emb

# return the original graph
class Identity():
    def __init__(self):
        super(Identity, self).__init__()

    def augment(self, x, edge_index, edge_type, rel_emb = None, edge_batch_idx=None):
        return x, edge_index, edge_type, rel_emb


def find_neghbors(index, edge_index):
    flow_in_mask = edge_index[1]==index
    flow_out_mask = edge_index[0]==index

    flow_in_nei = edge_index[0, flow_in_mask]
    flow_out_nei = edge_index[1, flow_out_mask]

    return flow_in_nei, flow_out_nei

# Personalized PageRank Diffusion
def compute_ppr(edge_index, edge_weight=None, alpha=0.2, eps=0.1, ignore_edge_attr=True, add_self_loop=True):
    N = edge_index.max().item() + 1
    if ignore_edge_attr or edge_weight is None:
        edge_weight = torch.ones(
            edge_index.size(1), device=edge_index.device)
    if add_self_loop:
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, fill_value=1, num_nodes=N)
        edge_index, edge_weight = coalesce(edge_index, edge_weight, N, N)
    edge_index, edge_weight = coalesce(edge_index, edge_weight, N, N)
    edge_index, edge_weight = GDC().transition_matrix(
        edge_index, edge_weight, N, normalization='sym')
    diff_mat = GDC().diffusion_matrix_exact(
        edge_index, edge_weight, N, method='ppr', alpha=alpha)
    edge_index, edge_weight = GDC().sparsify_dense(diff_mat, method='threshold', eps=eps)
    edge_index, edge_weight = coalesce(edge_index, edge_weight, N, N)
    edge_index, edge_weight = GDC().transition_matrix(
        edge_index, edge_weight, N, normalization='sym')

    return edge_index, edge_weight


# drop with pagerank weight 
def pr_drop_weights(edge_index, aggr: str = 'sink', k: int = 10):
    pv = compute_pr(edge_index, k=k)
    pv_row = pv[edge_index[0]].to(torch.float32)
    pv_col = pv[edge_index[1]].to(torch.float32)
    s_row = torch.log(pv_row)
    s_col = torch.log(pv_col)
    if aggr == 'sink':
        s = s_col
    elif aggr == 'source':
        s = s_row
    elif aggr == 'mean':
        s = (s_col + s_row) * 0.5
    else:
        s = s_col
    weights = (s.max() - s) / (s.max() - s.mean())

    return weights

def compute_pr(edge_index, damp: float = 0.85, k: int = 10):
    num_nodes = edge_index.max().item() + 1
    deg_out = degree(edge_index[0])
    x = torch.ones((num_nodes, )).to(edge_index.device).to(torch.float32)

    for i in range(k):
        edge_msg = x[edge_index[0]] / deg_out[edge_index[0]]
        agg_msg = scatter(edge_msg, edge_index[1], reduce='sum')

        x = (1 - damp) * x + damp * agg_msg

    return x


def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))
        
def edge_index_coalesce(edge_index, edge_attr, num_nodes, reduce="add", is_sorted=False, sort_by_row=True,):
    """Row-wise sorts :obj:`edge_index` and removes its duplicated entries.
    Duplicate entries in :obj:`edge_attr` are merged by scattering them
    together according to the given :obj:`reduce` option.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor or List[Tensor], optional): Edge weights or multi-
            dimensional edge features.
            If given as a list, will re-shuffle and remove duplicates for all
            its entries. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        reduce (string, optional): The reduce operation to use for merging edge
            features (:obj:`"add"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"mul"`). (default: :obj:`"add"`)
        is_sorted (bool, optional): If set to :obj:`True`, will expect
            :obj:`edge_index` to be already sorted row-wise.
        sort_by_row (bool, optional): If set to :obj:`False`, will sort
            :obj:`edge_index` column-wise.

    :rtype: :class:`LongTensor` if :attr:`edge_attr` is :obj:`None`, else
        (:class:`LongTensor`, :obj:`Tensor` or :obj:`List[Tensor]]`)
    """
    nnz = edge_index.size(1)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    idx = edge_index.new_empty(nnz + 1)
    idx[0] = -1
    idx[1:] = edge_index[1 - int(sort_by_row)]
    idx[1:].mul_(num_nodes).add_(edge_index[int(sort_by_row)])

    if not is_sorted:
        idx[1:], perm = idx[1:].sort()
        edge_index = edge_index[:, perm]
        if edge_attr is not None and isinstance(edge_attr, Tensor):
            edge_attr = edge_attr[perm]
        elif edge_attr is not None:
            edge_attr = [e[perm] for e in edge_attr]

    mask = idx[1:] > idx[:-1]

    # Only perform expensive merging in case there exists duplicates:
    if mask.all():
        return edge_index if edge_attr is None else (edge_index, edge_attr)

    edge_index = edge_index[:, mask]
    if edge_attr is None:
        return edge_index

    dim_size = edge_index.size(1)
    idx = torch.arange(0, nnz, device=edge_index.device)
    idx.sub_(mask.logical_not_().cumsum(dim=0))

    if isinstance(edge_attr, Tensor):
        edge_attr = scatter(edge_attr, idx, 0, None, dim_size, reduce)
    else:
        edge_attr = [scatter(e, idx, 0, None, dim_size, reduce) for e in edge_attr]

    return edge_index, edge_attr

# create inverse triples
def add_inverse_triples(triples):
    h, r, t = triples.t()
    r = r * 2 
    data = torch.cat([torch.stack([h, r, t], dim=1), torch.stack([t, r+1, h], dim=1)], dim=0)
    return data

def com_mult_new(a, b):
    r1, i1 = a.real, a.imag
    r2, i2 = b.real, b.imag
    real = r1 * r2 - i1 * i2
    imag = r1 * i2 + i1 * r2
    return torch.complex(real, imag)

def conj_new(a):
	a.imag = -a.imag
	return a

def cconv_new(a, b):
	return torch.fft.irfft(com_mult_new(torch.fft.rfft(a.float()), torch.fft.rfft(b.float())))#.half()

def ccorr_new(a, b):
	return torch.fft.irfft(com_mult_new(conj_new(torch.fft.rfft(a.float())), torch.fft.rfft(b.float())))#.half()


def com_mult(a, b):
    r1, i1 = a[..., 0], a[..., 1]
    r2, i2 = b[..., 0], b[..., 1]
    return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim = -1)

def conj(a):
	a[..., 1] = -a[..., 1]
	return a

def cconv(a, b):
	return torch.irfft(com_mult(torch.rfft(a, 1), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))

def ccorr(a, b):
	return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))

def rotate(node, edge):
    node_re, node_im = node.chunk(2, dim=-1)
    edge_re, edge_im = edge.chunk(2, dim=-1)
    message_re = node_re * edge_re - node_im * edge_im
    message_im = node_re * edge_im + node_im * edge_re
    message = torch.cat([message_re, message_im], dim=-1)
    return message
    
def add_noise(valid, noise):
    valid_num = valid.size(0)
    noise_num = noise.size(0)
    valid_in, valid_out = valid[:valid_num], valid[valid_num:]
    noise_in, noise_out = noise[:noise_num], noise[noise_num:]

    triple_in = torch.cat((valid_in, noise_in), dim=0)
    triple_out = torch.cat((valid_out, noise_out), dim=0)

    triples = torch.cat((triple_in, triple_out), dim=0)
    head, rel, tail = triples.t()
    return triples, head, rel, tail