import torch
from torch.functional import Tensor
import torch.nn as nn
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.nn import RGCNConv, FastRGCNConv
from utils import cconv, cconv_new, ccorr, ccorr_new, rotate
from torch_scatter import scatter_add
import math
from torch.nn import ModuleList, Sequential

class GateRGCN(nn.Module):
    def __init__(self, in_dim, out_dim, num_rel, dropout=0):
        super(GateRGCN, self).__init__()
        self.RGCN = RGCNConv(in_channels=in_dim, out_channels=out_dim, num_relations=num_rel)
        self.W = torch.nn.Linear(in_dim+out_dim, out_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_type):
        u = self.RGCN(x, edge_index, edge_type)
        u = self.dropout(u)
        z = self.W(torch.cat((u, x), dim=1))
        h = torch.mul(torch.tanh(u), z) + torch.mul(x, (1-z))
        return h

class FastGRGCN(nn.Module):
    def __init__(self, in_dim, out_dim, num_rel, dropout=0):
        super(FastGRGCN, self).__init__()
        self.RGCN = FastRGCNConv(in_channels=in_dim, out_channels=out_dim, num_relations=num_rel)
        self.W = torch.nn.Linear(in_dim+out_dim, out_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_type):
        u = self.RGCN(x, edge_index, edge_type)
        u = self.dropout(u)
        z = self.W(torch.cat((u, x), dim=1))
        h = torch.mul(torch.tanh(u), z) + torch.mul(x, (1-z))
        return h


# KBGAT without considering 2-hop neighbours！
class Simple_KBGAT(MessagePassing):
    def __init__(self, input_dim, rel_dim, output_dim, dropout=0, num_head=1, final_layer=False):
        super(Simple_KBGAT, self).__init__(aggr = "add", node_dim=0)
        self.input_dim = input_dim
        self.out_dim = output_dim
        self.W_r = nn.Linear(rel_dim, rel_dim, bias=False)
        self.w_1 = nn.Linear(2 * self.input_dim + rel_dim, num_head*self.out_dim, bias=False)
        self.w_2 = nn.Linear(self.out_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU()
        self.elu = nn.ELU()
        self.final_layer = final_layer
        self.num_head = num_head
        self.dropout = dropout

        torch.nn.init.xavier_uniform_(self.W_r.weight.data)
        torch.nn.init.xavier_uniform_(self.w_1.weight.data)
        torch.nn.init.xavier_uniform_(self.w_2.weight.data)

    def forward(self, x, relation_embedding, edge_index, edge_type, edge_weight=None, size=None):
        # should we add the initial embedding in the final layer?
        
        node_emb = self.propagate(x=x, edge_index=edge_index, edge_type=edge_type, edge_weight=edge_weight,\
            relation_embedding=relation_embedding)
        
        if self.final_layer:
            # mean
            node_emb = self.elu(node_emb.mean(dim=1))
        else:
            # concat
            node_emb = self.leaky_relu(node_emb).view(-1, self.num_head * self.out_dim)
        
        rel_emb = self.elu(self.W_r(relation_embedding))
        return node_emb, rel_emb

    def message(self, x_i, x_j, index, ptr, size_i, edge_type, relation_embedding, edge_weight):
        
        edge_emb = torch.index_select(relation_embedding, 0, edge_type)
        triple_emb = torch.cat((x_i, x_j, edge_emb), dim=1).cuda()
        c = self.w_1(triple_emb).view(-1, self.num_head, self.out_dim)

        b = self.leaky_relu(self.w_2(c)).view(-1, self.num_head, 1)
        alpha = softmax(b, index, ptr, size_i)
        
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        out = c * alpha.view(-1, self.num_head, 1)
        
        if edge_weight == None:
            return out
        else:
            out = out * edge_weight.view(-1, 1)
            return out


class TransGAT(MessagePassing):
    def __init__(self, input_dim, output_dim, dropout=0, final_layer=False, in_out=False):
        super(TransGAT, self).__init__(aggr = "add", node_dim=0)

        self.input_dim = input_dim
        self.out_dim = output_dim
        self.W_r = nn.Linear(self.input_dim, output_dim, bias=False)
        self.w_1 = nn.Linear(3 * self.input_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU()
        self.activation = nn.ReLU()
        self.final_layer = final_layer
        self.dropout = dropout
        self.in_out = in_out

        if in_out:
            self.W_in = nn.Linear(input_dim, output_dim, bias=False)
            self.W_out = nn.Linear(input_dim, output_dim, bias=False)

            torch.nn.init.xavier_uniform_(self.W_in.weight.data)
            torch.nn.init.xavier_uniform_(self.W_out.weight.data)
        else:
            self.W_2 = nn.Linear(input_dim, output_dim, bias=False)
            torch.nn.init.xavier_uniform_(self.W_2.weight.data)

    
    def forward(self, x, rel_embed, edge_index, edge_type, edge_weight):
        # NOTICE: edge_index should be bidirectional

        # normalization
        # row, col = edge_index
        # in_deg = degree(col, x.size(0), dtype=x.dtype)
        # out_deg = degree(row, x.size(0), dtype=x.dtype)
        # deg_inv = (in_deg + out_deg).pow(-1)
        # deg_inv[deg_inv == float("inf")] = 0
        # deg_inv = deg_inv.unsqueeze(dim=1)

        # compute node_embedding
        self.num_edges = edge_index.size(1)
        num_ent = x.size(0)

        self.edge_index = torch.cat((edge_index, torch.stack((edge_index[1], edge_index[0]), dim=0)), dim=1)
        self.edge_type = torch.cat((edge_type, edge_type))
        
        node_emb = self.propagate(edge_index=self.edge_index, x=x, edge_type=self.edge_type, rel_embed=rel_embed, edge_weight=edge_weight)
        
        # ego embedding
        node_out = self.leaky_relu(node_emb + x)
        # update relation embedding
        edge_emb = self.leaky_relu(self.W_r(rel_embed))

        return node_out, edge_emb

    def message(self, x_i, x_j, edge_type, rel_embed, index, ptr, size_i, edge_weight):

        edge_emb = torch.index_select(rel_embed, 0, edge_type)
        triple_emb = torch.cat((x_i, x_j, edge_emb), dim= 1).cuda()
        b = self.leaky_relu(self.w_1(triple_emb))
        alpha = softmax(b, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        x_j_in = x_j[:self.num_edges, :]
        x_j_out = x_j[self.num_edges:, :]
        
        alpha_in = alpha[:self.num_edges, :]
        alpha_out = alpha[self.num_edges:, :]
        
        if self.in_out:
            trans_emb_in = self.W_in(x_j_in + edge_emb[:self.num_edges, :])
            trans_emb_out = self.W_out(x_j_out - edge_emb[self.num_edges:, :])

        else:
            trans_emb_in = self.W_2(x_j_in + edge_emb[:self.num_edges, :])
            trans_emb_out = self.W_2(x_j_out - edge_emb[self.num_edges:, :])
        
        trans_emb_in *= alpha_in.view(-1, 1)
        trans_emb_out *= alpha_out.view(-1, 1)

        if edge_weight != None: 
            trans_emb_in *= edge_weight.view(-1, 1)
            trans_emb_out *= edge_weight.view(-1, 1)

        trans_emb = torch.cat((trans_emb_in, trans_emb_out), dim=0)

        return trans_emb
        
    def update(self, aggr_out):
        return aggr_out


class TransGCN(MessagePassing):
    def __init__(self, input_dim, output_dim):
        super(TransGCN, self).__init__(aggr = "add", node_dim=0)
        self.input_dim = input_dim
        self.out_dim = output_dim
        # self.num_rel = num_rels
        self.W_o = nn.Linear(input_dim, output_dim, bias=False)
        self.W_r = nn.Linear(input_dim, output_dim, bias=False)
        self.activation = nn.ReLU()
        
        torch.nn.init.kaiming_uniform_(self.W_o.weight)
        torch.nn.init.kaiming_uniform_(self.W_r.weight)

    def forward(self, x, rel_embed, edge_index, edge_type, edge_weight):
        # NOTICE: edge_index should be bidirectional
        # front half should be flow-in edges, and the latter half are flow-out edges

        # normalization
        row, col = edge_index
        in_deg = degree(col, x.size(0), dtype=x.dtype)
        out_deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv = (in_deg + out_deg).pow(-1)
        deg_inv[deg_inv == float("inf")] = 0
        deg_inv = deg_inv.unsqueeze(dim=1)

        # compute node_embedding
        # num_edges = edge_index.size(1) // 2
        # num_ent = x.size(0)
        # self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges]
        # self.in_type, self.out_type = edge_type[:num_edges], edge_type[num_edges:]

        self.in_index = edge_index
        self.out_index = torch.stack([edge_index[1], edge_index[0]], dim = 0)
        self.in_type = edge_type
        self.out_type = edge_type
        
        in_emb = self.propagate(edge_index=self.in_index, x=x, edge_type=self.in_type, rel_embed=rel_embed, edge_weight=edge_weight, mode="in")
        out_emb = self.propagate(edge_index=self.out_index, x=x, edge_type=self.out_type, rel_embed=rel_embed, edge_weight=edge_weight, mode="out")
        node_emb = torch.mul(self.W_o(in_emb + out_emb), deg_inv)
        # ego embedding
        node_out = self.activation(node_emb + x)
        # update relation embedding
        edge_emb = self.activation(self.W_r(rel_embed))

        return node_out, edge_emb

    def message(self, x_i, x_j, edge_type, rel_embed, mode, edge_weight):
        edge_emb = torch.index_select(rel_embed, 0, edge_type)
        if mode == "in":
            trans_emb = x_j + edge_emb
        if mode == "out":
            trans_emb = x_j - edge_emb
        
        if edge_weight == None:
            return trans_emb
        else:
            return trans_emb * edge_weight.view(-1, 1)

    def update(self, aggr_out):
        return aggr_out
    

# KBGAT without considering 2-hop neighbours！
class KBGAT_conv(MessagePassing):
    def __init__(self, in_channel, out_channel, rel_dim, dropout=0, final_layer=False):
        super(KBGAT_conv, self).__init__(aggr = "add", node_dim=0)
        self.ent_input_dim = in_channel
        self.out_dim = out_channel
        self.rel_dim = rel_dim
        self.w_1 = nn.Linear(2 * self.ent_input_dim + rel_dim, self.out_dim, bias=True)
        self.w_2 = nn.Linear(self.out_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU()
        self.elu = nn.ELU()
        self.final_layer = final_layer
        # self.num_head = num_head
        self.dropout = dropout

        torch.nn.init.xavier_uniform_(self.w_1.weight.data)
        torch.nn.init.xavier_uniform_(self.w_2.weight.data)

    def forward(self, x, relation_embedding, edge_index, edge_type, edge_weight=None):
        
        node_emb = self.propagate(x=x, edge_index=edge_index, edge_type=edge_type, edge_weight=edge_weight,\
            relation_embedding=relation_embedding)
        
        if self.final_layer:
            # mean
            node_emb = self.elu(node_emb.mean(dim=1))

        else:
            # concat
            node_emb = self.leaky_relu(node_emb)#.view(-1, self.num_head * self.out_dim)
        
        return node_emb

    def message(self, x_i, x_j, index, ptr, size_i, edge_type, relation_embedding, edge_weight):
        
        edge_emb = torch.index_select(relation_embedding, 0, edge_type)
        triple_emb = torch.cat((x_i, x_j, edge_emb), dim=1).cuda()
        c = self.w_1(triple_emb) #.view(-1, self.num_head, self.out_dim)

        b = self.leaky_relu(self.w_2(c))#.view(-1, self.num_head, 1)
        alpha = softmax(b, index, ptr, size_i)
        
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = c * alpha.view(-1, 1)

        if edge_weight == None:
            return out
        else:
            out = out * edge_weight.view(-1, 1)
            return out
        
    def update(self, aggr_out):
        return aggr_out

class CompGCN(MessagePassing):
    def __init__(self, in_channels, out_channels, drop, bias, op):
        super(CompGCN, self).__init__(aggr = "add", node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.op = op
        self.bias = bias

        self.w_loop = torch.nn.Linear(in_channels, out_channels, bias=False).cuda()
        self.w_in = torch.nn.Linear(in_channels, out_channels, bias=False).cuda()
        self.w_out = torch.nn.Linear(in_channels, out_channels, bias=False).cuda()
        self.w_rel = torch.nn.Linear(in_channels, out_channels, bias=False).cuda()
        self.loop_rel = torch.nn.Parameter(torch.Tensor(1, in_channels))
        torch.nn.init.xavier_normal_(self.loop_rel)

        self.drop = torch.nn.Dropout(drop)
        self.bn = torch.nn.BatchNorm1d(out_channels).to(torch.device("cuda"))
        self.activation = torch.nn.Tanh()

        if bias:
            self.register_parameter("bias_value", torch.nn.Parameter(torch.zeros(out_channels)))
    
    def forward(self, x, edge_index, edge_type, rel_emb):

        rel_emb = torch.cat([rel_emb, self.loop_rel], dim=0)
        num_edge = edge_index.size(1)//2
        num_ent = x.size(0)

        self.in_index, self.out_index = edge_index[:,:num_edge], edge_index[:,num_edge:]
        self.in_type, self.out_type = edge_type[:num_edge], edge_type[num_edge:]

        self.loop_index = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).cuda()
        self.loop_type = torch.full((num_ent,), rel_emb.size(0)-1, dtype=torch.long).cuda()
        
        self.in_norm = self.compute_norm(self.in_index, num_ent)
        self.out_norm = self.compute_norm(self.out_index, num_ent)

        in_res = self.propagate(edge_index=self.in_index, x=x, edge_type=self.in_type, rel_emb=rel_emb, edge_norm=self.in_norm, mode="in")
        loop_res = self.propagate(edge_index=self.loop_index, x=x, edge_type=self.loop_type, rel_emb=rel_emb, edge_norm=None, mode="loop")
        out_res = self.propagate(edge_index=self.out_index, x=x, edge_type=self.out_type, rel_emb=rel_emb, edge_norm=self.out_norm, mode="out")

        out = self.drop(in_res)*(1/3) + loop_res*(1/3) + self.drop(out_res)*(1/3)

        if self.bias:
            out = out + self.bias_value
        out = self.bn(out)
        out = self.activation(out)    
        return out, self.w_rel(rel_emb)[:-1]

    def message(self, x_j, edge_type, rel_emb, edge_norm, mode):
        
        rel_emb = torch.index_select(rel_emb, 0, edge_type)
        xj_rel = self.rel_transform(x_j, rel_emb)
        
        if mode == "in":
            out = self.w_in(xj_rel)
        if mode == "out":
            out = self.w_out(xj_rel)
        if mode == "loop":
            out = self.w_loop(xj_rel)

        if edge_norm==None:
            return out
        else:
            return out * edge_norm.view(-1, 1)

    def compute_norm(self, edge_index, num_ent):
        
        row, col=edge_index
        edge_weight= torch.ones_like(row).float()
        deg =scatter_add( edge_weight, row, dim=0, dim_size=num_ent)    # Summing number of weights of the edges
        deg_inv	= deg.pow(-0.5) # D^{-0.5}
        deg_inv[deg_inv	== float('inf')] = 0
        norm= deg_inv[row] * edge_weight * deg_inv[col] # D^{-0.5}
        
        return norm

    def update(self, aggr_out):
        return aggr_out

    def rel_transform(self, ent_embed, rel_emb):
        
        if self.op == 'corr':
            trans_embed  = ccorr(ent_embed, rel_emb)
        elif self.op == 'sub':
            trans_embed = ent_embed - rel_emb
        elif self.op == 'mult':
            trans_embed = ent_embed * rel_emb
        elif self.op == "corr_new":
            trans_embed = ccorr_new(ent_embed, rel_emb)
        elif self.op == "conv":
            trans_embed = cconv(ent_embed, rel_emb)
        elif self.op == "conv_new":
            trans_embed = cconv_new(ent_embed, rel_emb)
        else:
            raise NotImplementedError
        
        return trans_embed

class CompGAT(MessagePassing):
    def __init__(self, in_channels, out_channels, rel_dim, drop, bias, op):
        super(CompGAT, self).__init__(aggr = "add", node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.op = op
        self.bias = bias

        self.w_loop = torch.nn.Linear(in_channels, out_channels, bias=bias).cuda()
        self.w1 = torch.nn.Linear(in_channels, out_channels, bias=bias).cuda()
        self.w_rel = torch.nn.Linear(in_channels, out_channels, bias=bias).cuda()
        self.w_att = torch.nn.Linear(3*in_channels, 1).cuda()
        self.loop_rel = torch.nn.Parameter(torch.Tensor(1, in_channels)).cuda()
        torch.nn.init.xavier_uniform_(self.loop_rel)
        
        self.drop_ratio = drop
        self.drop = torch.nn.Dropout(drop, inplace=False)
        self.bn = torch.nn.BatchNorm1d(out_channels).to(torch.device("cuda"))
        
        self.activation = torch.nn.Tanh()
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if bias:
            self.register_parameter("bias_value", torch.nn.Parameter(torch.zeros(out_channels)))
        
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    
    def forward(self, x, edge_index, edge_type, rel_emb, edge_weight=None):
        rel_emb = torch.cat([rel_emb, self.loop_rel], dim=0)
        
        num_ent = x.size(0)
        loop_index = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).cuda()
        loop_type   = torch.full((num_ent,), rel_emb.size(0)-1, dtype=torch.long).cuda()

        in_res = self.propagate(edge_index=edge_index, x=x, edge_type=edge_type, rel_emb=rel_emb, edge_weight=edge_weight, mode="in")
        loop_res = self.propagate(edge_index=loop_index, x=x, edge_type=loop_type, rel_emb=rel_emb, edge_weight=edge_weight, mode="loop")

        out = self.drop(in_res) + self.drop(loop_res)

        if self.bias:
            out = out + self.bias_value
        
        out = self.bn(out)
        out = self.activation(out)

        return out, self.w_rel(rel_emb)[:-1]

    def message(self,x_i, x_j, edge_type, rel_emb, ptr, index, size_i, mode, edge_weight):
        
        rel_emb = torch.index_select(rel_emb, 0, edge_type)
        xj_rel = self.rel_transform(x_j, rel_emb)
        
        if mode == "in":
            out = self.w1(xj_rel)
            b = self.leaky_relu(self.w_att(torch.cat((x_i, rel_emb, x_j), dim=1))).cuda()
            alpha = softmax(b, index, ptr, size_i)
            alpha = F.dropout(alpha, p=self.drop_ratio, training=self.training, inplace=False)
            
            if edge_weight!=None:
                out = out * alpha.view(-1,1) * edge_weight.view(-1,1)
            else:
                out = out * alpha.view(-1,1)

        if mode == "loop":
            out = self.w_loop(xj_rel)

        return out

    def update(self, aggr_out):
        return aggr_out

    def rel_transform(self, ent_embed, rel_emb):
        
        if self.op == 'corr':
            trans_embed  = ccorr(ent_embed, rel_emb)
        elif self.op == 'sub':
            trans_embed = ent_embed - rel_emb
        elif self.op == 'mult':
            trans_embed = ent_embed * rel_emb
        elif self.op == "corr_new":
            trans_embed = ccorr_new(ent_embed, rel_emb)
        elif self.op == "conv":
            trans_embed = cconv(ent_embed, rel_emb)
        elif self.op == "conv_new":
            trans_embed = cconv_new(ent_embed, rel_emb)
        else:
            raise NotImplementedError
        
        return trans_embed

class ARGAT(MessagePassing):
    def __init__(self, in_channels, out_channels, rel_dim, drop, bias, op):
        super(ARGAT, self).__init__(aggr = "add", node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.op = op
        self.bias = bias

        self.w = torch.nn.Linear(in_channels, out_channels, bias=bias).cuda()
        self.w_rel = torch.nn.Linear(in_channels, out_channels, bias=bias).cuda()
        self.w_att = torch.nn.Linear(2*out_channels + rel_dim, 1).cuda()
        self.loop_rel = torch.nn.Parameter(torch.Tensor(1, in_channels)).cuda()
        torch.nn.init.xavier_normal_(self.loop_rel)
        
        self.drop_ratio = drop
        self.drop = torch.nn.Dropout(drop, inplace=True)
        self.bn = torch.nn.BatchNorm1d(out_channels).to(torch.device("cuda"))
        
        self.activation = torch.nn.Tanh()
        self.leaky_relu = torch.nn.LeakyReLU()

        if bias:
            self.register_parameter("bias_value", torch.nn.Parameter(torch.zeros(out_channels)))
    
    def forward(self, x, edge_index, edge_type, rel_emb, edge_weight=None):
        rel_emb = torch.cat([rel_emb, self.loop_rel], dim=0)
        
        num_ent = x.size(0)
        loop_index = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).cuda()
        loop_type   = torch.full((num_ent,), rel_emb.size(0)-1, dtype=torch.long).cuda()
        edge_index = torch.cat((edge_index, loop_index), dim=1)
        edge_type = torch.cat((edge_type, loop_type), dim=0)

        edge_norm = self.compute_norm(edge_index, num_ent)

        out = self.propagate(edge_index=edge_index, x=x, edge_type=edge_type, rel_emb=rel_emb, edge_weight=edge_weight, edge_norm=edge_norm)

        out = self.drop(out)

        if self.bias:
            out = out + self.bias_value
        
        out = self.activation(out)    
        out = self.bn(out)
        return out, self.w_rel(rel_emb)[:-1]

    def message(self, x_i, x_j, edge_type, rel_emb, ptr, index, size_i, edge_weight, edge_norm):
        rel_emb = torch.index_select(rel_emb, 0, edge_type)
        xj_rel = self.rel_transform(x_j, rel_emb) 
        xj_rel = self.w(xj_rel)
        
        b = self.leaky_relu(self.w_att(torch.cat((x_i, rel_emb, x_j), dim=1))).cuda()
        alpha = softmax(b, index, ptr, size_i)
        # alpha = F.dropout(alpha, p=self.drop_ratio, training=self.training)
        
        # print("xj_rel: {}".format(xj_rel.size()))
        # print("edge_norm: {}".format(edge_norm.size()))

        if edge_weight!=None:
            out = xj_rel * alpha.view(-1,1) * edge_weight.view(-1,1) * edge_norm.view(-1,1)
        else:
            out = xj_rel * alpha.view(-1,1) * edge_norm.view(-1,1)

        return out
    
    def compute_norm(self, edge_index, num_ent):
        
        row, col=edge_index
        edge_weight= torch.ones_like(row).float()
        deg =scatter_add( edge_weight, row, dim=0, dim_size=num_ent)    # Summing number of weights of the edges
        deg_inv	= deg.pow(-0.5) # D^{-0.5}
        deg_inv[deg_inv	== float('inf')] = 0
        norm= deg_inv[row] * edge_weight * deg_inv[col] # D^{-0.5}
        
        return norm

    def update(self, aggr_out):
        return aggr_out

    def rel_transform(self, ent_embed, rel_emb):
        
        if self.op == 'corr':
            trans_embed  = ccorr(ent_embed, rel_emb)
        elif self.op == 'sub':
            trans_embed = ent_embed - rel_emb
        elif self.op == 'mult':
            trans_embed = ent_embed * rel_emb
        elif self.op == "corr_new":
            trans_embed = ccorr_new(ent_embed, rel_emb)
        elif self.op == "conv":
            trans_embed = cconv(ent_embed, rel_emb)
        elif self.op == "conv_new":
            trans_embed = cconv_new(ent_embed, rel_emb)
        else:
            raise NotImplementedError
        
        return trans_embed

class CompGATv2(MessagePassing):
    def __init__(self, in_channels, out_channels, rel_dim, drop, bias, op):
        super(CompGATv2, self).__init__(aggr = "add", node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.op = op
        self.bias = bias

        self.w_loop = torch.nn.Linear(in_channels, out_channels).cuda()
        self.w1 = torch.nn.Linear(in_channels, out_channels).cuda()
        self.w_rel = torch.nn.Linear(in_channels, out_channels).cuda()
        self.w_att = torch.nn.Linear(3*in_channels, out_channels).cuda()
        self.a = torch.nn.Linear(out_channels, 1, bias=False).cuda()
        self.loop_rel = torch.nn.Parameter(torch.Tensor(1, in_channels)).cuda()
        torch.nn.init.xavier_uniform_(self.loop_rel)
        
        self.drop_ratio = drop
        self.drop = torch.nn.Dropout(drop, inplace=False)
        self.bn = torch.nn.BatchNorm1d(out_channels).to(torch.device("cuda"))
        
        self.activation = torch.nn.Tanh()
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if bias:
            self.register_parameter("bias_value", torch.nn.Parameter(torch.zeros(out_channels)))
        
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    
    def forward(self, x, edge_index, edge_type, rel_emb, edge_weight=None):
        rel_emb = torch.cat([rel_emb, self.loop_rel], dim=0)
        
        num_ent = x.size(0)
        loop_index = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).cuda()
        loop_type   = torch.full((num_ent,), rel_emb.size(0)-1, dtype=torch.long).cuda()

        in_res = self.propagate(edge_index=edge_index, x=x, edge_type=edge_type, rel_emb=rel_emb, edge_weight=edge_weight, mode="in")
        loop_res = self.propagate(edge_index=loop_index, x=x, edge_type=loop_type, rel_emb=rel_emb, edge_weight=edge_weight, mode="loop")

        out = self.drop(in_res) + self.drop(loop_res)

        if self.bias:
            out = out + self.bias_value
        
        out = self.bn(out)
        out = self.activation(out)

        return out, self.w_rel(rel_emb)[:-1]

    def message(self,x_i, x_j, edge_type, rel_emb, ptr, index, size_i, mode, edge_weight):
        
        rel_emb = torch.index_select(rel_emb, 0, edge_type)
        xj_rel = self.rel_transform(x_j, rel_emb)
        
        if mode == "in":
            out = self.w1(xj_rel)
            b = self.leaky_relu(self.w_att(torch.cat((x_i, rel_emb, x_j), dim=1))).cuda()
            b = self.a(b)
            alpha = softmax(b, index, ptr, size_i)
            alpha = F.dropout(alpha, p=self.drop_ratio, training=self.training, inplace=False)
            
            if edge_weight!=None:
                out = out * alpha.view(-1,1) * edge_weight.view(-1,1)
            else:
                out = out * alpha.view(-1,1)

        if mode == "loop":
            out = self.w_loop(xj_rel)

        return out

    def update(self, aggr_out):
        return aggr_out

    def rel_transform(self, ent_embed, rel_emb):
        
        if self.op == 'corr':
            trans_embed  = ccorr(ent_embed, rel_emb)
        elif self.op == 'sub':
            trans_embed = ent_embed - rel_emb
        elif self.op == 'mult':
            trans_embed = ent_embed * rel_emb
        elif self.op == "corr_new":
            trans_embed = ccorr_new(ent_embed, rel_emb)
        elif self.op == "conv":
            trans_embed = cconv(ent_embed, rel_emb)
        elif self.op == "conv_new":
            trans_embed = cconv_new(ent_embed, rel_emb)
        elif self.op == 'cross':
            trans_embed = ent_embed * rel_emb + ent_embed
        elif self.op == "corr_plus":
            trans_embed = ccorr_new(ent_embed, rel_emb) + ent_embed
        else:
            raise NotImplementedError
        
        return trans_embed

class CompGATv3(MessagePassing):
    def __init__(self, in_channels, out_channels, rel_dim, drop, bias, op, beta):
        super(CompGATv3, self).__init__(aggr = "add", node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.op = op
        self.bias = bias
        self.beta = beta
        
        # self.w_loop = torch.nn.Linear(in_channels, out_channels).cuda()
        self.w_in = torch.nn.Linear(in_channels, out_channels).cuda()
        self.w_out = torch.nn.Linear(in_channels, out_channels).cuda()
        self.w_rel = torch.nn.Linear(in_channels, out_channels).cuda()
        self.w_att = torch.nn.Linear(3*in_channels, out_channels).cuda()
        self.a = torch.nn.Linear(out_channels, 1, bias=False).cuda()
        # self.loop_rel = torch.nn.Parameter(torch.Tensor(1, in_channels)).cuda()
        # torch.nn.init.xavier_uniform_(self.loop_rel)
        
        self.drop_ratio = drop
        self.drop = torch.nn.Dropout(drop, inplace=False)
        self.bn = torch.nn.BatchNorm1d(out_channels).to(torch.device("cuda"))
        
        self.res_w = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.activation = torch.nn.Tanh() #torch.nn.Tanh()
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if bias:
            self.register_parameter("bias_value", torch.nn.Parameter(torch.zeros(out_channels)))
        
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    
    def forward(self, x, edge_index, edge_type, rel_emb, pre_alpha=None):
        # rel_emb = torch.cat([rel_emb, self.loop_rel], dim=0)
        
        num_ent = x.size(0)
        # loop_index = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).cuda()
        # loop_type   = torch.full((num_ent,), rel_emb.size(0)-1, dtype=torch.long).cuda()

        in_res = self.propagate(edge_index=edge_index, x=x, edge_type=edge_type, rel_emb=rel_emb, pre_alpha=pre_alpha)
        # loop_res = self.propagate(edge_index=loop_index, x=x, edge_type=loop_type, rel_emb=rel_emb, pre_alpha=pre_alpha, mode="loop")
        loop_res = self.res_w(x)
        out = self.drop(in_res) + self.drop(loop_res)

        if self.bias:
            out = out + self.bias_value
        
        out = self.bn(out)
        out = self.activation(out)

        return out, self.w_rel(rel_emb), self.alpha.detach()

    def message(self,x_i, x_j, edge_type, rel_emb, ptr, index, size_i, pre_alpha):
        
        rel_emb = torch.index_select(rel_emb, 0, edge_type)
        xj_rel = self.rel_transform(x_j, rel_emb)
        num_edge = xj_rel.size(0)//2

        in_message = xj_rel[:num_edge]
        out_message = xj_rel[num_edge:]
                
        trans_in = self.w_in(in_message)
        trans_out = self.w_out(out_message)
        
        out = torch.cat((trans_in, trans_out), dim=0)
        
        b = self.leaky_relu(self.w_att(torch.cat((x_i, rel_emb, x_j), dim=1))).cuda()
        b = self.a(b).float()
        alpha = softmax(b, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.drop_ratio, training=self.training, inplace=False)
        if pre_alpha!=None and self.beta != 0:
            self.alpha = alpha*(1-self.beta) + pre_alpha*(self.beta)
        else:
            self.alpha = alpha
        out = out * alpha.view(-1,1)


        return out

    def update(self, aggr_out):
        return aggr_out

    def rel_transform(self, ent_embed, rel_emb):
        
        if self.op == 'corr':
            trans_embed  = ccorr(ent_embed, rel_emb)
        elif self.op == 'sub':
            trans_embed = ent_embed - rel_emb
        elif self.op == 'mult':
            trans_embed = ent_embed * rel_emb
        elif self.op == "corr_new":
            trans_embed = ccorr_new(ent_embed, rel_emb)
        elif self.op == "conv":
            trans_embed = cconv(ent_embed, rel_emb)
        elif self.op == "conv_new":
            trans_embed = cconv_new(ent_embed, rel_emb)
        elif self.op == 'cross':
            trans_embed = ent_embed * rel_emb + ent_embed
        elif self.op == "corr_plus":
            trans_embed = ccorr_new(ent_embed, rel_emb) + ent_embed
        elif self.op == "rotate":
            trans_embed = rotate(ent_embed, rel_emb)
        else:
            raise NotImplementedError
        
        return trans_embed

class Transformer(MessagePassing):
    def __init__(self, in_channels, out_channels, rel_dim, drop, bias, op, beta, num_head=1, final_layer=False):
        super(Transformer, self).__init__(aggr = "add", node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.op = op
        self.bias = bias
        self.head = num_head
        self.final_layer = final_layer
        self.beta = beta
        self.w_in = torch.nn.Linear(in_channels, out_channels)
        self.w_out = torch.nn.Linear(in_channels, out_channels)
        self.w_res = torch.nn.Linear(in_channels, out_channels)
                
        self.lin_key = torch.nn.Linear(in_channels, num_head*out_channels, bias=bias)
        self.lin_query = torch.nn.Linear(in_channels, num_head*out_channels, bias=bias)
        # self.lin_value = torch.nn.Linear(in_channels, num_head*out_channels, bias=bias)
        # self.loop_rel = torch.nn.Parameter(torch.Tensor(1, rel_dim)).cuda()
        # torch.nn.init.xavier_normal_(self.loop_rel)

        if final_layer:
            self.w_rel = torch.nn.Linear(rel_dim, out_channels).cuda()
        else:
            self.w_rel = torch.nn.Linear(rel_dim, num_head*out_channels).cuda()

        self.drop =drop
        self.dropout = torch.nn.Dropout(drop)
        
        self.bn = torch.nn.BatchNorm1d(out_channels).to(torch.device("cuda"))
        self.activation = torch.nn.Tanh()

        if bias:
            self.register_parameter("bias_value", torch.nn.Parameter(torch.zeros(out_channels)))
    
    def forward(self, x, edge_index, edge_type, rel_emb, pre_alpha=None):
        
        out = self.propagate(edge_index=edge_index, x=x, edge_type=edge_type, rel_emb=rel_emb, pre_alpha=pre_alpha)    
        
        loop_res = self.w_res(x).view(-1, self.head, self.out_channels)
        out = self.dropout(out) + self.dropout(loop_res)
        
        if self.final_layer:
            out = out.mean(dim=1)
        else:
            out = out.view(-1, self.head*self.out_channels)
        
        out = self.activation(out)    
        out = self.bn(out)
        
        return out, self.w_rel(rel_emb), self.alpha.detach()

    def message(self, x_i, x_j, edge_type, rel_emb, ptr, index, size_i, pre_alpha):
        
        rel_emb = torch.index_select(rel_emb, 0, edge_type)
        xj_rel = self.rel_transform(x_j, rel_emb)
        num_edge = xj_rel.size(0)//2

        in_message = xj_rel[:num_edge]
        out_message = xj_rel[num_edge:]
        
        trans_in = self.w_in(in_message)
        trans_out = self.w_out(out_message)
        out = torch.cat((trans_in, trans_out), dim=0).view(-1, self.head, self.out_channels)
        
        query = self.lin_query(x_i).view(-1, self.head, self.out_channels)
        key = self.lin_key(xj_rel).view(-1, self.head, self.out_channels)

        alpha = (query * key).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.drop, training=self.training)

        if pre_alpha!=None and self.beta != 0:
            self.alpha = alpha*(1-self.beta) + pre_alpha*(self.beta)
        else:
            self.alpha = alpha
        # out = self.lin_value(xj_rel).view(-1, self.head, self.out_channels)
        out *= self.alpha.view(-1, self.head, 1)

        return out

    def update(self, aggr_out):
        return aggr_out

    def rel_transform(self, ent_embed, rel_emb):
        
        if self.op == 'corr':
            trans_embed  = ccorr(ent_embed, rel_emb)
        elif self.op == 'sub':
            trans_embed = ent_embed - rel_emb
        elif self.op == 'mult':
            trans_embed = ent_embed * rel_emb
        elif self.op == "corr_new":
            trans_embed = ccorr_new(ent_embed, rel_emb)
        elif self.op == "conv":
            trans_embed = cconv(ent_embed, rel_emb)
        elif self.op == "conv_new":
            trans_embed = cconv_new(ent_embed, rel_emb)
        else:
            raise NotImplementedError      
        return trans_embed

from aggregators import AGGREGATORS
from scalers import SCALERS
from typing import Optional, List, Dict

class CompGATv4(MessagePassing):
    def __init__(self, in_channels, out_channels, drop, op, deg, aggregators, scalers, edge_dim):
        super(CompGATv4, self).__init__(node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.op = op
        self.eps = 1e-6
        
        self.edge_dim = edge_dim
        # self.w_loop = torch.nn.Linear(in_channels, out_channels).cuda()
        self.w_in = torch.nn.Linear(in_channels, out_channels).cuda()
        self.w_out = torch.nn.Linear(in_channels, out_channels).cuda()
        self.w_rel = torch.nn.Linear(in_channels, out_channels).cuda()
        self.w_att = torch.nn.Linear(3*in_channels, out_channels).cuda()
        self.a = torch.nn.Linear(out_channels, 1, bias=False).cuda()
        
        deg = deg.to(torch.float)
        total_no_vertices = deg.sum()
        bin_degrees = torch.arange(len(deg))
        self.avg_deg: Dict[str, float] = {
            'lin': ((bin_degrees * deg).sum() / total_no_vertices).item(),
            'log': (((bin_degrees + 1).log() * deg).sum() / total_no_vertices).item(),
            'exp': ((bin_degrees.exp() * deg).sum() / total_no_vertices).item(),
        }

        self.aggregators = [AGGREGATORS[aggr] for aggr in aggregators]
        self.scalers = [SCALERS[scale] for scale in scalers]
        self.F_in = in_channels
        self.post_nns = torch.nn.ModuleList()
        modules = [torch.nn.Linear((3 if edge_dim else 2) * self.F_in, self.F_in)]
        self.post_nns.append(Sequential(*modules))
        self.lin = torch.nn.Linear(out_channels, out_channels)

        self.drop_ratio = drop
        self.drop = torch.nn.Dropout(drop, inplace=False)
        self.bn = torch.nn.BatchNorm1d(out_channels).to(torch.device("cuda"))
        
        self.activation = torch.nn.Tanh() #torch.nn.Tanh()
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    
    def forward(self, x, edge_index, edge_type, rel_emb, pre_alpha=None):
        
        x = x.view(-1, 1, self.F_in)

        out = self.propagate(edge_index=edge_index, x=x, edge_type=edge_type, rel_emb=rel_emb, pre_alpha=pre_alpha)
        out = torch.cat((x, out), dim=-1)
        outs = [nn(out[:, i]) for i, nn in enumerate(self.post_nns)]
        out = torch.cat(outs, dim=1)

        return self.lin(out)

    def message(self, x_i, x_j, edge_type, rel_emb, ptr, index, size_i, pre_alpha):
        
        rel_emb = torch.index_select(rel_emb, 0, edge_type)
        xj_rel = self.rel_transform(x_j, rel_emb)
        num_edge = xj_rel.size(0)//2

        in_message = xj_rel[:num_edge]
        out_message = xj_rel[num_edge:]
                
        trans_in = self.w_in(in_message)
        trans_out = self.w_out(out_message)
        
        out = torch.cat((trans_in, trans_out), dim=0)
        
        b = self.leaky_relu(self.w_att(torch.cat((x_i, rel_emb, x_j), dim=1))).cuda()
        b = self.a(b)
        alpha = softmax(b, index, ptr, size_i)
        # self.alpha = F.dropout(alpha, p=self.drop_ratio, training=self.training, inplace=False)
    
        out = out * alpha.view(-1,1)
        return out.unsqueeze(1)

    def aggregate(self, inputs, index: Tensor, ptr, dim_size) -> Tensor:
        outs = [aggr(inputs, index, dim_size) for aggr in self.aggregators]
        out = torch.cat(outs, dim=-1)

        deg = degree(index, dim_size, dtype=inputs.dtype).view(-1, 1, 1)
        outs = [scaler(out, deg, self.avg_deg) for scaler in self.scalers]
        return torch.cat(outs, dim=-1)
    
    def update(self, aggr_out):
        return aggr_out

    def rel_transform(self, ent_embed, rel_emb):
        
        if self.op == 'corr':
            trans_embed  = ccorr(ent_embed, rel_emb)
        elif self.op == 'sub':
            trans_embed = ent_embed - rel_emb
        elif self.op == 'mult':
            trans_embed = ent_embed * rel_emb
        elif self.op == "corr_new":
            trans_embed = ccorr_new(ent_embed, rel_emb)
        elif self.op == "conv":
            trans_embed = cconv(ent_embed, rel_emb)
        elif self.op == "conv_new":
            trans_embed = cconv_new(ent_embed, rel_emb)
        elif self.op == 'cross':
            trans_embed = ent_embed * rel_emb + ent_embed
        elif self.op == "corr_plus":
            trans_embed = ccorr_new(ent_embed, rel_emb) + ent_embed
        else:
            raise NotImplementedError
        
        return trans_embed