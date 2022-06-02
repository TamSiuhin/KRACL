import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn import functional as F, Parameter
from Encoder import CompGATv4, CompGCN, CompGAT, CompGATv2, Transformer, CompGATv3
from torch_geometric.nn import Sequential
from Encoder import ARGAT
from  utils import *
from Evaluation import Evaluator

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            # SimCLR loss
            mask = torch.eye(batch_size).float().to(device)
        elif labels is not None:
            # Supconloss
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # concat all contrast features at dim 0
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob

        # negative samples
        exp_logits = torch.exp(logits) * logits_mask
    
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # avoid nan loss when there's one sample for a certain class, e.g., 0,1,...1 for bin-cls , this produce nan for 1st in Batch
        # which also results in batch total loss as nan. such row should be dropped
        pos_per_sample=mask.sum(1) #B
        pos_per_sample[pos_per_sample<1e-6]=1.0
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_per_sample #mask.sum(1)

        #mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class relation_contrast(torch.nn.Module):
    def __init__(self, temperature, num_neg):
        super(relation_contrast, self).__init__()
        self.temperature = temperature
        self.num_neg = num_neg
        # self.all_pos_triples = all_pos_triple

    def forward(self, pos_scores, neg_scores):
        neg_scores = neg_scores.view(-1, self.num_neg, 1)
        pos = torch.exp(torch.div(pos_scores, self.temperature))
        neg = torch.exp(torch.div(neg_scores, self.temperature)).sum(dim=1)
        loss = -torch.log(torch.div(pos, neg)).mean()
        return loss
    
    # def forward(self, aug_emb, ent_emb, hrt_batch):
    #     device = (torch.device('cuda')
    #               if aug_emb.is_cuda
    #               else torch.device('cpu'))
    #     num_ent = ent_emb.size(0)
    #     label = hrt_batch[:, 2].contiguous().view(-1, 1).to(device)

    #     mask = torch.eq(label, torch.arange(0, num_ent).view(-1, 1).T.to(device)).float().to(device)
        
    #     filter_batch = self.create_sparse_positive_filter_(hrt_batch, self.all_pos_triples)
    #     mask[filter_batch[:, 0], filter_batch[:, 1]] = 1.0
        
    #     aug_abs = aug_emb.norm(dim=1)
    #     emb_abs = ent_emb.norm(dim=1)
    #     sim_matrix = torch.einsum('ik,jk->ij', aug_emb, ent_emb) / torch.einsum('i,j->ij', aug_abs, emb_abs)
    #     sim_matrix = torch.exp(sim_matrix / self.temperature)

    #     # sim_matrix = torch.div(torch.matmul(aug_emb, ent_emb.T), self.temperature)
    #     pos_sim = (mask * sim_matrix).sum(dim=1)
    #     loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    #     loss = -(torch.log(loss) / mask.sum(dim=1)).mean()

    #     return loss
    
    # def create_sparse_positive_filter_(self, hrt_batch, all_pos_triples, filter_col=2):
        
    #     relations = hrt_batch[:, 1:2]
    #     relation_filter = (all_pos_triples[:, 1:2]).view(1, -1) == relations

    #     # Split batch
    #     other_col = 2 - filter_col
    #     entities = hrt_batch[:, other_col : other_col + 1]

    #     entity_filter_test = (all_pos_triples[:, other_col : other_col + 1]).view(1, -1) == entities
    #     filter_batch = (entity_filter_test & relation_filter).nonzero(as_tuple=False)
    #     filter_batch[:, 1] = all_pos_triples[:, filter_col : filter_col + 1].view(1, -1)[:, filter_batch[:, 1]]

    #     return filter_batch

class CLKG_compgat_convE_newloss(nn.Module):
    def __init__(self, args):
        super(CLKG_compgat_convE_newloss, self).__init__()
        self.ent_dim = args.ent_dim
        self.learning_rate = args.cl_lr
        self.batch_size = args.cl_batch_size
        self.op = args.op

        self.ent_emb = torch.nn.Embedding(args.ent_num, args.ent_dim).to(torch.device("cuda"))
        self.rel_emb = torch.nn.Embedding(args.rel_num, args.ent_dim).to(torch.device("cuda"))
        torch.nn.init.xavier_uniform_(self.ent_emb.weight)
        torch.nn.init.xavier_uniform_(self.rel_emb.weight)

        self.ent_dim = args.ent_dim
        self.ent_h = args.ent_height
        self.ent_w = self.ent_dim // self.ent_h

        if args.gcn_layer == 2:
            self.encoder = Sequential("x, edge_index, edge_type, rel_emb",[
                (CompGCN(in_channels=args.ent_dim, out_channels=args.ent_dim, drop=args.encoder_drop, bias=True, op=args.op), "x, edge_index, edge_type, rel_emb -> x, rel_emb"),
                (torch.nn.Dropout(p=args.encoder_hid_drop), "x -> x"),
                (CompGCN(in_channels=args.ent_dim, out_channels=args.ent_dim, drop=args.encoder_drop, bias=True, op=args.op), "x, edge_index, edge_type, rel_emb -> x, rel_emb"),
            ]).to(torch.device("cuda"))
        
        elif args.gcn_layer == 1:
            self.encoder = Sequential("x, edge_index, edge_type, rel_emb",[
                (CompGCN(in_channels=args.ent_dim, out_channels=args.ent_dim, drop=args.encoder_drop, bias=True, op=self.op), "x, edge_index, edge_type, rel_emb -> x, rel_emb"),
            ]).to(torch.device("cuda"))
        
        self.input_drop = torch.nn.Dropout(args.input_drop)
        self.fea_drop = torch.nn.Dropout(args.fea_drop)
        self.hid_drop = torch.nn.Dropout(args.hid_drop)

        self.conv1 = torch.nn.Conv2d(1, args.filter_channel, (args.filter_size, args.filter_size), 1, 0, bias=False)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(args.filter_channel)
        self.bn2 = torch.nn.BatchNorm1d(args.ent_dim)
        self.register_parameter('b', torch.nn.Parameter(torch.zeros(args.ent_num)))
       
        num_in_features = (args.filter_channel*(2*args.ent_height-args.filter_size+1)*(self.ent_w-args.filter_size+1))

        if args.proj == "mlp":
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(num_in_features, args.ent_dim),
                torch.nn.BatchNorm1d(args.ent_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(args.ent_dim, args.ent_dim),
                torch.nn.BatchNorm1d(args.ent_dim))
        elif args.proj == "linear":
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(num_in_features, args.ent_dim),
                torch.nn.BatchNorm1d(args.ent_dim),
                torch.nn.ReLU(inplace=True))
        else:
            raise ValueError("Type of projection head should be mlp or linear!")

    def forward(self, edge_index, edge_type, h, r, t, ent_emb, rel_emb):

        ent_emb, rel_emb = self.encoder(ent_emb, edge_index, edge_type, rel_emb)
    
        head_emb = torch.index_select(ent_emb, 0, h)
        rel_emb = torch.index_select(rel_emb, 0, r).view(-1, 1, self.ent_dim)
        # tail_emb = torch.index_select(ent_emb, 0, t)
            
        e1_emb = head_emb.view(-1, 1, self.ent_dim)
        stack_input = torch.cat([e1_emb, rel_emb], 1)
        stack_input	= torch.transpose(stack_input, 2, 1).reshape((-1, 1, 2*self.ent_h, self.ent_w))

        x = self.bn0(stack_input)
        x = self.input_drop(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fea_drop(x)
        x = x.view(x.shape[0], -1)
        cl_x = self.proj(x)

        # cl_x = F.normalize(x, dim=1)
        # cl_x = x
        
        # x = self.hid_drop(x)
        # x = self.bn2(x)
        # x = F.relu(x, inplace=True)
        x = torch.mm(cl_x, ent_emb.transpose(1,0))
        score = x + self.b.expand_as(x)

        return cl_x, score, ent_emb
    
    def predict_t(self, hr_batch, ent_emb, rel_emb, edge_index, edge_type):
        
        with torch.no_grad():
            ent_emb, rel_emb = self.encoder(self.ent_emb.weight, edge_index, edge_type, self.rel_emb.weight)
            
            e1 = hr_batch[:, 0]
            rel = hr_batch[:, 1]
            e1_embedded= torch.index_select(ent_emb, 0, e1).view(-1, 1, self.ent_dim)
            rel_embedded = torch.index_select(rel_emb, 0, rel).view(-1, 1, self.ent_dim)

            stack_input = torch.cat([e1_embedded, rel_embedded], 1)
            stack_input	= torch.transpose(stack_input, 2, 1).reshape((-1, 1, 2*self.ent_h, self.ent_w))
            
            x = self.bn0(stack_input)
            x = self.input_drop(x)
            x= self.conv1(x)
            x= self.bn1(x)
            x= F.relu(x)
            x = self.fea_drop(x)
            x = x.view(x.shape[0], -1)
            x = self.proj(x)
            # x = self.hid_drop(x)
            # x = self.bn2(x)
            # x = F.relu(x, inplace=True)
            x = torch.mm(x, ent_emb.transpose(1,0))
            x += self.b.expand_as(x)

            return x

class CLKG_compgat_convr(nn.Module):
    def __init__(self, args):
        super(CLKG_compgat_convr, self).__init__()
        self.ent_dim = args.ent_dim
        self.learning_rate = args.cl_lr
        self.batch_size = args.cl_batch_size
        self.op = args.op

        self.ent_emb = torch.nn.Embedding(args.ent_num, args.ent_dim).to(torch.device("cuda"))
        self.rel_emb = torch.nn.Embedding(args.rel_num, args.ent_dim).to(torch.device("cuda"))

        if args.gcn_layer == 2:
            self.encoder = Sequential("x, edge_index, edge_type, rel_emb",[
                (CompGCN(in_channels=args.ent_dim, out_channels=args.ent_dim, drop=args.encoder_drop, bias=True, op=self.op), "x, edge_index, edge_type, rel_emb -> x, rel_emb"),
                (torch.nn.Dropout(p=args.encoder_drop), "x -> x"),
                (CompGCN(in_channels=args.ent_dim, out_channels=args.ent_dim, drop=args.encoder_drop, bias=True, op=self.op), "x, edge_index, edge_type, rel_emb -> x, rel_emb"),
            ]).to(torch.device("cuda"))
        
        elif args.gcn_layer == 1:
            self.encoder = Sequential("x, edge_index, edge_type, rel_emb",[
                (CompGCN(in_channels=args.ent_dim, out_channels=args.ent_dim, drop=args.encoder_drop, bias=True, op=self.op), "x, edge_index, edge_type, rel_emb -> x, rel_emb"),
            ]).to(torch.device("cuda"))
        
        self.input_drop = torch.nn.Dropout(args.input_drop)
        self.fea_drop = torch.nn.Dropout2d(args.fea_drop)
        self.hid_drop = torch.nn.Dropout(args.hid_drop)

        self.filter_size  = args.filter_size
        self.c = args.rel_dim // (args.filter_size**2)
        self.ent_dim = args.ent_dim
        self.ent_h = args.ent_height
        self.ent_w = self.ent_dim // self.ent_h

        self.bn_rel = torch.nn.BatchNorm1d(args.rel_dim)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.c)
        self.bn2 = torch.nn.BatchNorm1d(args.ent_dim)

        self.register_parameter('b', torch.nn.Parameter(torch.zeros(args.ent_num)))  # num_entities

        if args.proj == "mlp":
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(self.c*(self.ent_h-self.filter_size+1)*(self.ent_w-self.filter_size+1), args.ent_dim),
                torch.nn.BatchNorm1d(args.ent_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(args.ent_dim, args.proj_dim),
                torch.nn.BatchNorm1d(args.proj_dim))
        elif args.proj == "linear":
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(self.c*(self.ent_h-self.filter_size+1)*(self.ent_w-self.filter_size+1), args.ent_dim),
                torch.nn.BatchNorm1d(args.ent_dim),
                torch.nn.ReLU())
        else:
            raise ValueError("Type of projection head should be mlp or linear!")
        
        if args.init_emb.lower() == "random":
            torch.nn.init.xavier_uniform_(self.ent_emb.weight)
            torch.nn.init.xavier_uniform_(self.rel_emb.weight)
        else:
            raise ValueError("Please choose the initial embedding!")


    def forward(self, edge_index, edge_type, h, r, t, ent_emb, rel_emb):

        batch_size = h.size(0)

        ent_emb, rel_emb = self.encoder(self.ent_emb.weight, edge_index, edge_type, self.rel_emb.weight)

        head_emb = torch.index_select(ent_emb, 0, h).view(-1, 1, self.ent_h, self.ent_w)
        rel_emb = torch.index_select(rel_emb, 0, r)
        tail_emb = torch.index_select(ent_emb, 0, t)
        
        rel_emb = self.bn_rel(rel_emb)

        filters = rel_emb.view(-1, self.c, 1, self.filter_size, self.filter_size).view(-1, 1, self.filter_size, self.filter_size)
        x = self.bn0(head_emb).view(1, -1, self.ent_h, self.ent_w)
        x = F.conv2d(x, filters, stride=1, padding=0, groups=batch_size)
        x = x.view(batch_size, self.c, (self.ent_h-self.filter_size+1), (self.ent_w-self.filter_size+1))

        x = self.bn1(x)
        x = F.relu(x)
        x = self.fea_drop(x)
        x = x.view(x.shape[0], -1)  # batch_size,100*6*6
        x = self.proj(x)
        cl_x = F.normalize(x, dim=1)
        x = self.hid_drop(x)
        x = self.bn2(x)
        x = F.relu(x)  # batch_size,100
        x = torch.mm(x, ent_emb.transpose(1, 0))  # batch_size,num_entities
        x += self.b.expand_as(x)  # batch_size,num_entities
        
        return cl_x, x, tail_emb

    def predict_t(self, hr_batch, ent_emb, rel_emb, edge_index, edge_type):
        
        h = hr_batch[:, 0]
        r = hr_batch[:, 1]

        batch_size = h.size(0)

        ent_emb, rel_emb = self.encoder(self.ent_emb.weight, edge_index, edge_type, self.rel_emb.weight)

        head_emb = torch.index_select(ent_emb, 0, h).view(-1, 1, self.ent_h, self.ent_w)
        rel_emb = torch.index_select(rel_emb, 0, r)
        rel_emb = self.bn_rel(rel_emb)

        rel_emb = self.input_drop(rel_emb)
        head_emb = self.input_drop(head_emb)

        filters = rel_emb.view(-1, self.c, 1, self.filter_size, self.filter_size).view(-1, 1, self.filter_size, self.filter_size)
        
        x = self.bn0(head_emb).view(1, -1, self.ent_h, self.ent_w)
        x = F.conv2d(x, filters, stride=1, padding=0, groups=batch_size)
        x = x.view(batch_size, self.c, (self.ent_h-self.filter_size+1), (self.ent_w-self.filter_size+1))

        x = self.bn1(x)
        x = F.relu(x)
        x = self.fea_drop(x)
        x = x.view(x.shape[0], -1)  # batch_size,100*6*6
        x = self.proj(x)  # batch_size,100
        x = self.hid_drop(x)
        x = self.bn2(x)
        x = F.relu(x)  # batch_size,100
        x = torch.mm(x, ent_emb.transpose(1, 0))  # batch_size,num_entities
        x += self.b.expand_as(x)  # batch_size,num_entities

        return x

class CLKG_compgat_convE(nn.Module):
    def __init__(self, args):
        super(CLKG_compgat_convE, self).__init__()
        self.ent_dim = args.ent_dim
        self.learning_rate = args.cl_lr
        self.batch_size = args.cl_batch_size
        self.op = args.op

        self.ent_emb = torch.nn.Embedding(args.ent_num, args.init_dim).to(torch.device("cuda"))
        self.rel_emb = torch.nn.Embedding(args.rel_num, args.init_dim).to(torch.device("cuda"))
        torch.nn.init.xavier_uniform_(self.ent_emb.weight)
        torch.nn.init.xavier_uniform_(self.rel_emb.weight)

        self.ent_dim = args.ent_dim
        self.ent_h = args.ent_height
        self.ent_w = self.ent_dim // self.ent_h

        if args.gcn_layer == 2:
            self.encoder = Sequential("x, edge_index, edge_type, rel_emb",[
                (CompGAT(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op), "x, edge_index, edge_type, rel_emb -> x, rel_emb"),
                (torch.nn.Dropout(p=args.encoder_hid_drop), "x -> x"),
                (CompGAT(in_channels=args.ent_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op), "x, edge_index, edge_type, rel_emb -> x, rel_emb"),
            ]).to(torch.device("cuda"))
        
        elif args.gcn_layer == 1:
            self.encoder = Sequential("x, edge_index, edge_type, rel_emb",[
                (CompGAT(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=self.op), "x, edge_index, edge_type, rel_emb -> x, rel_emb"),
            ]).to(torch.device("cuda"))
        
        self.encoder_drop = torch.nn.Dropout(p=args.encoder_hid_drop, inplace=False)
        self.input_drop = torch.nn.Dropout(args.input_drop, inplace=False)
        self.fea_drop = torch.nn.Dropout(args.fea_drop, inplace=False)
        # self.hid_drop = torch.nn.Dropout(args.hid_drop)

        self.conv1 = torch.nn.Conv2d(1, args.filter_channel, (args.filter_size, args.filter_size), 1, 0, bias=False)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(args.filter_channel)
        self.bn2 = torch.nn.BatchNorm1d(args.ent_dim)
        self.register_parameter('b', torch.nn.Parameter(torch.zeros(args.ent_num)))
       
        num_in_features = (args.filter_channel*(2*args.ent_height-args.filter_size+1)*(self.ent_w-args.filter_size+1))

        if args.proj == "mlp":
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(num_in_features, args.ent_dim),
                torch.nn.BatchNorm1d(args.ent_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(args.ent_dim, args.ent_dim),
                torch.nn.BatchNorm1d(args.ent_dim))
        
        elif args.proj == "linear":
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(num_in_features, args.ent_dim),
                torch.nn.Dropout(args.hid_drop),
                torch.nn.BatchNorm1d(args.ent_dim),
                torch.nn.PReLU())
        else:
            raise ValueError("Type of projection head should be mlp or linear!")

        self.relu = torch.nn.PReLU()
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, edge_index, edge_type, h, r, t, ent_emb, rel_emb):

        ent_emb, rel_emb = self.encoder(ent_emb, edge_index, edge_type, rel_emb)
        ent_emb = self.encoder_drop(ent_emb)

        head_emb = torch.index_select(ent_emb, 0, h)
        rel_emb = torch.index_select(rel_emb, 0, r).view(-1, 1, self.ent_dim)
        tail_emb = torch.index_select(ent_emb, 0, t)
        
        x = head_emb.view(-1, 1, self.ent_dim)
        x = torch.cat([x, rel_emb], 1)
        x	= torch.transpose(x, 2, 1).reshape((-1, 1, 2*self.ent_h, self.ent_w))

        x = self.bn0(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fea_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.proj(x)
        
        # cl_x = F.normalize(x, dim=1)
        cl_x = x
        # x = self.hid_drop(x)
        # x = self.bn2(x)
        # x = F.relu(x, inplace=True)
        x = torch.mm(x, ent_emb.transpose(1,0))
        x += self.b.expand_as(x)

        return cl_x, x, tail_emb, head_emb
    
    def predict_t(self, hr_batch, ent_emb, rel_emb, edge_index, edge_type):
        
        with torch.no_grad():
            ent_emb, rel_emb = self.encoder(self.ent_emb.weight, edge_index, edge_type, self.rel_emb.weight)
            
            e1 = hr_batch[:, 0]
            rel = hr_batch[:, 1]
            e1_embedded= torch.index_select(ent_emb, 0, e1).view(-1, 1, self.ent_dim)
            rel_embedded = torch.index_select(rel_emb, 0, rel).view(-1, 1, self.ent_dim)

            x = torch.cat([e1_embedded, rel_embedded], 1)
            x	= torch.transpose(x, 2, 1).reshape((-1, 1, 2*self.ent_h, self.ent_w))
            
            x = self.bn0(x)
            x = self.input_drop(x)
            x= self.conv1(x)
            x= self.bn1(x)
            x= self.relu(x)
            x = self.fea_drop(x)
            x = x.view(x.shape[0], -1)
            x = self.proj(x)
            # x = self.hid_drop(x)
            # x = self.bn2(x)
            # x = F.relu(x, inplace=True)
            x = torch.mm(x, ent_emb.transpose(1,0))
            x += self.b.expand_as(x)

            return x

class CLKG_compgatv2_convE(nn.Module):
    def __init__(self, args):
        super(CLKG_compgatv2_convE, self).__init__()
        self.ent_dim = args.ent_dim
        self.learning_rate = args.cl_lr
        self.batch_size = args.cl_batch_size
        self.op = args.op

        self.ent_emb = torch.nn.Embedding(args.ent_num, args.init_dim).to(torch.device("cuda"))
        self.rel_emb = torch.nn.Embedding(args.rel_num, args.init_dim).to(torch.device("cuda"))
        torch.nn.init.xavier_uniform_(self.ent_emb.weight)
        torch.nn.init.xavier_uniform_(self.rel_emb.weight)

        self.ent_dim = args.ent_dim
        self.ent_h = args.ent_height
        self.ent_w = self.ent_dim // self.ent_h

        if args.gcn_layer == 2:
            self.encoder = Sequential("x, edge_index, edge_type, rel_emb",[
                (CompGATv2(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op), "x, edge_index, edge_type, rel_emb -> x, rel_emb"),
                (torch.nn.Dropout(p=args.encoder_hid_drop), "x -> x"),
                (CompGATv2(in_channels=args.ent_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op), "x, edge_index, edge_type, rel_emb -> x, rel_emb"),
            ]).to(torch.device("cuda"))
        
        elif args.gcn_layer == 1:
            self.encoder = Sequential("x, edge_index, edge_type, rel_emb",[
                (CompGATv2(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=self.op), "x, edge_index, edge_type, rel_emb -> x, rel_emb"),
            ]).to(torch.device("cuda"))
        
        self.encoder_drop = torch.nn.Dropout(p=args.encoder_hid_drop, inplace=False)
        self.input_drop = torch.nn.Dropout(args.input_drop, inplace=False)
        self.fea_drop = torch.nn.Dropout(args.fea_drop, inplace=False)
        # self.hid_drop = torch.nn.Dropout(args.hid_drop)

        self.conv1 = torch.nn.Conv2d(1, args.filter_channel, (args.filter_size, args.filter_size), 1, 0, bias=False)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(args.filter_channel)
        self.bn2 = torch.nn.BatchNorm1d(args.ent_dim)
        self.register_parameter('b', torch.nn.Parameter(torch.zeros(args.ent_num)))
       
        num_in_features = (args.filter_channel*(2*args.ent_height-args.filter_size+1)*(self.ent_w-args.filter_size+1))

        if args.proj == "mlp":
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(num_in_features, args.ent_dim),
                torch.nn.BatchNorm1d(args.ent_dim),
                torch.nn.PReLU(),
                torch.nn.Linear(args.ent_dim, args.ent_dim),
                torch.nn.BatchNorm1d(args.ent_dim))
        
        elif args.proj == "linear":
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(num_in_features, args.ent_dim),
                torch.nn.Dropout(args.hid_drop),
                torch.nn.BatchNorm1d(args.ent_dim),
                torch.nn.PReLU())
        else:
            raise ValueError("Type of projection head should be mlp or linear!")

        self.relu = torch.nn.PReLU()
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, edge_index, edge_type, h, r, t, ent_emb, rel_emb):

        ent_emb, rel_emb = self.encoder(ent_emb, edge_index, edge_type, rel_emb)
        ent_emb = self.encoder_drop(ent_emb)

        head_emb = torch.index_select(ent_emb, 0, h)
        rel_emb = torch.index_select(rel_emb, 0, r).view(-1, 1, self.ent_dim)
        tail_emb = torch.index_select(ent_emb, 0, t)
        
        x = head_emb.view(-1, 1, self.ent_dim)
        x = torch.cat([x, rel_emb], 1)
        x	= torch.transpose(x, 2, 1).reshape((-1, 1, 2*self.ent_h, self.ent_w))

        x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fea_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.proj(x)
        x = self.bn2(x)
        cl_x = x
        # x = self.hid_drop(x)
        
        # x = F.relu(x, inplace=True)
        # ent_emb = self.bn2(ent_emb)
        x = torch.mm(x, ent_emb.transpose(1,0))
        x += self.b.expand_as(x)

        return cl_x, x, tail_emb, head_emb
    
    def predict_t(self, hr_batch, ent_emb, rel_emb, edge_index, edge_type):
        
        with torch.no_grad():

            ent_emb, rel_emb = self.encoder(self.ent_emb.weight, edge_index, edge_type, self.rel_emb.weight)
            ent_emb = self.encoder_drop(ent_emb)

            e1 = hr_batch[:, 0]
            rel = hr_batch[:, 1]
            e1_embedded= torch.index_select(ent_emb, 0, e1).view(-1, 1, self.ent_dim)
            rel_embedded = torch.index_select(rel_emb, 0, rel).view(-1, 1, self.ent_dim)

            x = torch.cat([e1_embedded, rel_embedded], 1)
            x	= torch.transpose(x, 2, 1).reshape((-1, 1, 2*self.ent_h, self.ent_w))
            
            x = self.bn0(x)
            # x = self.input_drop(x)
            x= self.conv1(x)
            x= self.bn1(x)
            x= self.relu(x)
            x = self.fea_drop(x)
            x = x.view(x.shape[0], -1)
            x = self.proj(x)
            x = self.bn2(x)
            # x = F.relu(x, inplace=True)
            # ent_emb = self.bn2(ent_emb)
            x = torch.mm(x, ent_emb.transpose(1,0))
            x += self.b.expand_as(x)

            return x

class CLKG_compgatv3_convE(nn.Module):
    def __init__(self, args):
        super(CLKG_compgatv3_convE, self).__init__()
        self.ent_dim = args.ent_dim
        self.learning_rate = args.cl_lr
        self.batch_size = args.cl_batch_size
        self.op = args.op

        self.ent_emb = torch.nn.Embedding(args.ent_num, args.init_dim).to(torch.device("cuda"))
        self.rel_emb = torch.nn.Embedding(args.rel_num, args.init_dim).to(torch.device("cuda"))
        torch.nn.init.xavier_uniform_(self.ent_emb.weight)
        torch.nn.init.xavier_uniform_(self.rel_emb.weight)

        self.ent_dim = args.ent_dim
        self.ent_h = args.ent_height
        self.ent_w = self.ent_dim // self.ent_h
        if args.gcn_layer == 2:
            # self.encoder = Sequential("x, edge_index, edge_type, rel_emb",[
            #     (CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta), "x, edge_index, edge_type, rel_emb -> x, rel_emb, alpha"),
            #     (, "x -> x"),
            #     (CompGATv3(in_channels=args.ent_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta), "x, edge_index, edge_type, rel_emb, alpha -> x, rel_emb, alpha"),
            # ]).to(torch.device("cuda"))
            self.layer = 2
            self.gnn1 = CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            self.hid_drop = torch.nn.Dropout(args.hid_drop)
            self.gnn2 = CompGATv3(in_channels=args.ent_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            
        elif args.gcn_layer == 1:
            # self.encoder = Sequential("x, edge_index, edge_type, rel_emb",[
            #     (CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=self.op, beta=args.beta), "x, edge_index, edge_type, rel_emb -> x, rel_emb, alpha"),
            # ]).to(torch.device("cuda"))
            self.layer = 1
            self.gnn1 = CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            self.hid_drop = torch.nn.Dropout(args.hid_drop)
        
        elif args.gcn_layer == 4:
            self.layer = 4
            self.gnn1 = CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            self.gnn2 = CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            self.gnn3 = CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            self.gnn4 = CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            self.hid_drop = torch.nn.Dropout(args.hid_drop)

        self.encoder_drop = torch.nn.Dropout(p=args.encoder_hid_drop, inplace=False)
        # self.input_drop = torch.nn.Dropout(args.input_drop, inplace=False)
        self.fea_drop = torch.nn.Dropout(args.fea_drop, inplace=False)
        # self.hid_drop = torch.nn.Dropout(args.hid_drop)

        self.conv1 = torch.nn.Conv2d(1, args.filter_channel, (args.filter_size, args.filter_size), 1, 0, bias=False)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(args.filter_channel)
        self.bn2 = torch.nn.BatchNorm1d(args.ent_dim)
        self.register_parameter('b', torch.nn.Parameter(torch.zeros(args.ent_num)))
       
        num_in_features = (args.filter_channel*(2*args.ent_height-args.filter_size+1)*(self.ent_w-args.filter_size+1))

        if args.proj == "mlp":
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(num_in_features, args.ent_dim),
                torch.nn.BatchNorm1d(args.ent_dim),
                torch.nn.PReLU(),
                torch.nn.Linear(args.ent_dim, args.ent_dim),
                torch.nn.BatchNorm1d(args.ent_dim))
        
        elif args.proj == "linear":
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(num_in_features, args.ent_dim),
                torch.nn.Dropout(args.hid_drop),
                torch.nn.BatchNorm1d(args.ent_dim),
                torch.nn.PReLU())
        else:
            raise ValueError("Type of projection head should be mlp or linear!")

        self.relu = torch.nn.PReLU()
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, edge_index, edge_type, h, r, t, ent_emb, rel_emb, save=False):
        if self.layer==2:
            ent_emb1, rel_emb1, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
            ent_emb1 = self.hid_drop(ent_emb1)
            ent_emb2, rel_emb, alpha2 = self.gnn2(ent_emb1, edge_index, edge_type, rel_emb1, alpha1)
            ent_emb = self.hid_drop(ent_emb2)
        
        elif self.layer==1:
            ent_emb, rel_emb, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
            ent_emb = self.hid_drop(ent_emb)
        
        elif self.layer==4:
            ent_emb, rel_emb, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
            ent_emb = self.hid_drop(ent_emb)
            ent_emb, rel_emb, alpha2 = self.gnn2(ent_emb, edge_index, edge_type, rel_emb, alpha1)
            ent_emb = self.hid_drop(ent_emb)
            ent_emb, rel_emb, alpha3 = self.gnn3(ent_emb, edge_index, edge_type, rel_emb, alpha2)
            ent_emb = self.hid_drop(ent_emb)
            ent_emb, rel_emb, alpha4 = self.gnn4(ent_emb, edge_index, edge_type, rel_emb, alpha3)
            ent_emb = self.hid_drop(ent_emb)

        head_emb = torch.index_select(ent_emb, 0, h)
        r_emb = torch.index_select(rel_emb, 0, r).view(-1, 1, self.ent_dim)
        tail_emb = torch.index_select(ent_emb, 0, t)
        
        x = head_emb.view(-1, 1, self.ent_dim)
        x = torch.cat([x, r_emb], 1)
        x	= torch.transpose(x, 2, 1).reshape((-1, 1, 2*self.ent_h, self.ent_w))

        x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fea_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.proj(x)
        if save:
            return x, ent_emb
        x = self.bn2(x)
        cl_x = x
        # x = self.hid_drop(x)
        
        # x = F.relu(x, inplace=True)
        # ent_emb = self.bn2(ent_emb)
        x = torch.mm(x, ent_emb.transpose(1,0))
        x += self.b.expand_as(x)
        # x = torch.sigmoid(x)
        return cl_x, x, tail_emb, head_emb, ent_emb, rel_emb
    
    def score_hrt(self, hrt_batch, ent_emb, rel_emb):
        h,r,t = hrt_batch.t()
        head_emb = torch.index_select(ent_emb, 0, h)
        rel_emb = torch.index_select(rel_emb, 0, r).view(-1, 1, self.ent_dim)
        tail_emb = torch.index_select(ent_emb, 0, t)
        
        x = head_emb.view(-1, 1, self.ent_dim)
        x = torch.cat([x, rel_emb], 1)
        x	= torch.transpose(x, 2, 1).reshape((-1, 1, 2*self.ent_h, self.ent_w))

        x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fea_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.proj(x)
        # x = self.bn2(x)
        
        x = (F.normalize(x, dim=1)*F.normalize(tail_emb, dim=1)).sum(dim=1, keepdim=True)
        # x += self.b[t].unsqueeze(1)
        # x = torch.sigmoid(x)
        return x
    
    def predict_t(self, hr_batch, ent_emb, rel_emb, edge_index, edge_type, save_emb=False):
        
        with torch.no_grad():

            if self.layer==2:
                ent_emb1, rel_emb1, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
                ent_emb1 = self.hid_drop(ent_emb1)
                ent_emb2, rel_emb, alpha2 = self.gnn2(ent_emb1, edge_index, edge_type, rel_emb1, alpha1)
                ent_emb = self.hid_drop(ent_emb2)
            
            elif self.layer==1:
                ent_emb, rel_emb, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
                ent_emb = self.hid_drop(ent_emb)
            
            elif self.layer==4:
                ent_emb, rel_emb, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
                ent_emb = self.hid_drop(ent_emb)
                ent_emb, rel_emb, alpha2 = self.gnn2(ent_emb, edge_index, edge_type, rel_emb, alpha1)
                ent_emb = self.hid_drop(ent_emb)
                ent_emb, rel_emb, alpha3 = self.gnn3(ent_emb, edge_index, edge_type, rel_emb, alpha2)
                ent_emb = self.hid_drop(ent_emb)
                ent_emb, rel_emb, alpha4 = self.gnn4(ent_emb, edge_index, edge_type, rel_emb, alpha3)
                ent_emb = self.hid_drop(ent_emb)

            if save_emb:
                save = {"ent_emb": ent_emb,
                        "rel_emb": rel_emb}
                torch.save(save, "./emb/saved_emb.pt")
                
            e1 = hr_batch[:, 0]
            rel = hr_batch[:, 1]
            e1_embedded= torch.index_select(ent_emb, 0, e1).view(-1, 1, self.ent_dim)
            rel_embedded = torch.index_select(rel_emb, 0, rel).view(-1, 1, self.ent_dim)

            x = torch.cat([e1_embedded, rel_embedded], 1)
            x	= torch.transpose(x, 2, 1).reshape((-1, 1, 2*self.ent_h, self.ent_w))
            
            x = self.bn0(x)
            # x = self.input_drop(x)
            x= self.conv1(x)
            x= self.bn1(x)
            x= self.relu(x)
            x = self.fea_drop(x)
            x = x.view(x.shape[0], -1)
            x = self.proj(x)
            x = self.bn2(x)
            # x = F.relu(x, inplace=True)
            # ent_emb = self.bn2(ent_emb)
            x = torch.mm(x, ent_emb.transpose(1,0))
            x += self.b.expand_as(x)
            # x = torch.sigmoid(x)
            return x

class CLKG_compgatv3_rotatE(nn.Module):
    def __init__(self, args):
        super(CLKG_compgatv3_rotatE, self).__init__()
        self.ent_dim = args.ent_dim
        self.learning_rate = args.cl_lr
        self.batch_size = args.cl_batch_size
        self.op = args.op

        self.ent_emb = torch.nn.Embedding(args.ent_num, args.init_dim).to(torch.device("cuda"))
        self.rel_emb = torch.nn.Embedding(args.rel_num, args.init_dim).to(torch.device("cuda"))
        torch.nn.init.xavier_uniform_(self.ent_emb.weight)
        torch.nn.init.xavier_uniform_(self.rel_emb.weight)

        self.ent_dim = args.ent_dim
        self.ent_h = args.ent_height
        self.ent_w = self.ent_dim // self.ent_h
        if args.gcn_layer == 2:
            # self.encoder = Sequential("x, edge_index, edge_type, rel_emb",[
            #     (CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta), "x, edge_index, edge_type, rel_emb -> x, rel_emb, alpha"),
            #     (, "x -> x"),
            #     (CompGATv3(in_channels=args.ent_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta), "x, edge_index, edge_type, rel_emb, alpha -> x, rel_emb, alpha"),
            # ]).to(torch.device("cuda"))
            self.layer = 2
            self.gnn1 = CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            self.hid_drop = torch.nn.Dropout(args.hid_drop)
            self.gnn2 = CompGATv3(in_channels=args.ent_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            
        elif args.gcn_layer == 1:
            # self.encoder = Sequential("x, edge_index, edge_type, rel_emb",[
            #     (CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=self.op, beta=args.beta), "x, edge_index, edge_type, rel_emb -> x, rel_emb, alpha"),
            # ]).to(torch.device("cuda"))
            self.layer = 1
            self.gnn1 = CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            self.hid_drop = torch.nn.Dropout(args.hid_drop)
        
        elif args.gcn_layer == 4:
            self.layer = 4
            self.gnn1 = CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            self.gnn2 = CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            self.gnn3 = CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            self.gnn4 = CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            self.hid_drop = torch.nn.Dropout(args.hid_drop)

        self.encoder_drop = torch.nn.Dropout(p=args.encoder_hid_drop, inplace=False)
        # self.input_drop = torch.nn.Dropout(args.input_drop, inplace=False)
        self.fea_drop = torch.nn.Dropout(args.fea_drop, inplace=False)
        # self.hid_drop = torch.nn.Dropout(args.hid_drop)

        self.conv1 = torch.nn.Conv2d(1, args.filter_channel, (args.filter_size, args.filter_size), 1, 0, bias=False)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(args.filter_channel)
        self.bn2 = torch.nn.BatchNorm1d(args.ent_dim)
        self.register_parameter('b', torch.nn.Parameter(torch.zeros(args.ent_num)))
       
        num_in_features = (args.filter_channel*(2*args.ent_height-args.filter_size+1)*(self.ent_w-args.filter_size+1))

        if args.proj == "mlp":
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(num_in_features, args.ent_dim),
                torch.nn.BatchNorm1d(args.ent_dim),
                torch.nn.PReLU(),
                torch.nn.Linear(args.ent_dim, args.ent_dim),
                torch.nn.BatchNorm1d(args.ent_dim))
        
        elif args.proj == "linear":
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(num_in_features, args.ent_dim),
                torch.nn.Dropout(args.hid_drop),
                torch.nn.BatchNorm1d(args.ent_dim),
                torch.nn.PReLU())
        else:
            raise ValueError("Type of projection head should be mlp or linear!")

        self.relu = torch.nn.PReLU()
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, edge_index, edge_type, h, r, t, ent_emb, rel_emb):
        if self.layer==2:
            ent_emb1, rel_emb1, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
            ent_emb1 = self.hid_drop(ent_emb1)
            ent_emb2, rel_emb, alpha2 = self.gnn2(ent_emb1, edge_index, edge_type, rel_emb1, alpha1)
            ent_emb = self.hid_drop(ent_emb2)
        
        elif self.layer==1:
            ent_emb, rel_emb, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
            ent_emb = self.hid_drop(ent_emb)
        
        elif self.layer==4:
            ent_emb, rel_emb, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
            ent_emb = self.hid_drop(ent_emb)
            ent_emb, rel_emb, alpha2 = self.gnn2(ent_emb, edge_index, edge_type, rel_emb, alpha1)
            ent_emb = self.hid_drop(ent_emb)
            ent_emb, rel_emb, alpha3 = self.gnn3(ent_emb, edge_index, edge_type, rel_emb, alpha2)
            ent_emb = self.hid_drop(ent_emb)
            ent_emb, rel_emb, alpha4 = self.gnn4(ent_emb, edge_index, edge_type, rel_emb, alpha3)
            ent_emb = self.hid_drop(ent_emb)

        head_emb = torch.index_select(ent_emb, 0, h)
        r_emb = torch.index_select(rel_emb, 0, r)#.view(-1, 1, self.ent_dim)
        tail_emb = torch.index_select(ent_emb, 0, t)
        
        x = rotate(head_emb, r_emb)
        cl_x = x
        # x = self.hid_drop(x)
        
        # x = F.relu(x, inplace=True)
        # ent_emb = self.bn2(ent_emb)
        x = torch.mm(x, ent_emb.transpose(1,0))
        x += self.b.expand_as(x)
        # x = torch.sigmoid(x)
        return cl_x, x, tail_emb, head_emb, ent_emb, rel_emb
    
    def predict_t(self, hr_batch, ent_emb, rel_emb, edge_index, edge_type, save_emb=False):
        
        with torch.no_grad():

            if self.layer==2:
                ent_emb1, rel_emb1, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
                ent_emb1 = self.hid_drop(ent_emb1)
                ent_emb2, rel_emb, alpha2 = self.gnn2(ent_emb1, edge_index, edge_type, rel_emb1, alpha1)
                ent_emb = self.hid_drop(ent_emb2)
            
            elif self.layer==1:
                ent_emb, rel_emb, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
                ent_emb = self.hid_drop(ent_emb)
            
            elif self.layer==4:
                ent_emb, rel_emb, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
                ent_emb = self.hid_drop(ent_emb)
                ent_emb, rel_emb, alpha2 = self.gnn2(ent_emb, edge_index, edge_type, rel_emb, alpha1)
                ent_emb = self.hid_drop(ent_emb)
                ent_emb, rel_emb, alpha3 = self.gnn3(ent_emb, edge_index, edge_type, rel_emb, alpha2)
                ent_emb = self.hid_drop(ent_emb)
                ent_emb, rel_emb, alpha4 = self.gnn4(ent_emb, edge_index, edge_type, rel_emb, alpha3)
                ent_emb = self.hid_drop(ent_emb)

            if save_emb:
                save = {"ent_emb": ent_emb,
                        "rel_emb": rel_emb}
                torch.save(save, "./emb/saved_emb.pt")
                
            e1 = hr_batch[:, 0]
            rel = hr_batch[:, 1]
            e1_embedded= torch.index_select(ent_emb, 0, e1)#.view(-1, 1, self.ent_dim)
            rel_embedded = torch.index_select(rel_emb, 0, rel)#.view(-1, 1, self.ent_dim)

            x = rotate(e1_embedded, rel_embedded)

            x = torch.mm(x, ent_emb.transpose(1,0))
            x += self.b.expand_as(x)
            # x = torch.sigmoid(x)
            return x


class CLKG_compgatv3_distmult(nn.Module):
    def __init__(self, args):
        super(CLKG_compgatv3_distmult, self).__init__()
        self.ent_dim = args.ent_dim
        self.learning_rate = args.cl_lr
        self.batch_size = args.cl_batch_size
        self.op = args.op

        self.ent_emb = torch.nn.Embedding(args.ent_num, args.init_dim).to(torch.device("cuda"))
        self.rel_emb = torch.nn.Embedding(args.rel_num, args.init_dim).to(torch.device("cuda"))
        torch.nn.init.xavier_uniform_(self.ent_emb.weight)
        torch.nn.init.xavier_uniform_(self.rel_emb.weight)

        self.ent_dim = args.ent_dim
       
        if args.gcn_layer == 2:
            # self.encoder = Sequential("x, edge_index, edge_type, rel_emb",[
            #     (CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta), "x, edge_index, edge_type, rel_emb -> x, rel_emb, alpha"),
            #     (, "x -> x"),
            #     (CompGATv3(in_channels=args.ent_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta), "x, edge_index, edge_type, rel_emb, alpha -> x, rel_emb, alpha"),
            # ]).to(torch.device("cuda"))
            self.layer = 2
            self.gnn1 = CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            self.hid_drop = torch.nn.Dropout(args.hid_drop)
            self.gnn2 = CompGATv3(in_channels=args.ent_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            
        elif args.gcn_layer == 1:
            # self.encoder = Sequential("x, edge_index, edge_type, rel_emb",[
            #     (CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=self.op, beta=args.beta), "x, edge_index, edge_type, rel_emb -> x, rel_emb, alpha"),
            # ]).to(torch.device("cuda"))
            self.layer = 1
            self.gnn1 = CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            self.hid_drop = torch.nn.Dropout(args.hid_drop)
            
        self.encoder_drop = torch.nn.Dropout(p=args.encoder_hid_drop, inplace=False)
        # self.input_drop = torch.nn.Dropout(args.input_drop, inplace=False)
        self.fea_drop = torch.nn.Dropout(args.fea_drop, inplace=False)
        # self.hid_drop = torch.nn.Dropout(args.hid_drop)
        self.register_parameter('b', torch.nn.Parameter(torch.zeros(args.ent_num)))
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, edge_index, edge_type, h, r, t, ent_emb, rel_emb):
        if self.layer==2:
            ent_emb1, rel_emb1, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
            ent_emb1 = self.hid_drop(ent_emb1)
            ent_emb2, rel_emb, alpha2 = self.gnn2(ent_emb1, edge_index, edge_type, rel_emb1, alpha1)
            ent_emb = self.hid_drop(ent_emb2)
        
        elif self.layer==1:
            ent_emb, rel_emb, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
            ent_emb = self.hid_drop(ent_emb)

        head_emb = torch.index_select(ent_emb, 0, h)
        r_emb = torch.index_select(rel_emb, 0, r)
        tail_emb = torch.index_select(ent_emb, 0, t)
        
        x = head_emb * r_emb        
        cl_x = x
        # x = self.hid_drop(x)
        
        # x = F.relu(x, inplace=True)
        # ent_emb = self.bn2(ent_emb)
        x = torch.mm(x, ent_emb.transpose(1,0))
        x += self.b.expand_as(x)
        # x = torch.sigmoid(x)
        return cl_x, x, tail_emb, head_emb, ent_emb, rel_emb
    
    def score_hrt(self, hrt_batch, ent_emb, rel_emb):
        h,r,t = hrt_batch.t()
        head_emb = torch.index_select(ent_emb, 0, h)
        rel_emb = torch.index_select(rel_emb, 0, r)
        tail_emb = torch.index_select(ent_emb, 0, t)
        
        x = head_emb * rel_emb
        
        x = (F.normalize(x, dim=1)*F.normalize(tail_emb, dim=1)).sum(dim=1, keepdim=True)
        # x += self.b[t].unsqueeze(1)
        # x = torch.sigmoid(x)
        return x
    
    def predict_t(self, hr_batch, ent_emb, rel_emb, edge_index, edge_type, save_emb=False):
        
        with torch.no_grad():

            if self.layer==2:
                ent_emb1, rel_emb1, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
                ent_emb1 = self.hid_drop(ent_emb1)
                ent_emb2, rel_emb, alpha2 = self.gnn2(ent_emb1, edge_index, edge_type, rel_emb1, alpha1)
                ent_emb = self.hid_drop(ent_emb2)
            
            elif self.layer==1:
                ent_emb, rel_emb, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
                ent_emb = self.hid_drop(ent_emb)
            
            if save_emb:
                save = {"ent_emb": ent_emb,
                        "rel_emb": rel_emb}
                torch.save(save, "./emb/saved_emb.pt")
                
            e1 = hr_batch[:, 0]
            rel = hr_batch[:, 1]
            e1_embedded= torch.index_select(ent_emb, 0, e1)
            rel_embedded = torch.index_select(rel_emb, 0, rel)

            x = e1_embedded * rel_embedded
            x = torch.mm(x, ent_emb.transpose(1,0))
            x += self.b.expand_as(x)
            
            return x

class CLKG_compgatv3_transE(nn.Module):
    def __init__(self, args):
        super(CLKG_compgatv3_transE, self).__init__()
        self.ent_dim = args.ent_dim
        self.learning_rate = args.cl_lr
        self.batch_size = args.cl_batch_size
        self.op = args.op

        self.ent_emb = torch.nn.Embedding(args.ent_num, args.init_dim).to(torch.device("cuda"))
        self.rel_emb = torch.nn.Embedding(args.rel_num, args.init_dim).to(torch.device("cuda"))
        torch.nn.init.xavier_uniform_(self.ent_emb.weight)
        torch.nn.init.xavier_uniform_(self.rel_emb.weight)

        self.ent_dim = args.ent_dim
        
        if args.gcn_layer == 2:
            # self.encoder = Sequential("x, edge_index, edge_type, rel_emb",[
            #     (CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta), "x, edge_index, edge_type, rel_emb -> x, rel_emb, alpha"),
            #     (, "x -> x"),
            #     (CompGATv3(in_channels=args.ent_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta), "x, edge_index, edge_type, rel_emb, alpha -> x, rel_emb, alpha"),
            # ]).to(torch.device("cuda"))
            self.layer = 2
            self.gnn1 = CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            self.hid_drop = torch.nn.Dropout(args.hid_drop)
            self.gnn2 = CompGATv3(in_channels=args.ent_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            
        elif args.gcn_layer == 1:
            # self.encoder = Sequential("x, edge_index, edge_type, rel_emb",[
            #     (CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=self.op, beta=args.beta), "x, edge_index, edge_type, rel_emb -> x, rel_emb, alpha"),
            # ]).to(torch.device("cuda"))
            self.layer = 1
            self.gnn1 = CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            self.hid_drop = torch.nn.Dropout(args.hid_drop)
            
        self.encoder_drop = torch.nn.Dropout(p=args.encoder_hid_drop, inplace=False)
        # self.input_drop = torch.nn.Dropout(args.input_drop, inplace=False)
        self.fea_drop = torch.nn.Dropout(args.fea_drop, inplace=False)
        # self.hid_drop = torch.nn.Dropout(args.hid_drop)
        self.register_parameter('b', torch.nn.Parameter(torch.zeros(args.ent_num)))
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, edge_index, edge_type, h, r, t, ent_emb, rel_emb):
        if self.layer==2:
            ent_emb1, rel_emb1, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
            ent_emb1 = self.hid_drop(ent_emb1)
            ent_emb2, rel_emb, alpha2 = self.gnn2(ent_emb1, edge_index, edge_type, rel_emb1, alpha1)
            ent_emb = self.hid_drop(ent_emb2)
        
        elif self.layer==1:
            ent_emb, rel_emb, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
            ent_emb = self.hid_drop(ent_emb)

        head_emb = torch.index_select(ent_emb, 0, h)
        r_emb = torch.index_select(rel_emb, 0, r)
        tail_emb = torch.index_select(ent_emb, 0, t)
        
        x = head_emb + r_emb        
        cl_x = x
        
        x = torch.mm(x, ent_emb.transpose(1,0))
        x += self.b.expand_as(x)
        # x = torch.sigmoid(x)
        return cl_x, x, tail_emb, head_emb, ent_emb, rel_emb
    
    def score_hrt(self, hrt_batch, ent_emb, rel_emb):
        h,r,t = hrt_batch.t()
        head_emb = torch.index_select(ent_emb, 0, h)
        rel_emb = torch.index_select(rel_emb, 0, r)
        tail_emb = torch.index_select(ent_emb, 0, t)
        
        x = head_emb +  rel_emb
        
        x = (F.normalize(x, dim=1)*F.normalize(tail_emb, dim=1)).sum(dim=1, keepdim=True)
        # x += self.b[t].unsqueeze(1)
        # x = torch.sigmoid(x)
        return x
    
    def predict_t(self, hr_batch, ent_emb, rel_emb, edge_index, edge_type, save_emb=False):
        
        with torch.no_grad():

            if self.layer==2:
                ent_emb1, rel_emb1, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
                ent_emb1 = self.hid_drop(ent_emb1)
                ent_emb2, rel_emb, alpha2 = self.gnn2(ent_emb1, edge_index, edge_type, rel_emb1, alpha1)
                ent_emb = self.hid_drop(ent_emb2)
            
            elif self.layer==1:
                ent_emb, rel_emb, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
                ent_emb = self.hid_drop(ent_emb)
            
            if save_emb:
                save = {"ent_emb": ent_emb,
                        "rel_emb": rel_emb}
                torch.save(save, "./emb/saved_emb.pt")
                
            e1 = hr_batch[:, 0]
            rel = hr_batch[:, 1]
            e1_embedded= torch.index_select(ent_emb, 0, e1)
            rel_embedded = torch.index_select(rel_emb, 0, rel)

            x = e1_embedded + rel_embedded
            x = torch.mm(x, ent_emb.transpose(1,0))
            x += self.b.expand_as(x)
            
            return x


class CLKG_CompGCN_convE(nn.Module):
    def __init__(self, args):
        super(CLKG_CompGCN_convE, self).__init__()
        self.ent_dim = args.ent_dim
        self.learning_rate = args.cl_lr
        self.batch_size = args.cl_batch_size
        self.op = args.op

        self.ent_emb = torch.nn.Embedding(args.ent_num, args.init_dim).to(torch.device("cuda"))
        self.rel_emb = torch.nn.Embedding(args.rel_num, args.init_dim).to(torch.device("cuda"))
        torch.nn.init.xavier_uniform_(self.ent_emb.weight)
        torch.nn.init.xavier_uniform_(self.rel_emb.weight)

        self.ent_dim = args.ent_dim
        self.ent_h = args.ent_height
        self.ent_w = self.ent_dim // self.ent_h
        if args.gcn_layer == 2:
            # self.encoder = Sequential("x, edge_index, edge_type, rel_emb",[
            #     (CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta), "x, edge_index, edge_type, rel_emb -> x, rel_emb, alpha"),
            #     (, "x -> x"),
            #     (CompGATv3(in_channels=args.ent_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta), "x, edge_index, edge_type, rel_emb, alpha -> x, rel_emb, alpha"),
            # ]).to(torch.device("cuda"))
            self.layer = 2
            self.gnn1 = CompGCN(in_channels=args.ent_dim, out_channels=args.ent_dim, drop=args.encoder_drop, bias=True, op=args.op)
            self.hid_drop = torch.nn.Dropout(args.hid_drop)
            self.gnn2 = CompGCN(in_channels=args.ent_dim, out_channels=args.ent_dim, drop=args.encoder_drop, bias=True, op=args.op)
            
        elif args.gcn_layer == 1:
            # self.encoder = Sequential("x, edge_index, edge_type, rel_emb",[
            #     (CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=self.op, beta=args.beta), "x, edge_index, edge_type, rel_emb -> x, rel_emb, alpha"),
            # ]).to(torch.device("cuda"))
            self.layer = 1
            self.gnn1 = CompGCN(in_channels=args.ent_dim, out_channels=args.ent_dim, drop=args.encoder_drop, bias=True, op=args.op)
            self.hid_drop = torch.nn.Dropout(args.hid_drop)
            
        self.encoder_drop = torch.nn.Dropout(p=args.encoder_hid_drop, inplace=False)
        # self.input_drop = torch.nn.Dropout(args.input_drop, inplace=False)
        self.fea_drop = torch.nn.Dropout(args.fea_drop, inplace=False)
        # self.hid_drop = torch.nn.Dropout(args.hid_drop)

        self.conv1 = torch.nn.Conv2d(1, args.filter_channel, (args.filter_size, args.filter_size), 1, 0, bias=False)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(args.filter_channel)
        self.bn2 = torch.nn.BatchNorm1d(args.ent_dim)
        self.register_parameter('b', torch.nn.Parameter(torch.zeros(args.ent_num)))
       
        num_in_features = (args.filter_channel*(2*args.ent_height-args.filter_size+1)*(self.ent_w-args.filter_size+1))

        if args.proj == "mlp":
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(num_in_features, args.ent_dim),
                torch.nn.BatchNorm1d(args.ent_dim),
                torch.nn.PReLU(),
                torch.nn.Linear(args.ent_dim, args.ent_dim),
                torch.nn.BatchNorm1d(args.ent_dim))
        
        elif args.proj == "linear":
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(num_in_features, args.ent_dim),
                torch.nn.Dropout(args.hid_drop),
                torch.nn.BatchNorm1d(args.ent_dim),
                torch.nn.PReLU())
        else:
            raise ValueError("Type of projection head should be mlp or linear!")

        self.relu = torch.nn.PReLU()
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, edge_index, edge_type, h, r, t, ent_emb, rel_emb):
        if self.layer==2:
            ent_emb1, rel_emb1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
            ent_emb1 = self.hid_drop(ent_emb1)
            ent_emb2, rel_emb = self.gnn2(ent_emb1, edge_index, edge_type, rel_emb1)
            ent_emb = self.hid_drop(ent_emb2)
        
        elif self.layer==1:
            ent_emb, rel_emb = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
            ent_emb = self.hid_drop(ent_emb)

        head_emb = torch.index_select(ent_emb, 0, h)
        r_emb = torch.index_select(rel_emb, 0, r).view(-1, 1, self.ent_dim)
        tail_emb = torch.index_select(ent_emb, 0, t)
        
        x = head_emb.view(-1, 1, self.ent_dim)
        x = torch.cat([x, r_emb], 1)
        x	= torch.transpose(x, 2, 1).reshape((-1, 1, 2*self.ent_h, self.ent_w))

        x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fea_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.proj(x)
        x = self.bn2(x)
        cl_x = x
        # x = self.hid_drop(x)
        
        # x = F.relu(x, inplace=True)
        # ent_emb = self.bn2(ent_emb)
        x = torch.mm(x, ent_emb.transpose(1,0))
        x += self.b.expand_as(x)
        # x = torch.sigmoid(x)
        return cl_x, x, tail_emb, head_emb, ent_emb, rel_emb
    
    def score_hrt(self, hrt_batch, ent_emb, rel_emb):
        h,r,t = hrt_batch.t()
        head_emb = torch.index_select(ent_emb, 0, h)
        rel_emb = torch.index_select(rel_emb, 0, r).view(-1, 1, self.ent_dim)
        tail_emb = torch.index_select(ent_emb, 0, t)
        
        x = head_emb.view(-1, 1, self.ent_dim)
        x = torch.cat([x, rel_emb], 1)
        x	= torch.transpose(x, 2, 1).reshape((-1, 1, 2*self.ent_h, self.ent_w))

        x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fea_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.proj(x)
        # x = self.bn2(x)
        
        x = (F.normalize(x, dim=1)*F.normalize(tail_emb, dim=1)).sum(dim=1, keepdim=True)
        # x += self.b[t].unsqueeze(1)
        # x = torch.sigmoid(x)
        return x
    
    def predict_t(self, hr_batch, ent_emb, rel_emb, edge_index, edge_type, save_emb=False):
        
        with torch.no_grad():

            if self.layer==2:
                ent_emb1, rel_emb1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
                ent_emb1 = self.hid_drop(ent_emb1)
                ent_emb2, rel_emb = self.gnn2(ent_emb1, edge_index, edge_type, rel_emb1)
                ent_emb = self.hid_drop(ent_emb2)
            
            elif self.layer==1:
                ent_emb, rel_emb = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
                ent_emb = self.hid_drop(ent_emb)
                
            e1 = hr_batch[:, 0]
            rel = hr_batch[:, 1]
            e1_embedded= torch.index_select(ent_emb, 0, e1).view(-1, 1, self.ent_dim)
            rel_embedded = torch.index_select(rel_emb, 0, rel).view(-1, 1, self.ent_dim)

            x = torch.cat([e1_embedded, rel_embedded], 1)
            x	= torch.transpose(x, 2, 1).reshape((-1, 1, 2*self.ent_h, self.ent_w))
            
            x = self.bn0(x)
            # x = self.input_drop(x)
            x= self.conv1(x)
            x= self.bn1(x)
            x= self.relu(x)
            x = self.fea_drop(x)
            x = x.view(x.shape[0], -1)
            x = self.proj(x)
            x = self.bn2(x)
            # x = F.relu(x, inplace=True)
            # ent_emb = self.bn2(ent_emb)
            x = torch.mm(x, ent_emb.transpose(1,0))
            x += self.b.expand_as(x)
            # x = torch.sigmoid(x)
            return x

class CLKG_convE(nn.Module):
    def __init__(self, args):
        super(CLKG_CompGCN_convE, self).__init__()
        self.ent_dim = args.ent_dim
        self.learning_rate = args.cl_lr
        self.batch_size = args.cl_batch_size
        self.op = args.op

        self.ent_emb = torch.nn.Embedding(args.ent_num, args.init_dim).to(torch.device("cuda"))
        self.rel_emb = torch.nn.Embedding(args.rel_num, args.init_dim).to(torch.device("cuda"))
        torch.nn.init.xavier_uniform_(self.ent_emb.weight)
        torch.nn.init.xavier_uniform_(self.rel_emb.weight)

        self.ent_dim = args.ent_dim
        self.ent_h = args.ent_height
        self.ent_w = self.ent_dim // self.ent_h
            
        self.fea_drop = torch.nn.Dropout(args.fea_drop, inplace=False)
        # self.hid_drop = torch.nn.Dropout(args.hid_drop)

        self.conv1 = torch.nn.Conv2d(1, args.filter_channel, (args.filter_size, args.filter_size), 1, 0, bias=False)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(args.filter_channel)
        self.bn2 = torch.nn.BatchNorm1d(args.ent_dim)
        self.register_parameter('b', torch.nn.Parameter(torch.zeros(args.ent_num)))
       
        num_in_features = (args.filter_channel*(2*args.ent_height-args.filter_size+1)*(self.ent_w-args.filter_size+1))

        if args.proj == "mlp":
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(num_in_features, args.ent_dim),
                torch.nn.BatchNorm1d(args.ent_dim),
                torch.nn.PReLU(),
                torch.nn.Linear(args.ent_dim, args.ent_dim),
                torch.nn.BatchNorm1d(args.ent_dim))
        
        elif args.proj == "linear":
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(num_in_features, args.ent_dim),
                torch.nn.Dropout(args.hid_drop),
                torch.nn.BatchNorm1d(args.ent_dim),
                torch.nn.PReLU())
        else:
            raise ValueError("Type of projection head should be mlp or linear!")

        self.relu = torch.nn.PReLU()
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, edge_index, edge_type, h, r, t, ent_emb, rel_emb):

        head_emb = torch.index_select(ent_emb, 0, h)
        r_emb = torch.index_select(rel_emb, 0, r).view(-1, 1, self.ent_dim)
        tail_emb = torch.index_select(ent_emb, 0, t)
        
        x = head_emb.view(-1, 1, self.ent_dim)
        x = torch.cat([x, r_emb], 1)
        x	= torch.transpose(x, 2, 1).reshape((-1, 1, 2*self.ent_h, self.ent_w))

        x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fea_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.proj(x)
        x = self.bn2(x)
        cl_x = x

        x = torch.mm(x, ent_emb.transpose(1,0))
        x += self.b.expand_as(x)

        return cl_x, x, tail_emb, head_emb, ent_emb, rel_emb
    
    def predict_t(self, hr_batch, ent_emb, rel_emb, edge_index, edge_type, save_emb=False):
        
        with torch.no_grad():
                
            e1 = hr_batch[:, 0]
            rel = hr_batch[:, 1]
            e1_embedded= torch.index_select(ent_emb, 0, e1).view(-1, 1, self.ent_dim)
            rel_embedded = torch.index_select(rel_emb, 0, rel).view(-1, 1, self.ent_dim)

            x = torch.cat([e1_embedded, rel_embedded], 1)
            x	= torch.transpose(x, 2, 1).reshape((-1, 1, 2*self.ent_h, self.ent_w))
            
            x = self.bn0(x)
            # x = self.input_drop(x)
            x= self.conv1(x)
            x= self.bn1(x)
            x= self.relu(x)
            x = self.fea_drop(x)
            x = x.view(x.shape[0], -1)
            x = self.proj(x)
            x = self.bn2(x)
            # x = F.relu(x, inplace=True)
            # ent_emb = self.bn2(ent_emb)
            x = torch.mm(x, ent_emb.transpose(1,0))
            x += self.b.expand_as(x)
            # x = torch.sigmoid(x)
            return x

class CLKG_compgatv4_convE(nn.Module):
    def __init__(self, args, deg):
        super(CLKG_compgatv4_convE, self).__init__()
        self.ent_dim = args.ent_dim
        self.learning_rate = args.cl_lr
        self.batch_size = args.cl_batch_size
        self.op = args.op

        self.ent_emb = torch.nn.Embedding(args.ent_num, args.init_dim).to(torch.device("cuda"))
        self.rel_emb = torch.nn.Embedding(args.rel_num, args.init_dim).to(torch.device("cuda"))
        torch.nn.init.xavier_uniform_(self.ent_emb.weight)
        torch.nn.init.xavier_uniform_(self.rel_emb.weight)

        deg = torch.zeros(10, dtype=torch.long)

        
        self.ent_dim = args.ent_dim
        self.ent_h = args.ent_height
        self.ent_w = self.ent_dim // self.ent_h
        if args.gcn_layer == 2:
            # self.encoder = Sequential("x, edge_index, edge_type, rel_emb",[
            #     (CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta), "x, edge_index, edge_type, rel_emb -> x, rel_emb, alpha"),
            #     (, "x -> x"),
            #     (CompGATv3(in_channels=args.ent_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta), "x, edge_index, edge_type, rel_emb, alpha -> x, rel_emb, alpha"),
            # ]).to(torch.device("cuda"))
            self.layer = 2
            self.gnn1 = CompGATv4(in_channels=args.init_dim, out_channels=args.ent_dim, edge_dim=args.rel_dim, drop=args.encoder_drop, op=args.op, aggregators=["sum", "mean", "min", "max", "var", "std"], scalers=["attenuation"], deg=deg)
            self.hid_drop = torch.nn.Dropout(args.hid_drop)
            self.gnn2 = CompGATv4(in_channels=args.ent_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, op=args.op, aggregators=["sum", "mean", "min", "max", "var", "std"], scalers=["attenuation"], deg=deg)
            
        elif args.gcn_layer == 1:
            # self.encoder = Sequential("x, edge_index, edge_type, rel_emb",[
            #     (CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=self.op, beta=args.beta), "x, edge_index, edge_type, rel_emb -> x, rel_emb, alpha"),
            # ]).to(torch.device("cuda"))
            self.layer = 1
            self.gnn1 = CompGATv4(in_channels=args.init_dim, out_channels=args.ent_dim, edge_dim=args.rel_dim, drop=args.encoder_drop, op=args.op, aggregators=["sum", "mean", "min", "max", "var", "std"], scalers=["attenuation"], deg=deg)            
            self.hid_drop = torch.nn.Dropout(args.hid_drop)
            
        self.encoder_drop = torch.nn.Dropout(p=args.encoder_hid_drop, inplace=False)
        # self.input_drop = torch.nn.Dropout(args.input_drop, inplace=False)
        self.fea_drop = torch.nn.Dropout(args.fea_drop, inplace=False)
        # self.hid_drop = torch.nn.Dropout(args.hid_drop)

        self.conv1 = torch.nn.Conv2d(1, args.filter_channel, (args.filter_size, args.filter_size), 1, 0, bias=False)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(args.filter_channel)
        self.bn2 = torch.nn.BatchNorm1d(args.ent_dim)
        self.register_parameter('b', torch.nn.Parameter(torch.zeros(args.ent_num)))
       
        num_in_features = (args.filter_channel*(2*args.ent_height-args.filter_size+1)*(self.ent_w-args.filter_size+1))

        if args.proj == "mlp":
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(num_in_features, args.ent_dim),
                torch.nn.BatchNorm1d(args.ent_dim),
                torch.nn.PReLU(),
                torch.nn.Linear(args.ent_dim, args.ent_dim),
                torch.nn.BatchNorm1d(args.ent_dim))
        
        elif args.proj == "linear":
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(num_in_features, args.ent_dim),
                torch.nn.Dropout(args.hid_drop),
                torch.nn.BatchNorm1d(args.ent_dim),
                torch.nn.PReLU())
        else:
            raise ValueError("Type of projection head should be mlp or linear!")

        self.relu = torch.nn.PReLU()
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, edge_index, edge_type, h, r, t, ent_emb, rel_emb):
        if self.layer==2:
            ent_emb1, rel_emb1, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
            ent_emb1 = self.hid_drop(ent_emb1)
            ent_emb2, rel_emb, alpha2 = self.gnn2(ent_emb1, edge_index, edge_type, rel_emb1, alpha1)
            ent_emb = self.hid_drop(ent_emb2)
        
        elif self.layer==1:
            ent_emb, rel_emb, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
            ent_emb = self.hid_drop(ent_emb)

        head_emb = torch.index_select(ent_emb, 0, h)
        r_emb = torch.index_select(rel_emb, 0, r).view(-1, 1, self.ent_dim)
        tail_emb = torch.index_select(ent_emb, 0, t)
        
        x = head_emb.view(-1, 1, self.ent_dim)
        x = torch.cat([x, r_emb], 1)
        x	= torch.transpose(x, 2, 1).reshape((-1, 1, 2*self.ent_h, self.ent_w))

        x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fea_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.proj(x)
        x = self.bn2(x)
        cl_x = x
        # x = self.hid_drop(x)
        
        # x = F.relu(x, inplace=True)
        # ent_emb = self.bn2(ent_emb)
        x = torch.mm(x, ent_emb.transpose(1,0))
        x += self.b.expand_as(x)
        # x = torch.sigmoid(x)
        return cl_x, x, tail_emb, head_emb, ent_emb, rel_emb
    
    def score_hrt(self, hrt_batch, ent_emb, rel_emb):
        h,r,t = hrt_batch.t()
        head_emb = torch.index_select(ent_emb, 0, h)
        rel_emb = torch.index_select(rel_emb, 0, r).view(-1, 1, self.ent_dim)
        tail_emb = torch.index_select(ent_emb, 0, t)
        
        x = head_emb.view(-1, 1, self.ent_dim)
        x = torch.cat([x, rel_emb], 1)
        x	= torch.transpose(x, 2, 1).reshape((-1, 1, 2*self.ent_h, self.ent_w))

        x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fea_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.proj(x)
        # x = self.bn2(x)
        
        x = (F.normalize(x, dim=1)*F.normalize(tail_emb, dim=1)).sum(dim=1, keepdim=True)
        # x += self.b[t].unsqueeze(1)
        # x = torch.sigmoid(x)
        return x
    
    def predict_t(self, hr_batch, ent_emb, rel_emb, edge_index, edge_type, save_emb=False):
        
        with torch.no_grad():

            if self.layer==2:
                ent_emb1, rel_emb1, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
                ent_emb1 = self.hid_drop(ent_emb1)
                ent_emb2, rel_emb, alpha2 = self.gnn2(ent_emb1, edge_index, edge_type, rel_emb1, alpha1)
                ent_emb = self.hid_drop(ent_emb2)
            
            elif self.layer==1:
                ent_emb, rel_emb, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
                ent_emb = self.hid_drop(ent_emb)
            
            if save_emb:
                save = {"ent_emb": ent_emb,
                        "rel_emb": rel_emb}
                torch.save(save, "./emb/saved_emb.pt")
                
            e1 = hr_batch[:, 0]
            rel = hr_batch[:, 1]
            e1_embedded= torch.index_select(ent_emb, 0, e1).view(-1, 1, self.ent_dim)
            rel_embedded = torch.index_select(rel_emb, 0, rel).view(-1, 1, self.ent_dim)

            x = torch.cat([e1_embedded, rel_embedded], 1)
            x	= torch.transpose(x, 2, 1).reshape((-1, 1, 2*self.ent_h, self.ent_w))
            
            x = self.bn0(x)
            # x = self.input_drop(x)
            x= self.conv1(x)
            x= self.bn1(x)
            x= self.relu(x)
            x = self.fea_drop(x)
            x = x.view(x.shape[0], -1)
            x = self.proj(x)
            x = self.bn2(x)
            # x = F.relu(x, inplace=True)
            # ent_emb = self.bn2(ent_emb)
            x = torch.mm(x, ent_emb.transpose(1,0))
            x += self.b.expand_as(x)
            # x = torch.sigmoid(x)
            return x

class CompGCN_conve(nn.Module):
    def __init__(self, args, cl_ent, cl_rel):
        super(CompGCN_conve, self).__init__()
        self.ent_emb = torch.nn.Embedding(args.ent_num, args.ent_dim).to(torch.device("cuda"))
        self.rel_emb = torch.nn.Embedding(args.rel_num, args.ent_dim).to(torch.device("cuda"))
        torch.nn.init.xavier_uniform_(self.ent_emb.weight)
        torch.nn.init.xavier_uniform_(self.rel_emb.weight)

        self.cl_ent = cl_ent
        self.cl_rel = cl_rel

        self.evaluator = Evaluator(num_ent=args.ent_num, batch_size=2048)
        if args.gcn_layer == 2:
            self.encoder = Sequential("x, edge_index, edge_type, rel_emb",[
                (CompGCN(in_channels=args.ent_dim, out_channels=args.ent_dim, drop=args.encoder_drop, bias=True, op=args.op), "x, edge_index, edge_type, rel_emb -> x, rel_emb"),
                (torch.nn.Dropout(p=args.encoder_hid_drop), "x -> x"),
                (CompGCN(in_channels=args.ent_dim, out_channels=args.ent_dim, drop=args.encoder_drop, bias=True, op=args.op), "x, edge_index, edge_type, rel_emb -> x, rel_emb"),
            ]).to(torch.device("cuda"))
        
        elif args.gcn_layer == 1:
            self.encoder = Sequential("x, edge_index, edge_type, rel_emb",[
                (CompGCN(in_channels=args.ent_dim, out_channels=args.ent_dim, drop=args.encoder_drop, bias=True, op=self.op), "x, edge_index, edge_type, rel_emb -> x, rel_emb"),
            ]).to(torch.device("cuda"))

        self.ent_dim = args.ent_dim
        self.ent_h = args.ent_height
        self.ent_w = self.ent_dim // self.ent_h

        self.input_drop = torch.nn.Dropout(args.input_drop)
        self.fea_drop = torch.nn.Dropout(args.fea_drop)
        self.hid_drop = torch.nn.Dropout(args.hid_drop)

        self.conv1 = torch.nn.Conv2d(1, args.filter_channel, (args.filter_size, args.filter_size), 1, 0, bias=True)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(args.filter_channel)
        self.bn2 = torch.nn.BatchNorm1d(args.ent_dim)
        self.register_parameter('b', torch.nn.Parameter(torch.zeros(args.ent_num)))
       
        num_in_features = (args.filter_channel*(2*args.ent_height-args.filter_size+1)*(self.ent_w-args.filter_size+1))
        self.fc = torch.nn.Linear(num_in_features, args.ent_dim)

    def forward(self, edge_index, edge_type, h, r, t):

        ent_emb, rel_emb = self.encoder(self.ent_emb.weight, edge_index, edge_type, self.rel_emb.weight)
        ent_emb = self.fea_drop(ent_emb)

        ent_emb += self.cl_ent
        rel_emb += self.cl_rel

        head_emb = ent_emb[h].view(-1, 1, self.ent_dim)
        rel_emb = rel_emb[r].view(-1, 1, self.ent_dim)

        stack_input = torch.cat([head_emb, rel_emb], 1)
        stack_input	= torch.transpose(stack_input, 2, 1).reshape((-1, 1, 2*self.ent_h, self.ent_w))

        stacked_inputs = self.bn0(stack_input)
        x= self.input_drop(stacked_inputs)
        x= self.conv1(x)
        x= self.bn1(x)
        x= F.relu(x)
        x = self.fea_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hid_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, ent_emb.transpose(1,0))
        x += self.b.expand_as(x)

        return x

    def predict_t(self, hr_batch, ent_emb, rel_emb, edge_index, edge_type):
        
        ent_emb, rel_emb = self.encoder(self.ent_emb.weight, edge_index, edge_type, self.rel_emb.weight)
        
        ent_emb += self.cl_ent
        rel_emb += self.cl_rel
        
        e1 = hr_batch[:, 0]
        rel = hr_batch[:, 1]
        e1_embedded= ent_emb[e1].view(-1, 1, self.ent_dim)
        rel_embedded = rel_emb[rel].view(-1, 1, self.ent_dim)

        stack_input = torch.cat([e1_embedded, rel_embedded], 1)
        stack_input	= torch.transpose(stack_input, 2, 1).reshape((-1, 1, 2*self.ent_h, self.ent_w))
        x = self.bn0(stack_input)
        x= self.input_drop(x)
        x= self.conv1(x)
        x= self.bn1(x)
        x= F.relu(x)
        x = self.fea_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hid_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, ent_emb.transpose(1,0))
        x += self.b.expand_as(x)
        return x


class Compgatv3_convE(nn.Module):
    def __init__(self, args):
        super(Compgatv3_convE, self).__init__()
        self.ent_dim = args.ent_dim
        self.learning_rate = args.cl_lr
        self.batch_size = args.cl_batch_size
        self.op = args.op

        self.ent_emb = torch.nn.Embedding(args.ent_num, args.init_dim).to(torch.device("cuda"))
        self.rel_emb = torch.nn.Embedding(args.rel_num, args.init_dim).to(torch.device("cuda"))
        torch.nn.init.xavier_uniform_(self.ent_emb.weight)
        torch.nn.init.xavier_uniform_(self.rel_emb.weight)

        self.ent_dim = args.ent_dim
        self.ent_h = args.ent_height
        self.ent_w = self.ent_dim // self.ent_h
        if args.gcn_layer == 2:
            # self.encoder = Sequential("x, edge_index, edge_type, rel_emb",[
            #     (CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta), "x, edge_index, edge_type, rel_emb -> x, rel_emb, alpha"),
            #     (, "x -> x"),
            #     (CompGATv3(in_channels=args.ent_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta), "x, edge_index, edge_type, rel_emb, alpha -> x, rel_emb, alpha"),
            # ]).to(torch.device("cuda"))
            self.layer = 2
            self.gnn1 = CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            self.hid_drop = torch.nn.Dropout(args.hid_drop)
            self.gnn2 = CompGATv3(in_channels=args.ent_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            
        elif args.gcn_layer == 1:
            # self.encoder = Sequential("x, edge_index, edge_type, rel_emb",[
            #     (CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=self.op, beta=args.beta), "x, edge_index, edge_type, rel_emb -> x, rel_emb, alpha"),
            # ]).to(torch.device("cuda"))
            self.layer = 1
            self.gnn1 = CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            self.hid_drop = torch.nn.Dropout(args.hid_drop)
            
        self.encoder_drop = torch.nn.Dropout(p=args.encoder_hid_drop, inplace=False)
        # self.input_drop = torch.nn.Dropout(args.input_drop, inplace=False)
        self.fea_drop = torch.nn.Dropout(args.fea_drop, inplace=False)
        # self.hid_drop = torch.nn.Dropout(args.hid_drop)

        self.conv1 = torch.nn.Conv2d(1, args.filter_channel, (args.filter_size, args.filter_size), 1, 0, bias=False)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(args.filter_channel)
        self.bn2 = torch.nn.BatchNorm1d(args.ent_dim)
        self.register_parameter('b', torch.nn.Parameter(torch.zeros(args.ent_num)))
       
        num_in_features = (args.filter_channel*(2*args.ent_height-args.filter_size+1)*(self.ent_w-args.filter_size+1))

        if args.proj == "mlp":
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(num_in_features, args.ent_dim),
                torch.nn.BatchNorm1d(args.ent_dim),
                torch.nn.PReLU(),
                torch.nn.Linear(args.ent_dim, args.ent_dim),
                torch.nn.BatchNorm1d(args.ent_dim))
        
        elif args.proj == "linear":
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(num_in_features, args.ent_dim),
                torch.nn.Dropout(args.hid_drop),
                torch.nn.BatchNorm1d(args.ent_dim),
                torch.nn.PReLU())
        else:
            raise ValueError("Type of projection head should be mlp or linear!")

        self.relu = torch.nn.PReLU()
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, edge_index, edge_type, h, r, t, ent_emb, rel_emb):
        if self.layer==2:
            ent_emb1, rel_emb1, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
            ent_emb1 = self.hid_drop(ent_emb1)
            ent_emb2, rel_emb, alpha2 = self.gnn2(ent_emb1, edge_index, edge_type, rel_emb1, alpha1)
            ent_emb = self.hid_drop(ent_emb2)
        
        elif self.layer==1:
            ent_emb, rel_emb, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
            ent_emb = self.hid_drop(ent_emb)

        head_emb = torch.index_select(ent_emb, 0, h)
        r_emb = torch.index_select(rel_emb, 0, r).view(-1, 1, self.ent_dim)
        tail_emb = torch.index_select(ent_emb, 0, t)
        
        x = head_emb.view(-1, 1, self.ent_dim)
        x = torch.cat([x, r_emb], 1)
        x	= torch.transpose(x, 2, 1).reshape((-1, 1, 2*self.ent_h, self.ent_w))

        x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fea_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.proj(x)
        x = self.bn2(x)
        cl_x = x
        # x = self.hid_drop(x)
        
        # x = F.relu(x, inplace=True)
        # ent_emb = self.bn2(ent_emb)
        x = torch.mm(x, ent_emb.transpose(1,0))
        x += self.b.expand_as(x)
        # x = torch.sigmoid(x)
        return cl_x, x, tail_emb, head_emb, ent_emb, rel_emb
    
    def score_hrt(self, hrt_batch, ent_emb, rel_emb):
        h,r,t = hrt_batch.t()
        head_emb = torch.index_select(ent_emb, 0, h)
        rel_emb = torch.index_select(rel_emb, 0, r).view(-1, 1, self.ent_dim)
        tail_emb = torch.index_select(ent_emb, 0, t)
        
        x = head_emb.view(-1, 1, self.ent_dim)
        x = torch.cat([x, rel_emb], 1)
        x	= torch.transpose(x, 2, 1).reshape((-1, 1, 2*self.ent_h, self.ent_w))

        x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fea_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.proj(x)
        # x = self.bn2(x)
        
        x = (F.normalize(x, dim=1)*F.normalize(tail_emb, dim=1)).sum(dim=1, keepdim=True)
        # x += self.b[t].unsqueeze(1)
        # x = torch.sigmoid(x)
        return x
    
    def predict_t(self, hr_batch, ent_emb, rel_emb, edge_index, edge_type, save_emb=False):
        
        with torch.no_grad():

            if self.layer==2:
                ent_emb1, rel_emb1, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
                ent_emb1 = self.hid_drop(ent_emb1)
                ent_emb2, rel_emb, alpha2 = self.gnn2(ent_emb1, edge_index, edge_type, rel_emb1, alpha1)
                ent_emb = self.hid_drop(ent_emb2)
            
            elif self.layer==1:
                ent_emb, rel_emb, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
                ent_emb = self.hid_drop(ent_emb)
            
            if save_emb:
                save = {"ent_emb": ent_emb,
                        "rel_emb": rel_emb}
                torch.save(save, "./emb/saved_emb.pt")
                
            e1 = hr_batch[:, 0]
            rel = hr_batch[:, 1]
            e1_embedded= torch.index_select(ent_emb, 0, e1).view(-1, 1, self.ent_dim)
            rel_embedded = torch.index_select(rel_emb, 0, rel).view(-1, 1, self.ent_dim)

            x = torch.cat([e1_embedded, rel_embedded], 1)
            x	= torch.transpose(x, 2, 1).reshape((-1, 1, 2*self.ent_h, self.ent_w))
            
            x = self.bn0(x)
            # x = self.input_drop(x)
            x= self.conv1(x)
            x= self.bn1(x)
            x= self.relu(x)
            x = self.fea_drop(x)
            x = x.view(x.shape[0], -1)
            x = self.proj(x)
            x = self.bn2(x)
            # x = F.relu(x, inplace=True)
            # ent_emb = self.bn2(ent_emb)
            x = torch.mm(x, ent_emb.transpose(1,0))
            x += self.b.expand_as(x)
            # x = torch.sigmoid(x)
            return x


class CompGAT_conve(nn.Module):
    def __init__(self, args, cl_ent, cl_rel):
        super(CompGAT_conve, self).__init__()
        self.ent_emb = torch.nn.Embedding(args.ent_num, args.ent_dim).to(torch.device("cuda"))
        self.rel_emb = torch.nn.Embedding(args.rel_num, args.ent_dim).to(torch.device("cuda"))
        torch.nn.init.xavier_normal_(self.ent_emb.weight)
        torch.nn.init.xavier_normal_(self.rel_emb.weight)

        self.cl_ent = cl_ent
        self.cl_rel = cl_rel

        self.evaluator = Evaluator(num_ent=args.ent_num, batch_size=2048)
        if args.gcn_layer == 2:
            self.encoder = Sequential("x, edge_index, edge_type, rel_emb",[
                (CompGATv3(in_channels=args.ent_dim, out_channels=args.ent_dim, drop=args.encoder_drop, bias=True, op=args.op, rel_dim=args.rel_dim), "x, edge_index, edge_type, rel_emb -> x, rel_emb"),
                (torch.nn.Dropout(p=args.encoder_hid_drop), "x -> x"),
                (CompGATv3(in_channels=args.ent_dim, out_channels=args.ent_dim, drop=args.encoder_drop, bias=True, op=args.op, rel_dim=args.rel_dim), "x, edge_index, edge_type, rel_emb -> x, rel_emb"),
                # (torch.nn.Dropout(p=args.encoder_drop), "x -> x")
            ]).to(torch.device("cuda"))
        
        elif args.gcn_layer == 1:
            self.encoder = Sequential("x, edge_index, edge_type, rel_emb",[
                (CompGAT(in_channels=args.ent_dim, out_channels=args.ent_dim, drop=args.encoder_drop, bias=True, op=args.op, rel_dim=args.rel_dim), "x, edge_index, edge_type, rel_emb -> x, rel_emb"),
            ]).to(torch.device("cuda"))

        self.ent_dim = args.ent_dim
        self.ent_h = args.ent_height
        self.ent_w = self.ent_dim // self.ent_h

        self.input_drop = torch.nn.Dropout(args.input_drop)
        self.fea_drop = torch.nn.Dropout(args.fea_drop)
        self.hid_drop = torch.nn.Dropout(args.hid_drop)

        self.conv1 = torch.nn.Conv2d(1, args.filter_channel, (args.filter_size, args.filter_size), 1, 0, bias=True)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(args.filter_channel)
        self.bn2 = torch.nn.BatchNorm1d(args.ent_dim)
        self.register_parameter('b', torch.nn.Parameter(torch.zeros(args.ent_num)))
       
        num_in_features = (args.filter_channel*(2*args.ent_height-args.filter_size+1)*(self.ent_w-args.filter_size+1))
        self.fc = torch.nn.Linear(num_in_features, args.ent_dim)

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, edge_index, edge_type, h, r, t):

        ent_emb, rel_emb = self.encoder(self.ent_emb.weight, edge_index, edge_type, self.rel_emb.weight)
        # ent_emb = self.fea_drop(ent_emb)

        # ent_emb += self.cl_ent
        # rel_emb += self.cl_rel

        head_emb = torch.index_select(ent_emb, 0, h).view(-1, 1, self.ent_dim)
        rel_emb = torch.index_select(rel_emb, 0, r).view(-1, 1, self.ent_dim)

        stack_input = torch.cat([head_emb, rel_emb], 1)
        stack_input	= torch.transpose(stack_input, 2, 1).reshape((-1, 1, 2*self.ent_h, self.ent_w))

        x = self.bn0(stack_input)
        x= self.input_drop(x)
        x= self.conv1(x)
        x= self.bn1(x)
        x= F.relu(x)
        x = self.fea_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hid_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, ent_emb.transpose(1,0))
        x += self.b.expand_as(x)

        return x

    def predict_t(self, hr_batch, ent_emb, rel_emb, edge_index, edge_type):
        
        ent_emb, rel_emb = self.encoder(self.ent_emb.weight, edge_index, edge_type, self.rel_emb.weight)
        # ent_emb = self.fea_drop(ent_emb)

        # ent_emb += self.cl_ent
        # rel_emb += self.cl_rel
        
        e1 = hr_batch[:, 0]
        rel = hr_batch[:, 1]
        e1_embedded= torch.index_select(ent_emb, 0, e1).view(-1, 1, self.ent_dim)
        rel_embedded = torch.index_select(rel_emb, 0, rel).view(-1, 1, self.ent_dim)

        stack_input = torch.cat([e1_embedded, rel_embedded], 1)
        stack_input	= torch.transpose(stack_input, 2, 1).reshape((-1, 1, 2*self.ent_h, self.ent_w))
        
        x = self.bn0(stack_input)
        x = self.input_drop(x)
        x= self.conv1(x)
        x= self.bn1(x)
        x= F.relu(x)
        x = self.fea_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hid_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, ent_emb.transpose(1,0))
        x += self.b.expand_as(x)

        return x

class CompGAT_conve_res(nn.Module):
    def __init__(self, args, cl_ent, cl_rel):
        super(CompGAT_conve_res, self).__init__()
        self.ent_emb = torch.nn.Embedding(args.ent_num, args.ent_dim).to(torch.device("cuda"))
        self.rel_emb = torch.nn.Embedding(args.rel_num, args.ent_dim).to(torch.device("cuda"))
        torch.nn.init.xavier_normal_(self.ent_emb.weight)
        torch.nn.init.xavier_normal_(self.rel_emb.weight)

        self.cl_ent = cl_ent
        self.cl_rel = cl_rel

        self.evaluator = Evaluator(num_ent=args.ent_num, batch_size=2048)
        if args.gcn_layer == 2:
            self.encoder = Sequential("x, edge_index, edge_type, rel_emb",[
                (CompGATv2(in_channels=args.ent_dim, out_channels=args.ent_dim, drop=args.encoder_drop, bias=args.bias, op=args.op, rel_dim=args.rel_dim), "x, edge_index, edge_type, rel_emb -> x, rel_emb"),
                (torch.nn.Dropout(p=args.encoder_hid_drop), "x -> x"),
                (CompGATv2(in_channels=args.ent_dim, out_channels=args.ent_dim, drop=args.encoder_drop, bias=args.bias, op=args.op, rel_dim=args.rel_dim), "x, edge_index, edge_type, rel_emb -> x, rel_emb"),
                # (torch.nn.Dropout(p=args.encoder_drop), "x -> x")
            ]).to(torch.device("cuda"))
        
        elif args.gcn_layer == 1:
            self.encoder = Sequential("x, edge_index, edge_type, rel_emb",[
                (CompGATv2(in_channels=args.ent_dim, out_channels=args.ent_dim, drop=args.encoder_drop, bias=args.bias, op=args.op, rel_dim=args.rel_dim), "x, edge_index, edge_type, rel_emb -> x, rel_emb"),
            ]).to(torch.device("cuda"))

        self.ent_dim = args.ent_dim
        self.ent_h = args.ent_height
        self.ent_w = self.ent_dim // self.ent_h

        self.input_drop = torch.nn.Dropout(args.input_drop)
        self.fea_drop = torch.nn.Dropout(args.fea_drop)
        self.hid_drop = torch.nn.Dropout(args.hid_drop)

        self.conv1 = torch.nn.Conv2d(1, args.filter_channel, (args.filter_size, args.filter_size), 1, 0, bias=True)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(args.filter_channel)
        self.bn2 = torch.nn.BatchNorm1d(args.ent_dim)
        self.register_parameter('b', torch.nn.Parameter(torch.zeros(args.ent_num)))
       
        self.flat_sz = (args.filter_channel*(2*args.ent_height-args.filter_size+1)*(self.ent_w-args.filter_size+1))
        self.fc = torch.nn.Linear(self.flat_sz, args.ent_dim)

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    
    def forward(self, edge_index, edge_type, h, r, t):

        ent_emb, rel_emb = self.encoder(self.ent_emb.weight, edge_index, edge_type, self.rel_emb.weight)
        # ent_emb = self.fea_drop(ent_emb)

        # ent_emb += self.cl_ent
        # rel_emb += self.cl_rel

        head_emb = torch.index_select(ent_emb, 0, h).view(-1, 1, self.ent_dim)
        rel_emb = torch.index_select(rel_emb, 0, r).view(-1, 1, self.ent_dim)

        stack_input = torch.cat([head_emb, rel_emb], 1)
        stack_input	= torch.transpose(stack_input, 2, 1).reshape((-1, 1, 2*self.ent_h, self.ent_w))

        x = self.bn0(stack_input)
        x= self.input_drop(x)
        x= self.conv1(x)
        x= self.bn1(x)
        x= F.relu(x)
        x = self.fea_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hid_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, ent_emb.transpose(1,0))
        x += self.b.expand_as(x)

        return x

    def predict_t(self, hr_batch, ent_emb, rel_emb, edge_index, edge_type):
        
        ent_emb, rel_emb = self.encoder(self.ent_emb.weight, edge_index, edge_type, self.rel_emb.weight)
        # ent_emb = self.fea_drop(ent_emb)

        # ent_emb += self.cl_ent
        # rel_emb += self.cl_rel
        
        e1 = hr_batch[:, 0]
        rel = hr_batch[:, 1]
        e1_embedded= torch.index_select(ent_emb, 0, e1).view(-1, 1, self.ent_dim)
        rel_embedded = torch.index_select(rel_emb, 0, rel).view(-1, 1, self.ent_dim)

        stack_input = torch.cat([e1_embedded, rel_embedded], 1)
        stack_input	= torch.transpose(stack_input, 2, 1).reshape((-1, 1, 2*self.ent_h, self.ent_w))
        
        x = self.bn0(stack_input)
        x = self.input_drop(x)
        x= self.conv1(x)
        x= self.bn1(x)
        x= F.relu(x)
        x = self.fea_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hid_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, ent_emb.transpose(1,0))
        x += self.b.expand_as(x)

        return x

class CLKG_transformer_convE(nn.Module):
    def __init__(self, args):
        super(CLKG_transformer_convE, self).__init__()
        self.ent_dim = args.ent_dim
        self.learning_rate = args.cl_lr
        self.batch_size = args.cl_batch_size
        self.op = args.op

        self.ent_emb = torch.nn.Embedding(args.ent_num, args.init_dim).to(torch.device("cuda"))
        self.rel_emb = torch.nn.Embedding(args.rel_num, args.init_dim).to(torch.device("cuda"))
        torch.nn.init.xavier_uniform_(self.ent_emb.weight)
        torch.nn.init.xavier_uniform_(self.rel_emb.weight)

        self.ent_dim = args.ent_dim
        self.ent_h = args.ent_height
        self.ent_w = self.ent_dim // self.ent_h
        self.adapt = torch.nn.Linear(2*args.ent_dim, args.ent_dim)
        if args.gcn_layer == 2:
            # self.encoder = Sequential("x, edge_index, edge_type, rel_emb",[
            #     (CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta), "x, edge_index, edge_type, rel_emb -> x, rel_emb, alpha"),
            #     (, "x -> x"),
            #     (CompGATv3(in_channels=args.ent_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta), "x, edge_index, edge_type, rel_emb, alpha -> x, rel_emb, alpha"),
            # ]).to(torch.device("cuda"))
            self.layer = 2
            self.gnn1 = Transformer(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            self.hid_drop = torch.nn.Dropout(args.hid_drop)
            self.gnn2 = Transformer(in_channels=args.ent_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            
        elif args.gcn_layer == 1:
            # self.encoder = Sequential("x, edge_index, edge_type, rel_emb",[
            #     (CompGATv3(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=self.op, beta=args.beta), "x, edge_index, edge_type, rel_emb -> x, rel_emb, alpha"),
            # ]).to(torch.device("cuda"))
            self.layer = 1
            self.gnn1 = Transformer(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            self.hid_drop = torch.nn.Dropout(args.hid_drop)
        
        elif args.gcn_layer == 4:
            self.layer = 4
            self.gnn1 = Transformer(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            self.hid_drop = torch.nn.Dropout(args.hid_drop)
            self.gnn2 = Transformer(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            self.gnn3 = Transformer(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)
            self.gnn4 = Transformer(in_channels=args.init_dim, out_channels=args.ent_dim, rel_dim=args.rel_dim, drop=args.encoder_drop, bias=True, op=args.op, beta=args.beta)

        self.encoder_drop = torch.nn.Dropout(p=args.encoder_hid_drop, inplace=False)
        # self.input_drop = torch.nn.Dropout(args.input_drop, inplace=False)
        self.fea_drop = torch.nn.Dropout(args.fea_drop, inplace=False)
        # self.hid_drop = torch.nn.Dropout(args.hid_drop)

        self.conv1 = torch.nn.Conv2d(1, args.filter_channel, (args.filter_size, args.filter_size), 1, 0, bias=False)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(args.filter_channel)
        self.bn2 = torch.nn.BatchNorm1d(args.ent_dim)
        self.register_parameter('b', torch.nn.Parameter(torch.zeros(args.ent_num)))
       
        num_in_features = (args.filter_channel*(2*args.ent_height-args.filter_size+1)*(self.ent_w-args.filter_size+1))

        if args.proj == "mlp":
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(num_in_features, args.ent_dim),
                torch.nn.BatchNorm1d(args.ent_dim),
                torch.nn.PReLU(),
                torch.nn.Linear(args.ent_dim, args.ent_dim),
                torch.nn.BatchNorm1d(args.ent_dim))
        
        elif args.proj == "linear":
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(num_in_features, args.ent_dim),
                torch.nn.Dropout(args.hid_drop),
                torch.nn.BatchNorm1d(args.ent_dim),
                torch.nn.PReLU())
        else:
            raise ValueError("Type of projection head should be mlp or linear!")

        self.relu = torch.nn.PReLU()
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, edge_index, edge_type, h, r, t, ent_emb, rel_emb):
        if self.layer==2:
            ent_emb1, rel_emb1, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
            ent_emb1 = self.hid_drop(ent_emb1)
            ent_emb2, rel_emb, alpha2 = self.gnn2(ent_emb1, edge_index, edge_type, rel_emb1, alpha1)
            ent_emb = self.hid_drop(ent_emb2)
        
        elif self.layer==1:
            ent_emb, rel_emb, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
            ent_emb = self.hid_drop(ent_emb)
        
        elif self.layer == 4:
            ent_emb, rel_emb, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
            ent_emb = self.hid_drop(ent_emb)
            ent_emb, rel_emb, alpha2 = self.gnn2(ent_emb, edge_index, edge_type, rel_emb, alpha1)
            ent_emb = self.hid_drop(ent_emb)
            ent_emb, rel_emb, alpha3 = self.gnn3(ent_emb, edge_index, edge_type, rel_emb, alpha2)
            ent_emb = self.hid_drop(ent_emb)
            ent_emb, rel_emb, alpha4 = self.gnn4(ent_emb, edge_index, edge_type, rel_emb, alpha3)
            ent_emb = self.hid_drop(ent_emb)

        head_emb = torch.index_select(ent_emb, 0, h)
        rel_emb = torch.index_select(rel_emb, 0, r).view(-1, 1, self.ent_dim)
        tail_emb = torch.index_select(ent_emb, 0, t)
        
        x = head_emb.view(-1, 1, self.ent_dim)
        x = torch.cat([x, rel_emb], 1)
        x	= torch.transpose(x, 2, 1).reshape((-1, 1, 2*self.ent_h, self.ent_w))

        x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fea_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.proj(x)
        x = self.bn2(x)
        cl_x = x
        # x = self.hid_drop(x)
        
        # x = F.relu(x, inplace=True)
        # ent_emb = self.bn2(ent_emb)
        x = torch.mm(x, ent_emb.transpose(1,0))
        x += self.b.expand_as(x)
        # x = torch.sigmoid(x)
        return cl_x, x, tail_emb, head_emb
    
    def predict_t(self, hr_batch, ent_emb, rel_emb, edge_index, edge_type, save_emb=False):
        
        with torch.no_grad():

            if self.layer==2:
                ent_emb1, rel_emb1, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
                ent_emb1 = self.hid_drop(ent_emb1)
                ent_emb2, rel_emb, alpha2 = self.gnn2(ent_emb1, edge_index, edge_type, rel_emb1, alpha1)
                ent_emb = self.hid_drop(ent_emb2)
            
            elif self.layer==1:
                ent_emb, rel_emb, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
                ent_emb = self.hid_drop(ent_emb)
            
            elif self.layer==4:
                ent_emb, rel_emb, alpha1 = self.gnn1(ent_emb, edge_index, edge_type, rel_emb)
                ent_emb = self.hid_drop(ent_emb)
                ent_emb, rel_emb, alpha2 = self.gnn2(ent_emb, edge_index, edge_type, rel_emb, alpha1)
                ent_emb = self.hid_drop(ent_emb)
                ent_emb, rel_emb, alpha3 = self.gnn3(ent_emb, edge_index, edge_type, rel_emb, alpha2)
                ent_emb = self.hid_drop(ent_emb)
                ent_emb, rel_emb, alpha4 = self.gnn4(ent_emb, edge_index, edge_type, rel_emb, alpha3)
                ent_emb = self.hid_drop(ent_emb)

            if save_emb:
                save = {"ent_emb": ent_emb,
                        "rel_emb": rel_emb}
                torch.save(save, "./emb/saved_emb.pt")
                
            e1 = hr_batch[:, 0]
            rel = hr_batch[:, 1]
            e1_embedded= torch.index_select(ent_emb, 0, e1).view(-1, 1, self.ent_dim)
            rel_embedded = torch.index_select(rel_emb, 0, rel).view(-1, 1, self.ent_dim)

            x = torch.cat([e1_embedded, rel_embedded], 1)
            x	= torch.transpose(x, 2, 1).reshape((-1, 1, 2*self.ent_h, self.ent_w))
            
            x = self.bn0(x)
            # x = self.input_drop(x)
            x= self.conv1(x)
            x= self.bn1(x)
            x= self.relu(x)
            x = self.fea_drop(x)
            x = x.view(x.shape[0], -1)
            x = self.proj(x)
            x = self.bn2(x)
            # x = F.relu(x, inplace=True)
            # ent_emb = self.bn2(ent_emb)
            x = torch.mm(x, ent_emb.transpose(1,0))
            x += self.b.expand_as(x)
            # x = torch.sigmoid(x)
            return x




class ConvE(nn.Module):
    def __init__(self, args):
        super(ConvE, self).__init__()

        self.ent_emb = torch.nn.Embedding(args.ent_num, args.ent_dim, padding_idx=0).to(torch.device("cuda"))
        self.rel_emb = torch.nn.Embedding(args.rel_num, args.rel_dim, padding_idx=0).to(torch.device("cuda"))
        torch.nn.init.xavier_normal_(self.ent_emb.weight)
        torch.nn.init.xavier_normal_(self.rel_emb.weight)

        self.evaluator = Evaluator(num_ent=args.ent_num, batch_size=2048)

        self.ent_dim = args.ent_dim
        self.ent_h = args.ent_height
        self.ent_w = self.ent_dim // self.ent_h

        self.input_drop = torch.nn.Dropout(args.input_drop).cuda()
        self.fea_drop = torch.nn.Dropout2d(args.fea_drop).cuda()
        self.hid_drop = torch.nn.Dropout(args.hid_drop).cuda()

        self.conv1 = torch.nn.Conv2d(1, args.filter_channel, (args.filter_size, args.filter_size), 1, 0, bias=True).cuda()
        self.bn0 = torch.nn.BatchNorm2d(1).cuda()
        self.bn1 = torch.nn.BatchNorm2d(args.filter_channel).cuda()
        self.bn2 = torch.nn.BatchNorm1d(args.ent_dim).cuda()
        self.register_parameter('b', torch.nn.Parameter(torch.zeros(args.ent_num)))
       
        self.flat_sz = (args.filter_channel*(2*args.ent_height-args.filter_size+1)*(self.ent_w-args.filter_size+1))
        self.fc = torch.nn.Linear(self.flat_sz, args.ent_dim).cuda()

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    
    def forward(self, edge_index, edge_type, h, r, t):

        head_emb = torch.index_select(self.ent_emb.weight, 0, h).view(-1, 1, self.ent_dim)
        rel_emb = torch.index_select(self.rel_emb.weight, 0, r).view(-1, 1, self.ent_dim)

        stack_input = torch.cat([head_emb, rel_emb], 1)
        stack_input	= torch.transpose(stack_input, 2, 1).reshape((-1, 1, 2*self.ent_h, self.ent_w))

        x = self.bn0(stack_input)
        x= self.input_drop(x)
        x= self.conv1(x)
        x= self.bn1(x)
        x= F.relu(x)
        x = self.fea_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hid_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.ent_emb.weight.transpose(1,0))
        x += self.b.expand_as(x)

        return x

    def predict_t(self, hr_batch, edge_index, edge_type, ent_emb=None, rel_emb=None):
        
        h = hr_batch[:, 0]
        r = hr_batch[:, 1]
        head_emb = torch.index_select(self.ent_emb.weight, 0, h).view(-1, 1, self.ent_dim)
        rel_emb = torch.index_select(self.rel_emb.weight, 0, r).view(-1, 1, self.ent_dim)

        stack_input = torch.cat([head_emb, rel_emb], 1)
        stack_input	= torch.transpose(stack_input, 2, 1).reshape((-1, 1, 2*self.ent_h, self.ent_w))

        x = self.bn0(stack_input)
        x= self.input_drop(x)
        x= self.conv1(x)
        x= self.bn1(x)
        x= F.relu(x)
        x = self.fea_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hid_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.ent_emb.weight.transpose(1,0))
        x += self.b.expand_as(x)

        return x

