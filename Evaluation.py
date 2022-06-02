import torch
import numpy as np
import logging

class Evaluator():
    def __init__(self, num_ent, batch_size=1024, hits_k=[1,3,5,10]):
        self.ranks = []
        self.hits = {}
        self.hits_left = {}
        self.hits_right = {}
        self.ranks_left = []
        self.ranks_right = []
        self.hits_k = hits_k
        self.num_ent = num_ent
        self.batch_size = batch_size

    def evaluate(self, test_triples, all_pos_triples, model, ent_emb, rel_emb, edge_index, edge_type, save_emb=False):

        self.ranks = []
        self.ranks_left = []
        self.ranks_right = []
        self.hits = {}
        self.hits_left = {}
        self.hits_right = {}
        
        triples_num = test_triples.size(0)//2
        right_triples = test_triples[:triples_num]
        left_triples = test_triples[triples_num:]

        right_loader = torch.split(right_triples, self.batch_size, dim=0)
        left_loader = torch.split(left_triples, self.batch_size, dim=0)
        
        for batch in right_loader:
            self.evaluate_batch(batch, model, all_pos_triples, ent_emb, rel_emb, edge_index, edge_type, mode="right", save_emb=save_emb)
        
        for batch in left_loader:
            self.evaluate_batch(batch, model, all_pos_triples, ent_emb, rel_emb, edge_index, edge_type, mode="left")

        all_mr, left_mr, right_mr, all_mrr, left_mrr, right_mrr = self.compute_metrics()
        return all_mr, all_mrr, self.hits, left_mr, left_mrr, self.hits_left, right_mr, right_mrr, self.hits_right

    def compute_metrics(self):
        all_ranks = torch.cat(self.ranks).float()
        all_mr = all_ranks.mean()
        all_mrr = torch.reciprocal(all_ranks).mean()
        
        left_ranks = torch.cat(self.ranks_left).float()
        left_mr = left_ranks.mean()
        left_mrr = torch.reciprocal(left_ranks).mean()
        
        right_ranks = torch.cat(self.ranks_right).float()
        right_mr = right_ranks.mean()
        right_mrr = torch.reciprocal(right_ranks).mean()

        self.get_hits(self.hits_k)
        return all_mr, left_mr, right_mr, all_mrr, left_mrr, right_mrr

    def get_hits(self, k_list):

        all_ranks = torch.cat(self.ranks)
        left_ranks = torch.cat(self.ranks_left)
        right_ranks = torch.cat(self.ranks_right)
        for k in k_list:
            all_hits = (all_ranks<=k).float().mean()
            left_hits = (left_ranks<=k).float().mean()
            right_hits = (right_ranks<=k).float().mean()
            
            self.hits[str(k)] = all_hits
            self.hits_left[str(k)] = left_hits
            self.hits_right[str(k)] = right_hits
    
    
    def evaluate_batch(self, batch, model, all_pos_triples, ent_emb, rel_emb, edge_index, edge_type, mode="right", save_emb=False):
        batch_size = batch.size(0)
        filter_batch, relation_filter = self.create_sparse_positive_filter_(batch, all_pos_triples, relation_filter=None, filter_col=2)
        
        hr_batch = batch[:,0:2]
        t_batch = batch[:,2]
        scores = model.predict_t(hr_batch=hr_batch, ent_emb=ent_emb, rel_emb=rel_emb, edge_index=edge_index, edge_type=edge_type, save_emb=save_emb)
        true_scores = scores[torch.arange(batch_size), t_batch].view(-1, 1)
        # print(scores)
        # print(scores.size())
        filtered_scores = self.filter_scores(scores, filter_batch)
        
        ranks = self.get_ranks(filtered_scores, true_scores)
        self.ranks.append(ranks)
        
        if mode=="right":
            self.ranks_right.append(ranks)
        else:
            self.ranks_left.append(ranks)
    
    def get_ranks(self, all_scores, true_score):
        optimistic_rank = (all_scores > true_score).sum(dim=1) + 1
        return optimistic_rank

    def create_sparse_positive_filter_(self, hrt_batch, all_pos_triples, relation_filter=None, filter_col=2):

        if relation_filter is None:
            relations = hrt_batch[:, 1:2]
            relation_filter = (all_pos_triples[:, 1:2]).view(1, -1) == relations

        # Split batch
        other_col = 2 - filter_col
        entities = hrt_batch[:, other_col : other_col + 1]

        entity_filter_test = (all_pos_triples[:, other_col : other_col + 1]).view(1, -1) == entities
        filter_batch = (entity_filter_test & relation_filter).nonzero(as_tuple=False)
        filter_batch[:, 1] = all_pos_triples[:, filter_col : filter_col + 1].view(1, -1)[:, filter_batch[:, 1]]

        return filter_batch, relation_filter

    def filter_scores(self, scores, filter_batch):
        # Bind shape
        batch_size, num_entities = scores.shape

        # Set all filtered triples to NaN to ensure their exclusion in subsequent calculations
        scores[filter_batch[:, 0], filter_batch[:, 1]] = float("nan")

        # Warn if all entities will be filtered
        # (scores != scores) yields true for all NaN instances (IEEE 754), thus allowing to count the filtered triples.
        if ((scores != scores).sum(dim=1) == num_entities).any():
            logging.warning(
                "User selected filtered metric computation, but all corrupted triples exists also as positive " "triples",
            )

        return scores