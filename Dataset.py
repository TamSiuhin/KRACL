from itertools import chain
import torch
from torch.utils.data import Dataset
import torch
import torch.utils.data as Data
import json
from utils import add_inverse_triples, add_noise

# class WN18RR_Inverse(Dataset):
#     def __init__(self, name, batch_size, path="./data/WN18RR/"):
#         self.name = name
#         self.batch_size = batch_size
#         dataset = WN18RR(create_inverse_triples=True)

#         if name in ["train" ,"valid"]:
#             data = dataset.training.mapped_triples
#             h, r, t = data.t()
#             r = r*2
#             data = torch.cat([torch.stack([h, r, t], dim=-1), torch.stack([t, r+1, h], dim=1)], dim=0)
#             self.src, self.edge_type, self.dst = data.t()
#             self.edge_index = torch.stack((self.src, self.dst), dim=0)
        
#         if name == "valid":
#             self.train_data = add_inverse_triples(dataset.training.mapped_triples)
#             self.valid_data = add_inverse_triples(dataset.validation.mapped_triples)
        
#         if name in ["train", "valid"]:
#             transe_pre = torch.load(path)
#             self.x = transe_pre["entity_embeddings._embeddings.weight"]
#             self.rel_emb = transe_pre["relation_embeddings._embeddings.weight"]
#             self.rel_emb = torch.cat((self.rel_emb, -self.rel_emb), dim=1).view(-1, 200)
        
#         if name in ["test"]:
#             self.train_data = add_inverse_triples(dataset.training.mapped_triples)
#             self.valid_data = add_inverse_triples(dataset.validation.mapped_triples)
#             self.test_data = add_inverse_triples(dataset.testing.mapped_triples)
            
#     def __len__(self):
#         if self.name == "train":
#             return int(self.edge_index.size(1) / self.batch_size)
#         else:
#             return 1
    
#     def __getitem__(self, index):
#         if self.name in ["train", "train_get_emb"]:
#             return self.x, self.edge_index, self.rel_emb, self.edge_type
#         elif self.name == "valid":
#             return self.train_data, self.valid_data, self.x, self.edge_index, self.edge_type
#         elif self.name == "test":
#             return self.train_data, self.valid_data, self.test_data

# class FB15k237_Inverse(Dataset):
#     def __init__(self, name, batch_size, dim=100, path="./data/WN18RR/"):
#         self.name = name
#         self.batch_size = batch_size
#         dataset = FB15k237(create_inverse_triples=False)
#         self.dim = dim

#         if name in ["train" ,"train_get_emb"]:
#             data = dataset.training.mapped_triples
#             h, r, t = data.t()
#             r = r*2
#             data = torch.cat([torch.stack([h, r, t], dim=-1), torch.stack([t, r+1, h], dim=1)], dim=0)
#             self.src, self.edge_type, self.dst = data.t()
#             self.edge_index = torch.stack((self.src, self.dst), dim=0)
        
#         if name == "valid":
#             self.train_data = add_inverse_triples(dataset.training.mapped_triples)
#             self.valid_data = add_inverse_triples(dataset.validation.mapped_triples)
#             self.test_data = add_inverse_triples(dataset.testing.mapped_triples)
        
#         if name in ["train", "train_get_emb"]:
#             if self.dim == 100:
#                 transe_pre = torch.load(path + "TransE_FB15k237_100dim.ckpt")
#             if self.dim == 200:
#                 transe_pre = torch.load(path + "TransE_FB15k237_200dim.ckpt")

#             self.x = transe_pre["entity_embeddings._embeddings.weight"]
#             self.rel_emb = transe_pre["relation_embeddings._embeddings.weight"]
#             self.rel_emb = torch.cat((self.rel_emb, -self.rel_emb), dim=1).view(-1, 200)
            
#     def __len__(self):
#         if self.name == "train":
#             return int(self.edge_index.size(1) / self.batch_size)
#         else:
#             return 1
    
#     def __getitem__(self, index):
#         if self.name in ["train", "train_get_emb"]:
#             return self.x, self.edge_index, self.rel_emb, self.edge_type
#         elif self.name == "valid":
#             return self.train_data, self.valid_data, self.test_data

# class WN18RR_Triples(Dataset):
#     def __init__(self, name, batch_size):
#         self.name = name
#         self.batch_size = batch_size
#         dataset = WN18RR()

#         if name in ["train"]:
#             self.train_data = add_inverse_triples(dataset.training.mapped_triples)
#             src, self.edge_type, dst = self.train_data.t()
#             self.edge_index = torch.stack((src, dst), dim=0)

#         if name in ["valid"]:
#             self.train_data = add_inverse_triples(dataset.training.mapped_triples)
#             self.valid_data = add_inverse_triples(dataset.validation.mapped_triples)
#             self.src, self.edge_type, self.dst = self.train_data.t()
#             self.edge_index = torch.stack((self.src, self.dst), dim=0)

        
#         if name in ["test"]:
#             self.train_data = add_inverse_triples(dataset.training.mapped_triples)
#             self.valid_data = add_inverse_triples(dataset.validation.mapped_triples)
#             self.test_data = add_inverse_triples(dataset.testing.mapped_triples)
            
#     def __len__(self):
#         if self.name in ["train"]:
#             return int(self.edge_index.size(1) / self.batch_size)
#         else:
#             return 1
    
#     def __getitem__(self, index):
#         if self.name in ["train"]:
#             return self.edge_index, self.edge_type, self.train_data
#         elif self.name == "valid":
#             return self.train_data, self.valid_data, self.edge_index, self.edge_type
#         elif self.name == "test":
#             return self.train_data, self.valid_data, self.test_data

# # class KG_Triples(Dataset):
# #     def __init__(self, name, batch_size, data_name, per=1.0):
# #         self.name = name
# #         self.batch_size = batch_size
        
# #         if data_name.lower() in ["wn18rr"]:
# #             dataset = WN18RR()
# #         elif data_name.lower() in ["fb15k237"]:
# #             dataset = FB15k237()
# #         elif data_name.lower() in ["db100k"]:
# #             dataset = DB100K()
# #         elif data_name.lower() in ["wn18"]:
# #             dataset = WN18()
# #         elif data_name.lower() in ["fb15k"]:
# #             dataset = FB15k()
# #         elif data_name.lower() in ["yago310"]:
# #             dataset = YAGO310()
# #         elif data_name.lower() in ["kinship"]:
# #             dataset = Kinships()
# #         elif data_name.lower() in ["umls"]:
# #             dataset = UMLS()
# #         else:
# #             raise ValueError("there is no such dataset!")
        
# #         if (per != 1.0) and (name=="train"):
# #             mask = torch.rand(dataset.training.mapped_triples.size(0))< per
# #             dataset.training.mapped_triples = dataset.training.mapped_triples[mask]
            
# #         if name in ["train"]:
# #             self.train_data = add_inverse_triples(dataset.training.mapped_triples)
# #             src, self.edge_type, dst = self.train_data.t()
# #             self.edge_index = torch.stack((src, dst), dim=0)

# #         if name in ["valid"]:
# #             self.train_data = add_inverse_triples(dataset.training.mapped_triples)
# #             self.valid_data = add_inverse_triples(dataset.validation.mapped_triples)
# #             self.src, self.edge_type, self.dst = self.train_data.t()
# #             self.edge_index = torch.stack((self.src, self.dst), dim=0)

        
# #         if name in ["test"]:
# #             self.train_data = add_inverse_triples(dataset.training.mapped_triples)
# #             self.valid_data = add_inverse_triples(dataset.validation.mapped_triples)
# #             self.test_data = add_inverse_triples(dataset.testing.mapped_triples)
            
# #     def __len__(self):
# #         if self.name in ["train"]:
# #             return int(self.edge_index.size(1) / self.batch_size)
# #         else:
# #             return 1
    
# #     def __getitem__(self, index):
# #         if self.name in ["train"]:
# #             return self.edge_index, self.edge_type, self.train_data
# #         elif self.name == "valid":
# #             return self.train_data, self.valid_data, self.edge_index, self.edge_type
# #         elif self.name == "test":
# #             return self.train_data, self.valid_data, self.test_data

def txt2triples(path):
    with open(path, 'r') as f:
        data = f.read().split()

        src = data[1::3]
        dst = data[2::3]
        edge_type = data[3::3]

        src = torch.tensor([int(i) for i in src])
        dst = torch.tensor([int(i) for i in dst])
        rel = torch.tensor([int(i) for i in edge_type])
        
        data = add_inverse_triples(torch.stack((src, rel, dst), dim=1))
        src, rel, dst = data.t()
        return data, src, rel, dst

class NELL_txt(Dataset):
    def __init__(self, name, batch_size, path):
        self.name = name
        self.batch_size = batch_size

        if self.name == "train":
            self.train_data, src, rel, dst = txt2triples(path + "train2id.txt")
            self.edge_type = rel
            self.edge_index = torch.stack((src, dst), dim=0)

        if self.name == "valid":
            self.train_data, src, rel, dst = txt2triples(path + "train2id.txt")
            self.edge_type = rel
            self.edge_index = torch.stack((src, dst), dim=0)
            self.valid_data, _, _, _ = txt2triples(path + "valid2id.txt")

        if self.name == "test":
            self.train_data, src, rel, dst = txt2triples(path + "train2id.txt")
            self.edge_type = rel
            self.edge_index = torch.stack((src, dst), dim=0)
            self.valid_data, _, _, _ = txt2triples(path + "valid2id.txt")
            self.test_data, _, _, _ = txt2triples(path + "test2id.txt")
    
    def __len__(self):
        if self.name == "train":
            return int(self.train_data.size(0) / self.batch_size)
        else:
            return 1

    def __getitem__(self, index):
        if self.name == "train":
            return self.edge_index, self.edge_type, self.train_data
        elif self.name == "valid":
            return self.train_data, self.valid_data, self.edge_index, self.edge_type
        elif self.name == "test":
            return self.train_data, self.valid_data, self.test_data

class KG_Triples_txt(Dataset):
    def __init__(self, name, batch_size, path="/data2/whr/tzx/OpenKE/benchmarks/WN18RR/"):
        self.name = name
        self.batch_size = batch_size

        if self.name == "train":
            self.train_data, src, rel, dst = txt2triples(path + "train2id.txt")
            self.edge_type = rel
            self.edge_index = torch.stack((src, dst), dim=0)

        if self.name == "valid":
            self.train_data, src, rel, dst = txt2triples(path + "train2id.txt")
            self.edge_type = rel
            self.edge_index = torch.stack((src, dst), dim=0)
            self.valid_data, _, _, _ = txt2triples(path + "valid2id.txt")

        if self.name == "test":
            self.train_data, src, rel, dst = txt2triples(path + "train2id.txt")
            self.edge_type = rel
            self.edge_index = torch.stack((src, dst), dim=0)
            self.valid_data, _, _, _ = txt2triples(path + "valid2id.txt")
            self.test_data, _, _, _ = txt2triples(path + "test2id.txt")
    
    def __len__(self):
        if self.name == "train":
            return int(self.train_data.size(0) / self.batch_size)
        else:
            return 1

    def __getitem__(self, index):
        if self.name == "train":
            return self.edge_index, self.edge_type, self.train_data
        elif self.name == "valid":
            return self.train_data, self.valid_data, self.edge_index, self.edge_type
        elif self.name == "test":
            return self.train_data, self.valid_data, self.test_data

class Triple_Category(Dataset):
    def __init__(self, name, batch_size, path):
        self.name = name
        self.batch_size = batch_size

        if self.name == "train":
            self.train_data, src, rel, dst = txt2triples(path + "train2id.txt")
            self.edge_type = rel
            self.edge_index = torch.stack((src, dst), dim=0)

        if self.name == "valid":
            self.train_data, src, rel, dst = txt2triples(path + "train2id.txt")
            self.edge_type = rel
            self.edge_index = torch.stack((src, dst), dim=0)
            self.valid_data, _, _, _ = txt2triples(path + "valid2id.txt")

        if self.name == "test":
            self.train_data, src, rel, dst = txt2triples(path + "train2id.txt")
            self.edge_type = rel
            self.edge_index = torch.stack((src, dst), dim=0)
            self.valid_data, _, _, _ = txt2triples(path + "valid2id.txt")
            self.relation1to1, _, _, _ = txt2triples(path + "1-1.txt")
            self.relation1ton, _, _, _ = txt2triples(path + "1-n.txt")
            self.relationnto1, _, _, _ = txt2triples(path + "n-1.txt")
            self.relationnton, _, _, _ = txt2triples(path + "n-n.txt")

    def __len__(self):
        if self.name == "train":
            return int(self.train_data.size(0) / self.batch_size)
        else:
            return 1

    def __getitem__(self, index):
        if self.name == "train":
            return self.edge_index, self.edge_type, self.train_data
        elif self.name == "valid":
            return self.train_data, self.valid_data, self.edge_index, self.edge_type
        elif self.name == "test":
            return self.train_data, self.valid_data, self.relation1to1,self.relation1ton, self.relationnto1, self.relationnton


class KG_Triples(Dataset):
    def __init__(self, name, num_relations, num_ent, train_path="./data/FB15K237/", test_path="./data/FB15K237/", noise_path=None, num_negs_per_pos=5):
        self.name = name
        self.num_relations = num_relations
        self.num_entities = num_ent
        self.num_negs_per_pos = num_negs_per_pos
        self.noise_path = noise_path
        self.train_data, _, _, _ = txt2triples(train_path + "train2id.txt")

        if self.name == "train":
            self.train_data, _, _, _ = txt2triples(train_path + "train2id.txt")
            if self.noise_path is not None:
                noise_data, _, _, _ = txt2triples(self.noise_path + "train2id.txt")
                self.train_data, _, _, _ = add_noise(self.train_data, noise_data)

        if self.name == "valid":
            self.train_data, _, _, _ = txt2triples(test_path + "train2id.txt")
            # self.edge_type = rel
            # self.edge_index = torch.stack((src, dst), dim=0)
            self.valid_data, _, _, _ = txt2triples(test_path + "valid2id.txt")
            self.test_data, _, _, _ = txt2triples(test_path + "test2id.txt")

        if self.name == "test":
            self.train_data, _, _, _ = txt2triples(test_path + "train2id.txt")
            # self.edge_type = rel
            # self.edge_index = torch.stack((src, dst), dim=0)
            self.valid_data, _, _, _ = txt2triples(test_path + "valid2id.txt")
            self.test_data, _, _, _ = txt2triples(test_path + "test2id.txt")
    
    def corrupt_batch(self, positive_batch: torch.LongTensor) -> torch.LongTensor:  # noqa: D102
        
        batch_shape = positive_batch.shape[:-1]
        # Copy positive batch for corruption.
        # Do not detach, as no gradients should flow into the indices.
        negative_batch = positive_batch.clone()
        negative_batch = negative_batch.unsqueeze(dim=-2).repeat(*(1 for _ in batch_shape), self.num_negs_per_pos, 1)

        corruption_index = torch.randint(1, size=(*batch_shape, self.num_negs_per_pos))
        
        index_max = self.num_relations 
        mask = corruption_index == 1
        # To make sure we don't replace the {head, relation, tail} by the
        # original value we shift all values greater or equal than the original value by one up
        # for that reason we choose the random value from [0, num_{heads, relations, tails} -1]
        negative_indices = torch.randint(
            high=index_max,
            size=(mask.sum().item(),),
            device=positive_batch.device,
        )

        # determine shift *before* writing the negative indices
        shift = (negative_indices >= negative_batch[mask][:, 1]).long()
        negative_indices += shift

        # write the negative indices
        negative_batch[
            mask.unsqueeze(dim=-1) & (torch.arange(3) == 1).view(*(1 for _ in batch_shape), 1, 3)
        ] = negative_indices

        return negative_batch

    def __len__(self):
        if self.name == "train":
            return self.train_data.size(0)
        else:
            return 1

    def __getitem__(self, index):
        if self.name == "train":
            pos_batch = self.train_data[index]
            neg_batch = self.corrupt_batch(pos_batch)
            return pos_batch, neg_batch
        
        elif self.name == "valid":
            return self.train_data, self.valid_data, self.test_data
        
        elif self.name == "test":
            return self.train_data, self.valid_data, self.test_data