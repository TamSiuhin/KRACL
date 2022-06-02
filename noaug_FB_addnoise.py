import torch
from torch.nn import functional as F
import numpy as np
from torch.utils.data import DataLoader
from model import CLKG_compgatv3_convE, SupConLoss, relation_contrast
import pytorch_lightning as pl
from Dataset import KG_Triples_txt, txt2triples, KG_Triples
from utils import *
from torch_geometric.nn import Sequential
import argparse
from Evaluation import Evaluator
from pytorch_lightning.callbacks import ModelCheckpoint
from os import listdir
from pytorch_lightning.utilities.seed import seed_everything
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging

class CL(pl.LightningModule):
    def __init__(self, args, model):
        super(CL, self).__init__()
        self.learning_rate = args.cl_lr
        self.evaluator = Evaluator(num_ent=args.ent_num, batch_size=2048)
        # self.augmenter1 = Random_Choice(auglist1, args.num_choice)
        # self.augmenter2 = Random_Choice(auglist2, args.num_choice)
        self.model = model
        self.supconloss = SupConLoss(temperature=args.temp1, contrast_mode="all", base_temperature=args.temp1).to(torch.device("cuda"))
        self.rank_loss = torch.nn.CrossEntropyLoss()
        self.rel_cl = relation_contrast(args.temp2, args.neg_sample)
        self.lam1 = args.lam1
        self.lam2 = args.lam2
        self.wd = args.weight_decay
        valid_triple, src, rel, dst = txt2triples(args.train_path + "train2id.txt")
        if args.noise_path is not None:
            noise_triple, _, _, _ = txt2triples(args.noise_path + "train2id.txt")
            _, src, rel, dst = add_noise(valid_triple, noise_triple)
        
        self.edge_type = rel.cuda()
        self.edge_index = torch.stack((src, dst), dim=0).cuda()
        self.label_smoothing = args.label_smoothing
        self.num_entities = args.ent_num
        logging.basicConfig(filename= "./log/{}.log".format(args.info), filemode="w", level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.log2 = logging.getLogger(__name__)
        self.log2.info(args)
        
    # def before_run(self):
        
    def training_step(self, train_batch, batch_idx):   
        pos = train_batch[0]
        neg = train_batch[1].view(-1, 3)
        head, rel, tail = pos.t()
        self.model.train()
        # contrastive learning
        # Augmentation 1
        # aug1_node_emb, aug1_edge_index, aug1_edge_type, aug1_rel_emb = self.augmenter1.augment(x=self.model.ent_emb.weight, edge_index=self.edge_index, edge_type=self.edge_type, rel_emb=self.model.rel_emb.weight, edge_batch_idx=None)
        x1_node, score1, tail_emb1, _, ent_emb, rel_emb= self.model(self.edge_index, self.edge_type, head, rel, tail, self.model.ent_emb.weight, self.model.rel_emb.weight)
        tail_emb1 = F.normalize(tail_emb1, dim=1)
        x1_node = F.normalize(x1_node, dim=1)
        
        # calculate SupCon loss
        features1 = torch.cat((x1_node.unsqueeze(1), tail_emb1.unsqueeze(1)), dim=1)
        # features2 = torch.cat((x2_node.unsqueeze(1), tail_emb1.unsqueeze(1)), dim=1)
        
        # SupCon Loss
        supconloss1 = self.supconloss(features1, labels=tail, mask=None)
        # supconloss2 = self.supconloss(features2, labels=tail, mask=None)

        # CELoss
        celoss1 = self.rank_loss(score1, tail)
        # celoss2 = self.rank_loss(score2, tail)
                
        self.log("model_train_loss", supconloss1)
        loss = (supconloss1) + (celoss1) 
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(recurse=True), lr=self.learning_rate, weight_decay=self.wd)
        # return optimizer
        scheduler = CosineAnnealingLR(optimizer, T_max=16, eta_min=0)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler
            },
        }

    def validation_step(self, valid_batch, valid_idx):
        train_triples = valid_batch[0].squeeze(0)
        valid_triples = valid_batch[1].squeeze(0)
        test_triples = valid_batch[2].squeeze(0)
        
        self.model.eval()
        filter_triples = torch.cat((train_triples, valid_triples, test_triples), dim=0)
        
        all_mr, all_mrr, hits, left_mr, left_mrr, hits_left, right_mr, right_mrr, \
        hits_right = self.evaluator.evaluate(valid_triples, filter_triples, self.model, self.model.ent_emb.weight, self.model.rel_emb.weight, self.edge_index, self.edge_type)
        
        self.log("MR_all", all_mr)
        self.log("MR_right", right_mr)
        self.log("MR_left", left_mr)
        self.log("MRR_all", all_mrr)
        self.log("MRR_right", right_mrr)
        self.log("MRR_left", left_mrr)

        self.log_dict(hits)
        self.log_dict(hits_left)
        self.log_dict(hits_right)
        
        self.log2.info('performance on valid set - Epoch {}, MRR {}, MR {}, H@10 {}, H@3 {}, H@1 {}'.format(self.current_epoch, all_mrr, all_mr, hits["10"], hits["3"], hits["1"]))
        
        # print("MR: {}".format(all_mr))
        # print("MRR: {}".format(all_mrr))

    def test_step(self, test_batch, batch_idx):
        train_triples = test_batch[0].squeeze(0)
        valid_triples = test_batch[1].squeeze(0)
        test_triples = test_batch[2].squeeze(0)

        self.model.eval()
        src, rel, dst = train_triples.t()
        # edge_index = torch.stack((src, dst), dim=0)
        # edge_type = rel
        
        filter_triples = torch.cat((train_triples, valid_triples, test_triples), dim=0)
        all_mr, all_mrr, hits, left_mr, left_mrr, hits_left, right_mr, right_mrr, \
        hits_right = self.evaluator.evaluate(test_triples, filter_triples, self.model, self.model.ent_emb.weight, self.model.rel_emb.weight, self.edge_index, self.edge_type, save_emb=args.save_emb)
        
        self.log("MR_all_test", all_mr)
        self.log("MR_right_test", right_mr)
        self.log("MR_left_test", left_mr)
        self.log("MRR_all_test", all_mrr)
        self.log("MRR_right_test", right_mrr)
        self.log("MRR_left_test", left_mrr)

        self.log_dict(hits)
        self.log_dict(hits_left)
        self.log_dict(hits_right )

        self.log2.info('performance on test set: MRR {}, MR {}, H@10 {}, H@3 {}, H@1 {}'.format(all_mrr, all_mr, hits["10"], hits["3"], hits["1"]))
        # print("MR: {}".format(all_mr))
        # print("MRR: {}".format(all_mrr))

    def predict_step(self, batch, batch_idx):        
        return self.model
    
    def train_dataloader(self):
        train_dataset = KG_Triples(name="train", batch_size=args.cl_batch_size, path=args.triple_path, p=args.p)
        return DataLoader(train_dataset, batch_size=args.cl_batch_size, shuffle=True)



parser = argparse.ArgumentParser(description="Implementation of CLKG-convE")

parser.add_argument("--train_path", type=str, default="./data/FB15K237/", help="knowledge graph dataset path")
parser.add_argument("--test_path", type=str, default="./data/FB15K237/", help="knowledge graph dataset path")
parser.add_argument("--noise_path", type=str, default=None, help="knowledge graph dataset path")

parser.add_argument("--rel_num", type=int, default=237*2, help="number of relations in Knowledge Graph")
parser.add_argument("--ent_num", type=int, default=14541, help="number of entites in Knowledge Graph")
parser.add_argument("--init_dim", type=int, default=200, help="dimension of entities embeddings")
parser.add_argument("--ent_dim", type=int, default=200, help="dimension of entities embeddings")
parser.add_argument("--rel_dim", type=int, default=200, help="dimension of relations embeddings")
parser.add_argument("--filter_size", type=int, default=7, help="size of relation specific kernels")
parser.add_argument("--cl_batch_size", type=int, default=2048, help="training batch size")
parser.add_argument("--cl_lr", type=float, default=1e-3, help="learning rate of contrastive learning")
parser.add_argument("--decode_batch_size", type=int, default=2048, help="learning rate of decode stage")
parser.add_argument("--decode_lr", type=float, default=5e-4, help="learning rate of decoding")
parser.add_argument("--decode_epochs", type=int, default=100, help="max epochs of decode training")
parser.add_argument("--cl_epochs", type=int, default=1000, help="epochs of contrastive learning")                    
parser.add_argument("--ent_height", type=int, default=10, help="enttities embedding height after reshaping")
parser.add_argument("--encoder_drop", type=float, default=0.1, help="dropout ratio for encoder")
parser.add_argument("--encoder_hid_drop", type=float, default=0.3, help="dropout ratio for encoder")
parser.add_argument("--proj_hid", type=int, default=200, help="hidden dimension of projection head")
parser.add_argument("--temp1", type=float, default=0.07, help="temperature of contrastive loss")
parser.add_argument("--temp2", type=float, default=0.07, help="temperature of contrastive loss")
parser.add_argument("--label_smoothing", type=float, default=0.0, help="label smoothing value")
parser.add_argument("--op", type=str, default="corr_new", help="aggregation opration")
parser.add_argument("--init_emb", type=str, default="random", help="initial embedding")
parser.add_argument("--gcn_layer", type=int, default=1, help="number of gcn layer")
parser.add_argument("--proj", type=str, default="linear", help="projection head type")
parser.add_argument("--input_drop", type=float, default=0.2, help="input dropout ratio")
parser.add_argument("--fea_drop", type=float, default=0.3, help="feature map dropout ratio")
parser.add_argument("--hid_drop", type=float, default=0.3, help="hidden feature dropout ratio")       
parser.add_argument("--filter_channel", type=int, default=256, help="number of filter channels")
parser.add_argument("--bias", type=bool, default=True, help="whether to use bias in convolution opeation")
parser.add_argument("--kg_md", type=str, default="conve", help="Knowledge Graph prediction model")
parser.add_argument("--dataset", type=str, default="wn18rr", help="choose the dataset to perform the model")
parser.add_argument("--proj_dim", type=int, default=128, help="projection dimension")
parser.add_argument("--valid_routine", type=int, default=1, help="valid_routine")
parser.add_argument("--random_seed", type=int, default=None, help="random seed")
parser.add_argument("--lam1", type=float, default=1, help="weight for two loss function")
parser.add_argument("--lam2", type=float, default=1, help="weight for two loss function")
parser.add_argument("--info", type=str, default="FB15K237", help="description for experiment")
parser.add_argument("--beta", type=float, default=0, help="description for experiment")
parser.add_argument("--save_emb", type=bool, default=False, help="description for experiment")
parser.add_argument("--neg_sample", type=int, default=2, help="description for experiment")
parser.add_argument("--p", type=float, default=1.0, help="training data percentage")
parser.add_argument("--num_worker", type=int, default=4, help="num workers")
parser.add_argument("--weight_decay", type=float, default=0, help="num workers")

# PRETRAIN_PATH = "/new_temp/fsb/TZX/CLKG/pretrained_emb/WN18RR/"
# PRETRAIN_PATH = "/data2/whr/tzx/pretrained_emb/

def main():
    global args
        
    args = parser.parse_args()
    torch.set_num_threads(1)

    if args.random_seed != None:
        seed_everything(args.random_seed)

    # train clkg
    train_dataset = KG_Triples(name="train", num_relations=args.rel_num, num_ent=args.ent_num, train_path=args.train_path, noise_path=args.noise_path)
    valid_dataset = KG_Triples(name="valid", num_relations=args.rel_num, num_ent=args.ent_num, test_path=args.test_path)
    test_dataset = KG_Triples(name="test", num_relations=args.rel_num, num_ent=args.ent_num, test_path=args.test_path)

    cl_train_dataloader = DataLoader(train_dataset, batch_size=args.cl_batch_size, shuffle=True, num_workers=args.num_worker)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1)
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    # decode_dataloader = DataLoader(train_dataset, batch_size=args.decode_batch_size, shuffle=True)
    checkpoint_callback = ModelCheckpoint(
        monitor='MRR_all',
        mode='max',
        filename='{MRR_all:.4f}',
        save_top_k=1,
        verbose=True)
    
    cl_trainer = pl.Trainer(gpus=1, num_nodes=1, max_epochs=args.cl_epochs, precision=32, callbacks=[checkpoint_callback])
    # decode_trainer = pl.Trainer(gpus=1, num_nodes=1, max_epochs=args.decode_epochs, precision=16, callbacks=[checkpoint_callback])
    dir = './lightning_logs/version_{}/checkpoints'.format(cl_trainer.logger.version)

    model =  CLKG_compgatv3_convE(args)
    print("Model Training begin!")
    cl_model = CL(args=args, model=model)
    # cl_trainer.tune(cl_model)

    cl_trainer.fit(cl_model, cl_train_dataloader, valid_dataloader)
    
    best_path = './lightning_logs/version_{}/checkpoints/{}'.format(cl_trainer.logger.version, listdir(dir)[0])
    best_model = cl_model.load_from_checkpoint(checkpoint_path=best_path, args=args, model=model)

    cl_trainer.test(best_model, test_dataloader, verbose=True, ckpt_path="best")

if __name__ == "__main__":
    
    main()
