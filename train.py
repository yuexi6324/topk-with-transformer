from agent_topk.model.train.new_model import VQAShotSelector
from agent_topk.dataset.dataset import VQADataset
from torch.utils.data import Dataset, DataLoader
from agent_topk.model.config import Config
# def train(cfg):
#     cfg = Config()
#     device = cfg.device
#     dataset = VQADataset(cfg.question_path, cfg.shots_path,cfg.ori_path,cfg.label_path)
#     dataloader = DataLoader(dataset,batch_size = cfg.batch_size, num_workers= cfg.num_workers)
#     for epoch in range(cfg.num_epochs):
#         for i , () in enumerate(dataloader):
