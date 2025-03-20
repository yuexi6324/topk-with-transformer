import torch.nn as nn
import torch
from Position import PositionalEncoding
from PointerDecoder import PointerDecoder


class VQAShotSelector(nn.Module):
    def __init__(self, cfg):
        super(VQAShotSelector, self).__init__()
        
        # self.embedding_trans = nn.Linear(input_d, d_model) #
        self.d_model = cfg.d_model
        self.max_len = cfg.max_seq_length
        self.nhead = cfg.nhead
        self.dim_feedforward = cfg.feedforward
        self.dropout = cfg.dropout

        
        # 编码器
        self.encoder_layer = nn.TransformerEncoderLayer(self.d_model, self.nhead, self.dim_feedforward, self.dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, self.num_encoder_layers)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model=self.d_model,max_len = self.max_len)
        
        # 指针解码层
        self.decoder = PointerDecoder(cfg) 

        
        
    def forward(self, src, tgt):
        # src: [src_len, batch_size, d_model]，编码器输入
        # tgt: [tgt_len, batch_size,d_model]，解码器输入

        # src = self.embedding_trans(src)
        # 编码器部分
        src_pos_embedding = self.pos_encoder(src)

        #编码
        memory = self.encoder(src_pos_embedding)  # [src_len, batch_size, d_model]
        
        #解码阶段
        seq = self.decoder(memory,tgt) #[batch_size,src_len, vocab_size]
           
        return seq