import torch.nn as nn
import torch
import torch.nn.functional as F
class PointerDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model
        self.nhead = cfg.nhead
        self.max_seq_len = cfg.max_seq_len  # 包括终止元素
        self.vocab_size = cfg.tgt_vocab_size
        
        self.query = nn.Embedding
        self.attention = nn.MultiheadAttention(self.d_model, self.nhead)
        self.fc = nn.Linear(self.d_model, self.vocab_size)
        
        
        
    def forward(self, memory, tgt, mask, tgt_mask=None):
        """
        Args:
            memory: encoder embedding   shape:[batch_size,seq_len,d_model]
            tgt:target seq embedding    shape:[batch_size,seq_len,d_model]
            mask:decoder cross-attention mask   shape:[batch_size, seq_len, seq_len]
            
        Returns:
            output:softmax result
        """
        attn_output, attn_weights = self.attention(memory, tgt, tgt,key_padding_mask = mask)
        
        output_vocab_size = self.fc(attn_output)  #在这里添加mask

        return F.softmax(output_vocab_size)


