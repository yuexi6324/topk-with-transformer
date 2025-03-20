
import torch
import torch.nn as nn
import numpy as np
import math
from train import Config


class AttentionMask():
    """
    Implementation of Attention Mask 
    padding_mask for encoder
    padding_mask + look_ahead_mask for decoder
    """
    def __init__(self, cfg):
        self.NEG_INFTY = cfg.NEG_INFTY
        self.max_sequence_length = cfg.max_sequence_length
        self.vocab_size = cfg.tgt_vocab_size

    def tgt_look_mask(self,target_seq):
        
        tgt_look_mask = torch.full([self.max_sequence_length, self.max_sequence_length],False)
      

        for i in range(len(target_seq)):
            mask_region = target_seq[0:i + 1]
            tgt_look_mask[i, mask_region] = True
        return tgt_look_mask
    
    def softmax_mask(self, target_seq):
        softmax_mask = torch.full([self.max_sequence_length, self.vocab_size],False)
        for i in range(len(target_seq)):
            mask_region = target_seq[0:i + 1]
            softmax_mask[i, mask_region] = True
        print(f'softmax_mask:{softmax_mask.shape}')
        # print(softmax_mask)
        return softmax_mask

    def create_masks(self, src_batch, tgt_batch):
        batch_size = src_batch.size(0)

        encoder_padding_mask = torch.full([batch_size, self.max_sequence_length, self.max_sequence_length], False)
        #decoder_padding_mask_self_attention = torch.full([batch_size, self.max_sequence_length, self.max_sequence_length], False)
        decoder_padding_mask_cross_attention = torch.full([batch_size, self.max_sequence_length, self.max_sequence_length], False)
        decoder_softmax_mask = torch.full([batch_size,  self.max_sequence_length, self.vocab_size], False)

        for idx in range(batch_size):
            source_sentence_length, target_sentence_length = len(src_batch[idx]), len(tgt_batch[idx])
            look_ahead_mask = torch.full([self.max_sequence_length, self.max_sequence_length], True)
            look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
            target_seq = tgt_batch[idx]
           

            source_chars_to_padding_mask = np.arange(source_sentence_length + 1, self.max_sequence_length)
            target_chars_to_padding_mask = np.arange(target_sentence_length + 1, self.max_sequence_length)
            encoder_padding_mask[idx, :, source_chars_to_padding_mask] = True
            encoder_padding_mask[idx, source_chars_to_padding_mask, :] = True
            decoder_padding_mask_cross_attention[idx, :, source_chars_to_padding_mask] = True
            decoder_padding_mask_cross_attention[idx, target_chars_to_padding_mask, :] = True
            decoder_softmax_mask[idx, target_chars_to_padding_mask,:] = True
            

            #1.在cross-attention中添加look_ahead_mask
            decoder_look_ahead_mask = self.tgt_look_mask(target_seq)
            decoder_padding_mask_cross_attention = decoder_padding_mask_cross_attention + decoder_look_ahead_mask

            #softmax_mask
            softmax_mask_region = self.softmax_mask(target_seq)
            decoder_softmax_mask[idx] = softmax_mask_region + decoder_softmax_mask[idx]

        encoder_self_attention_mask = torch.where(encoder_padding_mask, self.NEG_INFTY, 0)
        decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, self.NEG_INFTY, 0)
        decoder_softmax_mask = torch.where(decoder_softmax_mask, self.NEG_INFTY , 0)

        return encoder_self_attention_mask, decoder_cross_attention_mask,decoder_softmax_mask





if __name__ == "__main__":

    # 模拟编码器输入数据
    batch_size = 2
    src_len = 3
    d_model = 2
    src = torch.randn(batch_size, src_len)
    # print(f'src_shape:{src.shape}')
    # 模拟解码器输入数据
    tgt_len = 3
    tgt = torch.tensor([
        [1,2],
        [0,2]
    ])
    cfg =  Config()
    mask = AttentionMask(cfg)
    self_attention_mask, decoder_attention_mask,decoder_softmax_mask = mask.create_masks(src,tgt)
    print(src)
    print(tgt)

    print(decoder_attention_mask[0])
    print(decoder_softmax_mask[1])
    print(decoder_attention_mask.shape)
  