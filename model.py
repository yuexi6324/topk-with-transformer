import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torch.nn import Transformer
from torch.utils.data import Dataset, DataLoader
import math
import random
from agent_topk.model.dataset.dataset import VQASelectorDataset
# 超参数设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d_model =1024#ransformer模型的维度
nhead = 8  # 多头注意力的头数
num_encoder_layers = 6  # 编码器层数
num_decoder_layers = 6  # 解码器层数

dim_feedforward = 2048  # 前馈网络的维度
dropout = 0.1
max_seq_length = 32  # 最大序列长度
batch_size = 32
learning_rate = 0.001
num_epochs = 10


question_path = '/ai/data/OFv2_ICL_VQA/agent_topk/data/ok_vqa_question_rs_clip.npy'
shots_path = '/ai/data/OFv2_ICL_VQA/agent_topk/data/ok_vqa_shots_rs_clip.npy'
ori_path = '/ai/data/OFv2_ICL_VQA/agent_topk/data/ok_vqa_rs_clip.json'
label_path = '/ai/data/OFv2_ICL_VQA/agent_topk/data/ok_vqa_results_rs_clip.json'
# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=1024, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
       
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class PointerDecoder(nn.Module):
    def __init__(self, d_model=1024, nhead=8, max_seq_len=32):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.max_seq_len = max_seq_len  # 包括终止元素
        
        self.query_embed = nn.Embedding(1, d_model)  # 初始查询向量
        self.attention = nn.MultiheadAttention(d_model, nhead)
        self.fc = nn.Linear(d_model, 33)
        self.pad_token = 0                
        
    def forward(self, memory, tgt, num_steps=None):
        """
        Args:
            memory: 来自编码器的输出，形状为 [seq_len, batch_size, d_model]
            num_steps: 生成序列的步数，如果为 None，则默认为 memory 的序列长度
            tgt:[max_len,batch_size,d_model]整理好的已经填充过的target目标序列
            
        Returns:
            generated_sequence: 生成的序列索引，形状为 [batch_size, num_steps]
        """

        if num_steps is None:
            num_steps = tgt.size(1)  # 默认生成序列长度为 max_seq_len
        
        batch_size = memory.size(1) #batch_size
        seq_len = memory.size(0) #seq_length
        
        # 初始化生成序列
        generated_sequence = torch.zeros(batch_size, num_steps, dtype=torch.long, device=memory.device) - 1 #生成序列全部设置为-1
        
        # 初始化掩码，排除第一个元素（索引为0的位置）
        select_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=memory.device)

        select_mask[:, 0] = True  # 排除第一个元素



        # 查询向量为原始VQA的memory
        query = memory[0,:,:].unsqueeze(1).transpose(0,1)  # 形状为 [1, batch_size, d_model]
        print(f'query_shape:{query.shape}')

        for step in range(num_steps):
            # 计算注意力
            attn_output, attn_weights = self.attention(query, memory, memory,key_padding_mask = select_mask)
         
            # 计算分数
            scores = attn_weights.squeeze(1)
            #scores = self.fc(attn_weights).squeeze(-1)  # 形状为 [1, batch_size]
            # print(f'scores.shape:{scores.shape}')
            # 应用掩码，已经选择的位置分数设为 -inf
            scores = scores.masked_fill(select_mask,torch.finfo(scores.dtype).min)
           # print(scores)
            # 转换为概率分布
            probs = torch.softmax(scores, dim=-1)
            #抽取index
            selected_index = torch.multinomial(probs, 1)  # 形状为 [batch_size, 1]
            # else:
            #     # 在推理模式下，根据概率分布采样
            #     selected_index = torch.argmax(probs, dim=-1, keepdim=True)  # 形状为 [batch_size, 1]
            
            # 检查是否选择了终止元素
            
            # 更新生成序列
            generated_sequence[:, step] = selected_index.squeeze(-1)
            
            # 更新掩码，将已选择的位置设为 True
            select_mask.scatter_(1, selected_index, True)
            selected_index = selected_index.squeeze()
            # print(selected_index.shape)
            # 更新查询向量，使用当前选择的位置的真实编码作为新的查询向量
            #query = memory[selected_index, torch.arange(batch_size), :].unsqueeze(0)
            # print(query)
            # print(f'query_shape:{query.shape}')
        
        return generated_sequence

    def inference(self,memory, num_step = None):
        if num_steps is None:
            num_steps = self.max_seq_len  # 默认生成序列长度为 max_seq_len
        batch_size = memory.size(1) #batch_size
        seq_len = memory.size(0) #seq_length
        
        # 初始化生成序列
        generated_sequence = torch.zeros(batch_size, num_steps, dtype=torch.long, device=memory.device) - 1 #生成序列全部设置为-1
        
        # 初始化掩码，排除第一个元素（索引为0的位置)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=memory.device)
        print(f'mask_shape:{mask.shape}')

         # 查询向量为原始VQA的memory
        query = memory[0,:,:].unsqueeze(1).transpose(0,1)  # 形状为 [1, batch_size, d_model]
        print(f'query_shape:{query.shape}')

        for step in range(num_steps):
            # 计算注意力
            attn_output, attn_weights = self.attention(query, memory, memory, key_padding_mask=mask)
         
            # 计算分数
            scores = attn_weights.squeeze(1)
            #scores = self.fc(attn_weights).squeeze(-1)  # 形状为 [1, batch_size]
            # print(f'scores.shape:{scores.shape}')
            # 应用掩码，已经选择的位置分数设为 -inf
            scores = scores.masked_fill(mask,torch.finfo(scores.dtype).min)
           # print(scores)
            # 转换为概率分布
            probs = torch.softmax(scores, dim=-1)
            #抽取index
            selected_index = torch.multinomial(probs, 1)  # 形状为 [batch_size, 1]
            # else:
            #     # 在推理模式下，根据概率分布采样
            # selected_index = torch.argmax(probs, dim=-1, keepdim=True)  # 形状为 [batch_size, 1]
            
            # 检查是否选择了终止元素
            if selected_index == 0:
                break
            # 更新生成序列
            generated_sequence[:, step] = selected_index.squeeze(-1)
            
            # 更新掩码，将已选择的位置设为 True
            mask.scatter_(1, selected_index, True)
            selected_index = selected_index.squeeze()
            # print(selected_index.shape)
            # 更新查询向量，使用当前选择的位置的真实编码作为新的查询向量
            query = memory[selected_index, torch.arange(batch_size), :].unsqueeze(0)
            # print(query)
            # print(f'query_shape:{query.shape}')




# 模型定义
class VQAShotSelector(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers,  dim_feedforward, dropout, max_len, input_d = 1024):
        super(VQAShotSelector, self).__init__()
        #将1024维度转化成512维度
        # self.embedding_trans = nn.Linear(input_d, d_model) #

        self.d_model = d_model
        print(self.d_model)
        self.max_len = max_len
        
        # 编码器
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model=self.d_model)
        
        # 指针解码层
        self.decoder = PointerDecoder() 

        
        
    def forward(self, src, tgt):
        # src: [src_len, batch_size, d_model]，编码器输入
        # tgt: [tgt_len, batch_size]，解码器输入
        #降维
        print(src.shape)
        # src = self.embedding_trans(src)
        print(f'降维后的src:{src.shape}')

        # 编码器部分
        src_pos_embedding = self.pos_encoder(src)
        print(f'位置编码后的src:{src.shape}')
        #编码
        memory = self.encoder(src_pos_embedding)  # [src_len, batch_size, d_model]
        print(f'encoder编码后的memory:{memory.shape}')
        
        #解码阶段
        seq = self.decoder(memory,tgt)
           
        return seq
    

# model.eval()
# seq = model(src,tgt)
# print(seq)
# print(f'推理阶段的model scores：{seq.shape}')

# def eval(model, src, end_symbol = 33,max_len = 33):
#     model.eval()

#     end_symbol = 33
#     max_len = 33

#     # 初始化目标序列，从起始标记开始
#     tgt = torch.ones(1, src.size(1)).long().to(src.device)  # (1, batch_size)
#     print(tgt)

#     # 初始化一个布尔掩码，用于记录哪些句子已经终止
#     finished = torch.zeros(src.size(1), dtype=torch.bool).to(src.device)

#     # 推理循环
#     for _ in range(max_len - 1):
#         # 前向传播
#         with torch.no_grad():
#             output = model(src, tgt)
#             print(output)
#         # 获取最后一个时间步的输出
#         next_token_logits = output  # (batch_size, vocab_size)

#         # 选择概率最大的词作为下一个词
#         next_token = torch.argmax(next_token_logits, dim=-1)  # (batch_size,)

#         # 检查哪些句子已经终止
#         finished = torch.logical_or(finished, next_token == end_symbol)

#         # 如果所有句子都已终止，提前退出循环
#         if finished.all():
#             break

#         # 将下一个词添加到目标序列中
#         next_token = next_token.unsqueeze(0)  # (1, batch_size)
#         tgt = torch.cat([tgt, next_token], dim=0)  # (seq_len + 1, batch_size)

#     # 截取每个句子到终止标记的位置
#     generated_sequences = []
#     for i in range(tgt.size(1)):
#         seq = tgt[:, i]
#         if end_symbol in seq:
#             end_idx = seq.tolist().index(end_symbol)
#             generated_sequences.append(seq[:end_idx + 1])
#         else:
#             generated_sequences.append(seq)
#     return generated_sequences



        
torch.manual_seed(42)

# 模拟编码器输入数据
batch_size = 5
src_len = 33
d_model = 1024
src = torch.randn(src_len, batch_size, 1024).to(device)
print(f'src_shape:{src.shape}')
# 模拟解码器输入数据
tgt_len = 6
tgt = torch.randint(0, 33, (tgt_len, batch_size)).to(device)
print(f'tgt_shape:{tgt.shape}')

# 生成随机序列
# batch_size = 5
# max_length = 32
# end_symbol = 33

# sequences = []
# lengths = []

# for _ in range(batch_size):
#     # 随机生成序列长度
#     seq_length = random.randint(1, max_length)
#     lengths.append(seq_length)
    
#     # 生成随机序列，元素为 1 到 seq_length 的索引值
#     sequence = torch.arange(1, seq_length + 1)
    
#     # 在序列后添加终止符号
#     sequence = torch.cat([sequence, torch.tensor([end_symbol])])
    
#     sequences.append(sequence)

# # 将序列填充到相同长度，并堆叠成一个二维张量
# tgt = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)

# print(f'tgt_shape形状:{tgt.shape}')



# model.train()
# seq = model(src,tgt)
# print(f'训练阶段的model outputs：{seq.shape}')
# print(seq)
# seq = eval(model,src)
# print(seq)
root_dir = "/ai/data/OFv2_ICL_VQA/agent_topk/data/ok_vqa_results_rs_clip.json"


dataset = VQASelectorDataset(question_path, shots_path,ori_path,label_path)
# 设定随机种子，确保结果可复现
torch.manual_seed(42)
# 将数据集拆分为训练集和验证集
total_samples = len(dataset)
train_size = int(0.8 * total_samples)  # 80%用作训练
val_size = total_samples - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=2)


#损失函数
criterion = nn.CrossEntropyLoss()

model = VQAShotSelector(d_model,nhead,num_encoder_layers, dim_feedforward, dropout, max_seq_length).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#model src:[33,batch_size,1024]
#model tgt:[batch_size, tgt_len]
print(model)
for epoch in range(num_epochs):
    for i,(question_feature,shots_feature,tgt,_) in enumerate(train_dataloader):
        #question_feature:[batch_size,1, 1024]
        #shots_feature:[batch_size,32,1024]
        #tgt:[batch_size,tgt_len]
        src_feature = torch.cat((question_feature,shots_feature),dim=1).transpose(0,1).to(device)

        tgt = tgt.to(device)

        output = model(src_feature,tgt).float()
        print(output.shape)
        print(tgt.shape)

        loss = criterion(output,tgt.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"epoch:{epoch+1} , iter:{i}, Loss: {loss.item()}")