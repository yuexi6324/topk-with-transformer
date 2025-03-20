import torch

class Config():
    """
        Args for top-k transformer-pointer model
    """

    """file path"""
    question_path = '/ai/data/OFv2_ICL_VQA/agent_topk/data/ok_vqa_question_rs_clip.npy' 
    shots_path = '/ai/data/OFv2_ICL_VQA/agent_topk/data/ok_vqa_shots_rs_clip.npy'
    ori_path = '/ai/data/OFv2_ICL_VQA/agent_topk/data/ok_vqa_rs_clip.json'
    label_path = '/ai/data/OFv2_ICL_VQA/agent_topk/data/ok_vqa_results_rs_clip.json'
    dataset_name = 'ok_vqa'

    """Model params"""
    d_model = 512
    nhead = 8  # 多头注意力的头数
    num_encoder_layers = 6  # 编码器层数
    num_decoder_layers = 6  # 解码器层数

    dim_feedforward = 1024  # 前馈网络的维度
    dropout = 0.1
    max_seq_length = 5  # 最大序列长度
    max_sequence_length = max_seq_length

    tgt_vocab_size = 34 #输出的序列id

    start_sign_id = 0
    end_sign_id = 33

    """train params"""
    num_workers = 4
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """mask config"""
    NEG_INFTY = 1e-9 