import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


def get_decode_result(x, tokenizer):
    return tokenizer.batch_decode(x[:, x.shape[1]//2:], skip_special_tokens=True)[0]

@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, tokenizer=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    # 创建 shape = (1, prompt长度+gen_length) 的张量，全部初始化为 mask_id
    # 前半部分根据输入的 prompt 填充，后半部分为待生成区域（全是 mask）
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    # bool 张量，标记哪些 token 是 prompt 内容（不是 mask）
    prompt_index = (x != mask_id)

    # 生成长度必须能被 block 长度整除（便于后续分块采样）
    assert gen_length % block_length == 0
    # 按照block为单位进行便利生成
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    # 计算每个 block 的扩散步数（细分步数）
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        # 把 x 当前 block 区域里哪些位置是 mask 标记出来（mask_index）
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        # 计算本 block 里每一步要去除多少个 mask（保证整个 block 匀速“扩散恢复”）
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        # 遍历每个细分步
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                # 预测每个被 [MASK] 掩盖位置最可能是什么 token
                logits = model(x).logits

            # 用采样策略挑出置信度低或随机几个位置，正式“揭开”填上预测词
            # 新的 [MASK] 格局输入给模型，继续推理，重复步骤，直到生成区域全部“揭开”

            # 给 logits 添加 Gumbel noise（以温度参数控制采样随机性），促进更平滑的采样策略
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            # 根据 logits 计算每个位置的预测 token（argmax）
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            # 'low_confidence' 时，对应 token 按 softmax 概率作为置信度，置信度低的优先揭开
            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            # 'random' 时，全部随机分数
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            # 本步只生成当前 block，不是当前block以外的内容置信度全部设为 -∞，不会被采样到
            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            # 只对当前还没揭开的 mask 位使用新预测/置信度，其他位保持原样或设为极低置信度
            x0 = torch.where(mask_index, x0, x)
            # 根据 mask_index 选择 x0_p 或 -np.inf（-np.inf 表示置信度无限低）
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

            print(f"num block {num_block} step {i} result:\n{get_decode_result(x, tokenizer)}")
            print("-"*50)

    return x


def main():
    device = 'cuda'
    model_path = "/mnt/youwei-data/zhuohang/model/GSAI-ML/LLaDA-8B-Instruct"

    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence', tokenizer=tokenizer)
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])


if __name__ == '__main__':
    main()
