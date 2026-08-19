import torch
import numpy.typing as npt
import numpy as np
import os
import sys

from cs336_basics.utils.tokenizer import BPETokenizer

def tokenize_and_save(dataset_path, tokenizer, output_path, max_length=None):

    if os.path.exists(os.path.join(output_path, "tokens.npy")):
        return

    vocab_size = tokenizer.vocab_size
    assert vocab_size <= 65535,  f"Vocabulary {vocab_size} exceeds uint16 limit"
    print(f"✓ Vocabulary size: {vocab_size} (< 65,535)")

    f = open(dataset_path, "r", encoding='utf-8')

    all_token_ids = []

    for id in tokenizer.encode_iterable(f):
        all_token_ids.append(id)

    f.close()

    # Convert to uint16 numpy array
    token_array = np.array(all_token_ids, dtype=np.uint16)

    # Save to disk
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, "tokens.npy")
    np.save(output_path, token_array)
    
    print(f"✓ Saved {len(token_array):,} tokens to {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1e6:.2f} MB")
    print(f"  dtype: {token_array.dtype}")

    return token_array


def tokenize_and_save_mmap(dataset_path, tokenizer, output_path, max_length=None):
    """
    使用内存映射方式处理tokenization
    """
    if os.path.exists(os.path.join(output_path, "tokens.npy")):
            return
    
    vocab_size = tokenizer.vocab_size
    assert vocab_size <= 65535, f"Vocabulary {vocab_size} exceeds uint16 limit"
    print(f"✓ Vocabulary size: {vocab_size} (< 65,535)")
    
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "tokens.npy")
    temp_file = os.path.join(output_path, "tokens.temp.npy")
    
    # 第一次遍历：计算总token数
    print("第一次遍历：统计token总数...")
    total_tokens = 0
    with open(dataset_path, "r", encoding='utf-8') as f:
        for text in f:
            if text.strip():
                tokens = tokenizer.encode(text.strip())
                if max_length:
                    tokens = tokens[:max_length]
                total_tokens += len(tokens)
    print(f"  总token数: {total_tokens:,}")
    
    # 创建内存映射文件
    print("创建内存映射文件...")
    token_array = np.memmap(temp_file, dtype=np.uint16, mode='w+', shape=(total_tokens,))
    
    # 第二次遍历：填充数据
    print("第二次遍历：填充数据...")
    current_pos = 0
    with open(dataset_path, "r", encoding='utf-8') as f:
        for text in f:
            if text.strip():
                tokens = tokenizer.encode(text.strip())
                if max_length:
                    tokens = tokens[:max_length]
                token_array[current_pos:current_pos + len(tokens)] = tokens
                current_pos += len(tokens)
                
                # 定期输出进度
                if current_pos % 1000000 == 0:
                    print(f"  已处理 {current_pos:,} / {total_tokens:,} tokens...")
    
    # 刷新内存映射
    del token_array
    
    # 复制为普通numpy文件（可选）
    print("保存最终文件...")
    data = np.memmap(temp_file, dtype=np.uint16, mode='r', shape=(total_tokens,))
    np.save(output_file, data)
    del data
    
    # 删除临时文件
    os.remove(temp_file)
    
    print(f"✓ Saved {total_tokens:,} tokens to {output_file}")
    print(f"  File size: {os.path.getsize(output_file) / 1e6:.2f} MB")
    
    # 返回文件路径而不是加载到内存
    return output_file


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """

    # 随机采样起始位置
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    
    # 构造输入
    x = np.stack([dataset[i:i+context_length] for i in starts])

    # 构造标签(右移一位)
    y = np.stack([dataset[i+1:i+context_length+1] for i in starts])

    # 转为tensor
    x = torch.tensor(x, dtype=torch.long, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)

    return x, y


"""
Usage
tokens = tokenizer.encode(text)

arr = np.array(tokens, dtype=np.uint16)

np.save("train.npy", arr)

train_data = np.load(
    "train.npy",
    mmap_mode="r"
)

while True:
    x, y = get_batch(train_data, ...)
"""

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("用法: python script.py <vocab_size>") 
        sys.exit(1)

    vocab_filepath = os.path.join('..', 'data', 'TinyStoriesV2-GPT4', f'vs_{sys.argv[1]}', 'vocab.csv')
    merges_filepath = os.path.join('..', 'data', 'TinyStoriesV2-GPT4', f'vs_{sys.argv[1]}', 'merges.csv')

    tokenizer = BPETokenizer.from_files(
                                       vocab_filepath=vocab_filepath,
                                       merges_filepath=merges_filepath,
                                       special_tokens=['<|endoftext|>']
                                       )
    

    train_path = os.path.join('..', '..', "data", "TinyStoriesV2-GPT4-train.txt")
    valid_path = os.path.join('..', '..', "data", "TinyStoriesV2-GPT4-valid.txt")
    output_dir = os.path.join("..", "data", "TinyStoriesV2-GPT4", f"vs_{tokenizer.vocab_size}", "token_ids")

    tokenize_and_save_mmap(train_path, tokenizer, os.path.join(output_dir, "train"))
    tokenize_and_save(valid_path, tokenizer, os.path.join(output_dir, "valid"))