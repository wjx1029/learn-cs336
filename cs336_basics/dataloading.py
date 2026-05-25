import torch
import numpy.typing as npt
import numpy as np
import os

def tokenize_and_save(dataset_path, tokenizer, output_path, max_length=None):
    vocab_size = len(tokenizer.id2token)
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
