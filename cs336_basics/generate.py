from cs336_basics.utils.checkpointing import load_checkpoint
from cs336_basics.utils.decode import decode
from cs336_basics.prenorm_transformer_block import TransformerModel
from cs336_basics.utils.tokenizer import BPETokenizer
from cs336_basics.utils.device import try_gpu
import torch

device = try_gpu()

vocab_size=10000

model = TransformerModel(
        vocab_size=vocab_size,
        context_length=256,
        num_layers=4,
        d_model=512,
        num_heads=16,
        rope_theta=1e4,
    ).to(device)

load_checkpoint(src='runs/base_iter_10000/best.pth',
                model=model,
                optimizer=None
                )

bpe_tokenizer = BPETokenizer.from_files(vocab_filepath=f'./data/TinyStoriesV2-GPT4/vs_{vocab_size}/vocab.csv',
                                        merges_filepath=f'./data/TinyStoriesV2-GPT4/vs_{vocab_size}/merges.csv',
                                        special_tokens=['<|endoftext|>']
                                        )

prompt = "Once upon a time"
print(f"prompt:\n{prompt}")

tokens = bpe_tokenizer.encode(prompt)
tokens = torch.tensor(
    tokens,
    dtype=torch.long,
    device=device
).unsqueeze(0)

# print(tokens)

output = decode(model=model,
                input_tokens=tokens,
                max_length=256,
                )

# print(output)

output = bpe_tokenizer.decode(output.squeeze(0).tolist())

print(f"model output:\n{output}")

