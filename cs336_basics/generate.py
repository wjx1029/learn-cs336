from cs336_basics.utils.checkpointing import load_checkpoint
from cs336_basics.utils.decode import decode
from cs336_basics.prenorm_transformer_block import TransformerModel
from cs336_basics.utils.tokenizer import BPETokenizer
from cs336_basics.utils.device import try_gpu
import torch

device = try_gpu()

model = TransformerModel(
        vocab_size=10000,
        context_length=256,
        num_layers=4,
        d_model=512,
        num_heads=16,
        rope_theta=1e4,
        rms_eps=1e-5,
    ).to(device)

load_checkpoint(src='checkpoints/model_best.ckpt',
                model=model,
                optimizer=None
                )

bpe_tokenizer = BPETokenizer.from_files(vocab_filepath='./data/TinyStoriesV2-GPT4/vocab.csv',
                                        merges_filepath='./data/TinyStoriesV2-GPT4/merges.csv',
                                        special_tokens=['<|endoftext|>']
                                        )

prompt = "Once upon a time"

tokens = bpe_tokenizer.encode(prompt)
tokens = torch.tensor(
    tokens,
    dtype=torch.long,
    device=device
).unsqueeze(0)

print(tokens)

output = decode(model=model,
                input_tokens=tokens,
                max_length=128,
                )

print(output)

output = bpe_tokenizer.decode(output.squeeze(0).tolist())

print(output)
