import torch


def decode(
    model,
    input_tokens,
    max_length=1024,
    temperature=0.8,
    top_p=0.9,
):
    while input_tokens.size(-1) < max_length:

        logits = model(input_tokens)[:,-1,:]

        probs = torch.softmax(
            logits / temperature,
            dim=-1
        )

        sorted_probs, sorted_idx = torch.sort(
            probs,
            descending=True
        )

        cumulative_probs = torch.cumsum(
            sorted_probs,
            dim=-1
        )

        mask = cumulative_probs > top_p

        # 保留第一个超过top_p的token
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False

        sorted_probs[mask] = 0

        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

        sampled = torch.multinomial(
            sorted_probs,
            num_samples=1
        )

        next_token = sorted_idx.gather(
            -1,
            sampled
        )

        if next_token.item() == 0:
            break

        input_tokens = torch.cat(
            [input_tokens, next_token],
            dim=-1
        )

    return input_tokens


# test

# class DummyModel:
#     def __call__(self, tokens):
#         # vocab_size = 5
#         return torch.tensor([[
#             1.0,   # token 0
#             2.0,   # token 1
#             5.0,   # token 2
#             0.5,   # token 3
#             0.1    # token 4
#         ]])

# model = DummyModel()

# input_tokens = torch.tensor([1, 2])

# output = decode(
#     model,
#     input_tokens,
#     max_length=10,
#     temperature=1.0,
#     top_p=0.9
# )

# print(output)


