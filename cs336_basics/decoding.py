import torch


def decode(model, 
           input_tokens,
           max_length=1024,
           temperature=0.8,
           top_p=0.9
           ):
    
    """
    input_tokens -> (sequence_length, vocab_size)
    """
    if len(input_tokens) >= max_length:
        return input_tokens
    
    infer_nums = max_length - len(input_tokens)

    for _ in range(infer_nums):

        predict_token = model.forward(input_tokens)[-1]

        tmp = torch.exp(predict_token / temperature)

        predict_token = tmp / tmp.sum()

        sorted_probs, sorted_idx = torch.sort(predict_token, descending=True)

        cumulative_prob = 0.0
        candidate_num = 0

        for prob in sorted_probs:

            cumulative_prob += prob
            candidate_num += 1

            if cumulative_prob >= top_p:
                break

        sorted_probs[candidate_num:-1] = 0

        sorted_probs /= sorted_probs.sum(sorted_probs)

        sampled = torch.multinomial(sorted_probs, num_samples=1)

        token = sorted_idx.gather(-1, sampled)





