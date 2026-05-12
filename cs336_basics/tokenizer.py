import regex as re
import os
from collections import defaultdict
from typing import Dict, Tuple, List, Iterable, Iterator

import pretokenization_example


def count_word_freq(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_processes: int = 40
) -> Dict[Tuple[bytes, ...], int]:

    word_freq: Dict[Tuple[bytes, ...], int] = defaultdict(int)

    if special_tokens is None:
        special_tokens = []

    pattern_split_text = "|".join(map(re.escape, special_tokens))
    pattern_split_word = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    with open(input_path, 'rb') as f:
        boundaries = pretokenization_example.find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        id_chunk = 1
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            text_list = re.split(pattern_split_text, chunk)
            
            for text in text_list:
                word_list = re.findall(pattern_split_word, text)

                for word in word_list:
                    if not word:
                        continue
                    # byte-level：每个 byte 是一个 token
                    key = tuple(ch.encode("utf-8") for ch in word)
                    word_freq[key] += 1

            print(f"共计{num_processes}个分块,第{id_chunk}个分块处理完毕.当前词表共有{len(word_freq)}个word.")
            id_chunk += 1

    return word_freq


def get_pair_stats(word_freq: Dict[Tuple[bytes, ...], int]):
    pairs = defaultdict(int)

    for word, freq in word_freq.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq

    return pairs


def merge_pair(pair: Tuple[bytes, bytes], word_freq: Dict[Tuple[bytes, ...], int]) -> Dict[Tuple[bytes, ...], int]:
    new_word_freq= {}

    for word, freq in word_freq.items():
        new_word = []
        i = 0

        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(word[i] + word[i + 1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        
        new_word = tuple(new_word)
        new_word_freq[new_word] = new_word_freq.get(new_word, 0) + freq

    return new_word_freq


def save_vocab(vocab, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for id, token in vocab.items():
            f.write(f"{id},{token.hex()}\n")


def save_merges(merges, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in merges:
            f.write(f"{pair[0].hex()},{pair[1].hex()}\n")


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # raise NotImplementedError

    word_freq= count_word_freq(input_path, special_tokens, num_processes = 32)

    vocab: dict[int, bytes] = {}
    token2id: dict[bytes, int] = {}
    merges: list[tuple[bytes, bytes]] = []

    next_token_id = 356

    for i, s_tok in enumerate(special_tokens):
            vocab[i] = s_tok.encode('utf-8')
            token2id[s_tok.encode('utf-8')] = i

    reserve = 100 # 预留100个special token
    # 将256个基础词添加到词表
    for i in range(256):
        vocab[reserve + i] = chr(i).encode("utf-8")
        token2id[chr(i).encode("utf-8")] = reserve + i

    # train loop
    for merge_step in range(vocab_size - next_token_id):
        # 统计所有相邻对的出现频率
        pair_freq = get_pair_stats(word_freq)

        if not pair_freq:
            break

        # 找到频率最高的对，如果频率相同选择字典序最大的
        max_freq = max(pair_freq.values())
        best_pairs = [pair for pair, freq in pair_freq.items() if max_freq == freq]
        best_pair = max(best_pairs)

        # 创建新 token
        new_token = best_pair[0] + best_pair[1]
        if new_token not in token2id:
            token2id[new_token] = next_token_id
            vocab[next_token_id] = new_token
            next_token_id += 1

        # 更新word_freq, 记录merge的pair
        word_freq = merge_pair(best_pair, word_freq)
        merges.append(best_pair)

        if merge_step % 500 == 0:
            best_pair_str = f"{best_pair[0].decode('utf-8', errors='ignore')} {best_pair[1].decode('utf-8', errors='ignore')}"
            print(f"第{merge_step + 1}次合并: 合并 '{best_pair_str}' (频率: {max_freq}) \n")

    return vocab, merges


class BPETokenizer:
    def __init__(self, 
                vocab: dict[int, bytes], 
                merges: list[tuple[bytes, bytes]],
                special_tokens: list[str] | None = None):

        self.id2token = vocab
        if special_tokens is not None:
            self.special_tokens = special_tokens
            for i, s_tok in enumerate(special_tokens):
                self.id2token[i] = s_tok.encode('utf-8')

        self.token2id = {token: id for id, token in self.id2token.items()}

        self.merges = merges

        
    @classmethod    
    def from_files(cls,
                    vocab_filepath: str,
                    merges_filepath: str,
                    special_tokens: list[str] | None = None):

        vocab: dict[int, bytes] = {}
        merges: list[tuple[bytes, bytes]] = []

        # _parse_bytes_repr = lambda s: s[2:-1].encode('utf-8')
        
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip('\n')
                if not line:
                    continue

                part = line.split(',', 1)
                if len(part) != 2:
                    raise ValueError(f"Invalid vocab line: {line}")

                id = int(part[0])
                token = bytes.fromhex(part[1])

                vocab[id] = token

        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip('\n')
                if not line:
                    continue

                pair = line.split(',', 1)
                if len(pair) != 2:
                    raise ValueError(f"Invalid vocab line: {line}")

                merges.append((bytes.fromhex(pair[0]), bytes.fromhex(pair[1])))

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []

        # pre-tokenize
        # 自动转义并拼接特殊标记
        special_pattern = '|'.join(re.escape(token) for token in self.special_tokens)
        pattern = f"{special_pattern}|" + r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

        pre_token_list = re.findall(pattern, text)

        for pre_token in pre_token_list:

            pre_token_bytes = pre_token.encode('utf-8')

            if pre_token in self.special_tokens:
                ids.append(self.token2id[pre_token_bytes])
                continue

            symbols = [bytes([b]) for b in pre_token_bytes]

            while True:
                merged = False

                for pair in self.merges:

                    i = 0
                    while i < len(symbols) - 1:

                        current_pair = (symbols[i], symbols[i + 1])

                        if current_pair == pair:

                            merged_token = symbols[i] + symbols[i + 1]

                            symbols = symbols[:i] + [merged_token] + symbols[i + 2:]

                            merged = True

                            break

                        i += 1

                    if merged:
                        break
                
                if not merged:
                    break

            ids.extend([self.token2id[token] for token in symbols])

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:

        for text in iterable:    # file handle 实际就是逐行读取

            for token_ids in self.encode(text):
                yield token_ids
                # for t_id in token_ids:
                #     yield t_id

    def decode(self, ids: list[int]) -> str:
        bytes_chunks = []

        for id in ids:
            bytes_chunks.append(self.id2token[id])

        raw_bytes = b"".join(bytes_chunks)

        return raw_bytes.decode("utf-8", errors="replace")

            