from typing import Dict, Tuple, List, Iterable, Iterator

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

            