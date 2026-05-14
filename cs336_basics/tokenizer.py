import regex as re
import os
from collections import defaultdict
from typing import Dict, Tuple, List, Iterable, Iterator

class BPETokenizer:
    def __init__(self, 
                vocab: dict[int, bytes], 
                merges: list[tuple[bytes, bytes]],
                special_tokens: list[str] | None = None):

        self.id2token = vocab

        self.token2id = {token: id for id, token in self.id2token.items()}

        self.merges = {pair : i for i, pair in enumerate(merges)}

        self.special_tokens = special_tokens or []

        # 构建特殊 Token 的正则表达式
        if self.special_tokens:
             # 这样正则引擎会优先匹配最长的特殊标记，防止重叠标记（如 <|a|><|b|>）被错误拆分。
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            # 使用 re.escape 确保标记中的特殊字符（如 | 或 [ ）被当作普通字符处理
            special_pattern = "|".join(re.escape(t) for t in sorted_special)
            self.special_regex = re.compile(special_pattern)
        else:
            self.special_regex = None

        self.gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
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

        if not text:
            return []

        # 如果我们在初始化时没有定义任何特殊标记（或者特殊标记列表为空），
        # 那么整个文本都可以被视为一段连续的“普通文本”。
        # 我们直接调用内部方法 _encode_text_segment 进行 BPE 处理并返回结果。
        if not self.special_regex:
            return self._encode_text_segment(text)

        tokens = []

        last_pos = 0    # last_pos 用于记录上一次匹配结束的位置，帮助我们定位“特殊标记”之间的“缝隙”。

        # 使用 finditer 遍历文本中所有符合特殊标记模式的匹配项。
        # finditer 的好处是它提供了 match.start() 和 match.end()，
        # 这让我们能够精确地知道特殊标记在哪里开始，在哪里结束。
        for match in self.special_regex.finditer(text):

            # 1.提取并处理“前置普通文本”
            # 这里的区间是 [last_pos, match.start())。
            # " hello <|endoftext|> world"
            # 这段文本是夹在两个特殊标记之间（或者开头到第一个特殊标记之间）的普通文字。
            pre_text = text[last_pos:match.start()]

            # 如果这两个标记之间确实有文字（长度 > 0）
            if len(pre_text) > 0:
                tokens.extend(self._encode_text_segment(pre_text))

            # 2.处理“当前特殊标记”
            # match.group() 拿到的就是被识别出来的特殊标记字符串（如 "<|endoftext|>"）。
            special_tok = match.group()
            tokens.append(self.token2id[special_tok.encode('utf-8')])

            # 更新游标
            # 将游标移动到当前匹配项的末尾，为寻找下一个片段做准备
            last_pos = match.end()

        # 如果最后一个特殊标记后面还有文字（例如 "Hello<|end|>World" 中的 "World"），
        # 或者整个文本根本没有特殊标记匹配（虽然逻辑上 Case A 已处理，但这里是双重保险），
        # 我们需要处理从 last_pos 到字符串末尾的所有剩余字符。
        remaining_text = text[last_pos:]
        if len(remaining_text) > 0:
            tokens.extend(self._encode_text_segment(remaining_text))
        
        return tokens
    
    def _encode_text_segment(self, text: str) -> list[int]:
        ids = []
        pre_token_list = self.gpt2_pat.findall(text)

        for pre_token in pre_token_list:
            # 第一步：将当前片段转为字节序列，并将每个字节看作一个独立的“部分（Part）”
            # "Hello" -> [b'H', b'e', b'l', b'l', b'o']
            symbols = [bytes([b]) for b in pre_token.encode('utf-8')]

            # 第二步：反复执行合并，直到没有符合条件的合并规则为止
            while len(symbols) >= 2:

                best_pair = None
                min_rank = float('inf')

                for i in range(len(symbols) - 1):
                    pair = (symbols[i], symbols[i+1])
                    if pair in self.merges:
                        rank = self.merges[pair]
                        if rank < min_rank:
                            min_rank = rank
                            best_pair = pair

                # 如果找不到任何可以合并的规则，退出当前片段的合并过程
                if best_pair is None:
                    break

                # 第三步：执行合并操作。
                # 遍历当前序列，将所有出现的 best_pair 替换成合并后的长字节块。
                new_symbols = []
                i = 0
                while i < len(symbols):
                    if i < len(symbols) - 1 and (symbols[i], symbols[i+1]) == best_pair:
                        new_symbols.append(best_pair[0] + best_pair[1])
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                symbols = new_symbols   # 更新序列，进入下一轮 while 循环

            ids.extend([self.token2id[token] for token in symbols])

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:

        for text in iterable:    # file handle 实际就是逐行读取
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        bytes_chunks = []

        for id in ids:
            bytes_chunks.append(self.id2token[id])

        raw_bytes = b"".join(bytes_chunks)

        return raw_bytes.decode("utf-8", errors="replace")


if __name__ == "__main__":

    vocab_filepath = "./data/TinyStoriesV2-GPT4/vocab.csv"
    merges_filepath = "./data/TinyStoriesV2-GPT4/merges.csv"
    special_tokens = ["<|endoftext|>"]

    bpe_tokenizer = BPETokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)

    tokens = bpe_tokenizer.encode("\nOnce upon a time there was a little boy named Ben.")

    for tokenid in tokens:
        print(tokenid)

    print(bpe_tokenizer.decode(tokens))