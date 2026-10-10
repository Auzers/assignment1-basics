import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pretokenize(text: str) -> list[tuple[bytes, ...]]:
    """用 GPT-2 regex 预分词，并把每个 pre-token 编码成 UTF-8 单字节元组。"""
    return [
        tuple(bytes([b]) for b in m.group().encode("utf-8"))
        for m in re.finditer(PAT, text)
    ]


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    # special token 是硬边界：先从语料里切掉，不参与 pair 统计。
    if special_tokens:
        delim = "|".join(re.escape(t) for t in special_tokens)
        segments = re.split(delim, text)
    else:
        segments = [text]

    # 预分词 + 计数：word（单字节元组）-> 出现次数
    word_counts: dict[tuple[bytes, ...], int] = {}
    for seg in segments:
        for word in pretokenize(seg):
            word_counts[word] = word_counts.get(word, 0) + 1 # 没有word就是0 + 1

    # 初始化词表：special tokens 占最前面的 ID，然后是 256 个字节。
    vocab: dict[int, bytes] = {}
    for i, tok in enumerate(special_tokens):
        vocab[i] = tok.encode("utf-8")
    for b in range(256):
        vocab[len(special_tokens) + b] = bytes([b])

    merges = []
    num_merges = vocab_size - len(vocab)
    for _ in range(num_merges):
        pair_counts = {}
        for word, count in word_counts.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + count
        if not pair_counts:
            break
    
        
        best = max(pair_counts, key=lambda p: (pair_counts[p], p))
        merged = best[0] + best[1]
        merges.append(best)
        vocab[len(vocab)] = merged

        # 由于新token的加入，更新word_count
        new_word_counts = {}
        for word, count in word_counts.items():
            new_word = [] # 不能直接写 new_word = () 因为元组不可变，必须先写list然后转换成tuple
            i = 0
            while i < len(word):
                if i + 1 < len(word) and (word[i], word[i + 1]) == best:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_t = tuple(new_word)
            new_word_counts[new_word_t] = new_word_counts.get(new_word_t, 0) + count
        word_counts = new_word_counts
    return vocab, merges


  