import regex as re
import json
from collections.abc import Iterable, Iterator
from cs336_basics.bpe import PAT
class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens) if special_tokens else []

        # 不在词表里的追加到末尾
        for tok in self.special_tokens:
            b = tok.encode("utf-8")
            if b not in self.vocab.values():
                self.vocab[len(self.vocab)] = b

        self._id_to_bytes = dict(self.vocab)
        self._bytes_to_id = {b: i for i, b in self.vocab.items()} # 变量名前加一个单下划线--供类内部使用的结构，不建议在类的外部直接访问或修改
        self._ranks = {a + b: i for i, (a, b) in enumerate(self.merges)}

        # special token 正则：先处理长的

        self._special_set = set(self.special_tokens)
        if self.special_tokens:
            special_alt = "|".join(
                re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True)
            ) # 长的 special token在前面

            self._special_re = re.compile(f"({special_alt})") # 转换成可使用的正则分割
        else:
            self._special_re = None

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        # 要优先处理 special tokens
        if self._special_re is None:
            return self._encode_ordinary(text)
        for part in self._special_re.split(text):
            if part == "":
                continue
            if part in self._special_set:
                ids.append(self._bytes_to_id[part.encode("utf-8")])
            else:
                ids.extend(self._encode_ordinary(part))
        return ids

    def _encode_ordinary(self, text: str) -> list[int]:
        ids: list[int] = []
        for m in re.finditer(PAT, text):
            ids.extend(self._encode_pretoken(m.group().encode("utf-8")))
        return ids

    def _encode_pretoken(self, piece: bytes) -> list[int]:
        tokens = [bytes([b]) for b in piece]
        while len(tokens) >= 2:
            min_idx = None
            min_rank = None
            for i in range(len(tokens) - 1):
                rank = self._ranks.get(tokens[i] + tokens[i + 1])
                if rank is not None and (min_rank is None or rank < min_rank):
                    min_idx = i
                    min_rank = rank
            if min_rank is None:
                break

            tokens = tokens[:min_idx] + [tokens[min_idx] + tokens[min_idx + 1]] + tokens[min_idx + 2:]
        return [self._bytes_to_id[t] for t in tokens]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # 如果没有定义特殊token，直接退化为普通流式编码
        if self._special_re is None:
            yield from self._encode_ordinary_iterable(iterable)
            return
        buffer = ""
        max_special = max([len(t) for t in self.special_tokens], default=0)
        for chunk in iterable:
            buffer += chunk
            if len(buffer) <= max_special:
                continue
            smatches = list(self._special_re.finditer(buffer))

            if smatches:
                cut = smatches[-1].start()
                yield from self._encode_complete_region(buffer[: cut])
            else: # 没有 special token
                safe = buffer[:-max_special]
                matches = list(re.finditer(PAT, safe))
                for m in matches[:-1]:
                    yield from self._encode_pretoken(m.group().encode("utf-8"))
                cut = matches[-1].start() if matches else len(safe)
            buffer = buffer[cut:]
        # buffer 可能还有剩余
        yield from self._encode_complete_region(buffer)



    def _encode_complete_region(self, text: str) -> Iterator[int]:
        for part in self._special_re.split(text):
            if part == "":
                continue
            if part in self._special_set:
                yield self._bytes_to_id[part.encode("utf-8")]
            else:
                yield from self._encode_ordinary(part)

    def _encode_ordinary_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        for chunk in iterable:
            buffer += chunk
            matches = list(re.finditer(PAT, buffer))
            if not matches:
                continue
            for m in matches[:-1]:
                yield from self._encode_pretoken(m.group().encode("utf-8"))
            buffer = buffer[matches[-1].start() :]
        for m in re.finditer(PAT, buffer):
            yield from self._encode_pretoken(m.group().encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return b"".join(self._id_to_bytes[i] for i in ids).decode("utf-8", errors="replace")


    def save(self, vocab_filepath: str, merges_filepath: str) -> None:
        with open(vocab_filepath, "w", encoding="utf-8") as f:
            json.dump({str(i): list(b) for i, b in self.vocab.items()}, f)
        with open(merges_filepath, "w", encoding="utf-8") as f:
            json.dump([[list(a), list(b)] for a, b in self.merges], f)

    @classmethod
    def from_files(
        cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None
    ) -> "Tokenizer":
        with open(vocab_filepath, encoding="utf-8") as f:
            vocab_raw = json.load(f)
        vocab = {int(k): bytes(v) for k, v in vocab_raw.items()}
        with open(merges_filepath, encoding="utf-8") as f:
            merges_raw = json.load(f)
        merges = [(bytes(a), bytes(b)) for a, b in merges_raw]
        return cls(vocab, merges, special_tokens)
