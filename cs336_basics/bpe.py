import regex as re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str]) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    # 1. 初始化
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    
    # 预分词与统计
    word_freqs = Counter()
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pattern = re.compile(PAT)
    
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
        # 按特殊 token 切割
        if special_tokens:
            split_pattern = "|".join(re.escape(st) for st in special_tokens)
            chunks = re.split(split_pattern, text)
        else:
            chunks = [text]
            
        for chunk in chunks:
            for match in pattern.finditer(chunk):
                word = match.group()
                word_bytes = tuple(bytes([b]) for b in word.encode("utf-8"))
                word_freqs[word_bytes] += 1

    # ================= 优化开始 =================
    
    # 全局统计所有相邻对的频率
    pair_freqs = Counter()
    # 记录每个 pair 出现在哪些词里（用于后续快速定位）
    # 结构: pair -> set(包含该 pair 的 word)
    pair_to_words = defaultdict(set) 

    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            pair_freqs[pair] += freq
            pair_to_words[pair].add(word)

    merges = []
    
    # 2. 核心合并循环
    while len(vocab) < vocab_size:
        if not pair_freqs:
            break
            
        # 找最高频 pair (平局时选字典序大的)
        best_pair = max(pair_freqs.keys(), key=lambda p: (pair_freqs[p], p))
        new_token_bytes = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token_bytes
        merges.append(best_pair)
        
        # --- 增量更新：只处理包含 best_pair 的词 ---
        # 找出所有含有 best_pair 的旧词
        words_to_update = pair_to_words[best_pair].copy() 
        
        for word in words_to_update:
            freq = word_freqs[word]
            
            # 1. 从全局 pair_freqs 中减去这个旧词的所有 pair 频率
            for i in range(len(word) - 1):
                old_pair = (word[i], word[i+1])
                pair_freqs[old_pair] -= freq
                if pair_freqs[old_pair] == 0:
                    del pair_freqs[old_pair]
                # 从 pair_to_words 中移除
                pair_to_words[old_pair].discard(word)
            
            # 2. 对这个词进行合并操作（用之前的 while 指针逻辑）
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == best_pair[0] and word[i+1] == best_pair[1]:
                    new_word.append(new_token_bytes)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            
            # 3. 将新词加入字典，并更新频次
            word_freqs[new_word] += freq
            del word_freqs[word]
            
            # 4. 将新词的所有相邻对加入全局统计
            for i in range(len(new_word) - 1):
                new_pair = (new_word[i], new_word[i+1])
                pair_freqs[new_pair] += freq
                pair_to_words[new_pair].add(new_word)
        # ================= 优化结束 =================

    return vocab, merges


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.bytes_to_id = {v: k for k, v in vocab.items()}
        
        # 【修复1】：特殊 token 排序与正则分组
        special_pat = ""
        if self.special_tokens:
            # 按长度降序排列，防止重叠截断（比如长 token 被短的抢先匹配）
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            # 用括号分组，确保它们作为一个整体匹配
            special_pat = "(" + "|".join(re.escape(st) for st in sorted_specials) + ")|"
            
        PAT = special_pat + r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.pattern = re.compile(PAT)

    def encode(self, text: str) -> list[int]:
        token_ids = []
        
        # 【绝对防御】：如果存在特殊 token，先用字符串的 split 把它切开
        if self.special_tokens:
            # 按长度降序排列，先切长的，再切短的，防止误伤
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            
            # 构建一个正则，用于按特殊 token 分割文本
            split_pattern = "(" + "|".join(re.escape(st) for st in sorted_specials) + ")"
            
            # 使用 re.split，这样文本会被特殊 token 切分成多段，且特殊 token 也会保留在列表中
            parts = re.split(split_pattern, text)
            
            for part in parts:
                if not part:
                    continue
                # 如果这一段刚好是个特殊 token，直接转成 ID 存入，跳过 BPE
                if part in self.special_tokens:
                    token_ids.append(self.bytes_to_id[part.encode("utf-8")])
                else:
                    # 如果是普通文本，递归调用真正的 BPE 编码逻辑
                    token_ids.extend(self._encode_normal(part))
            return token_ids
        else:
            # 如果没有特殊 token，直接走正常逻辑
            return self._encode_normal(text)

    def _encode_normal(self, text: str) -> list[int]:
        # 把原来 encode 里的 BPE 逻辑原封不动搬到这里
        token_ids = []
        cache = {}
        for match in self.pattern.finditer(text):
            word = match.group()
            if word in cache:
                token_ids.extend(cache[word])
                continue
                
            word_bytes = [bytes([b]) for b in word.encode("utf-8")]
            for pair in self.merges:
                i = 0
                new_word = []
                while i < len(word_bytes):
                    if i < len(word_bytes) - 1 and word_bytes[i] == pair[0] and word_bytes[i+1] == pair[1]:
                        new_word.append(pair[0] + pair[1])
                        i += 2
                    else:
                        new_word.append(word_bytes[i])
                        i += 1
                word_bytes = new_word
                
            encoded_word = [self.bytes_to_id.get(b, 0) for b in word_bytes]
            cache[word] = encoded_word
            token_ids.extend(encoded_word)
        return token_ids
        
    def decode(self, ids: list[int]) -> str:
        byte_list = [self.vocab.get(i, b'') for i in ids]
        return b''.join(byte_list).decode("utf-8", errors="replace")

    def encode_iterable(self, iterable):
        """
        极速流式编码。按固定大小块读取（默认 1MB）并处理块边界截断。
        """
        # 这里的 iterable 是文件句柄，我们按 1MB 的大小来读
        chunk_size = 1024 * 1024 
        
        # 缓冲区，用来存放上一次没读完的“半截单词”
        buffer = ""
        
        while True:
            # 1. 每次读 1MB。如果是二进制文件，这里直接读 bytes
            chunk = iterable.read(chunk_size)
            if not chunk:
                break
                
            # 2. 将之前的“半截单词”和这一次读的数据拼接
            chunk = buffer + chunk
            
            # 3. 寻找最后一个空格或换行符（确保不会把单词切断）
            # 如果当前块还有后续（没到文件尾），就找到最后一个空格切分
            last_space_idx = max(chunk.rfind(" "), chunk.rfind("\n"))
            
            if last_space_idx != -1 and len(chunk) == chunk_size:
                # 如果找到了空格，就把空格之前的全编码了
                text_to_encode = chunk[:last_space_idx]
                # 把空格之后的部分留作下一次的 buffer
                buffer = chunk[last_space_idx:]
            else:
                # 如果没找到空格（或者文件只有一行，没有空格），就只能全量编码
                text_to_encode = chunk
                buffer = ""
                
            # 4. 对切分好的安全文本块进行编码
            for token_id in self.encode(text_to_encode):
                yield token_id
                
        # 5. 循环结束后，把最后遗留的 buffer 也编码掉
        if buffer:
            for token_id in self.encode(buffer):
                yield token_id