from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
import json
from abc import ABC, abstractmethod

@dataclass
class VocabularyItem:
    """Represents a token in the vocabulary"""
    id: int
    text: str
    frequency: int = 0

class TokenizerBase(ABC):
    """Abstract base class for tokenizers"""
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        pass
    
    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        pass

class VocabularyManager:
    """Manages the tokenizer vocabulary and its operations"""
    def __init__(self):
        self._id_to_token: Dict[int, str] = {}
        self._token_to_id: Dict[str, int] = {}
        self._next_id: int = 0

    def add_token(self, token: str) -> int:
        """Add a new token to the vocabulary"""
        if token in self._token_to_id:
            return self._token_to_id[token]
        
        token_id = self._next_id
        self._id_to_token[token_id] = token
        self._token_to_id[token] = token_id
        self._next_id += 1
        return token_id

    def get_token(self, token_id: int) -> Optional[str]:
        """Get token string from token ID"""
        return self._id_to_token.get(token_id)

    def get_id(self, token: str) -> Optional[int]:
        """Get token ID from token string"""
        return self._token_to_id.get(token)

    def __len__(self) -> int:
        return len(self._id_to_token)

class BPEMergeTracker:
    """Tracks BPE merge operations and their frequencies"""
    def __init__(self):
        self._merges: Dict[Tuple[int, int], int] = {}
        self._frequencies: Counter = Counter()

    def add_merge(self, pair: Tuple[int, int], new_id: int):
        """Record a new merge operation"""
        self._merges[pair] = new_id

    def get_merge_id(self, pair: Tuple[int, int]) -> Optional[int]:
        """Get the resulting token ID for a merge pair"""
        return self._merges.get(pair)

    def update_frequencies(self, token_ids: List[int]):
        """Update pair frequencies from a sequence of token IDs"""
        self._frequencies.clear()
        self._frequencies.update(zip(token_ids, token_ids[1:]))

    def get_most_frequent_pair(self) -> Optional[Tuple[int, int]]:
        """Get the most frequent pair of tokens"""
        if not self._frequencies:
            return None
        return max(self._frequencies.items(), key=lambda x: x[1])[0]

class TextPreprocessor:
    """Handles text preprocessing operations"""
    def __init__(self, special_tokens: Set[str] = None):
        self.special_tokens = special_tokens or {"<|endoftext|>"}
        self.WORD_SEPARATOR = "Ġ"

    def preprocess(self, text: str) -> str:
        """Preprocess text by handling spaces and special characters"""
        processed = []
        for i, char in enumerate(text):
            if char == " " and i != 0:
                processed.append(self.WORD_SEPARATOR)
            elif char != " ":
                processed.append(char)
        return "".join(processed)

    def postprocess(self, text: str) -> str:
        """Convert processed text back to normal form"""
        return text.replace(self.WORD_SEPARATOR, " ")

class CustomBPETokenizer(TokenizerBase):
    """Custom implementation of BPE tokenizer with improved organization and performance"""
    def __init__(self):
        self.vocab_manager = VocabularyManager()
        self.merge_tracker = BPEMergeTracker()
        self.preprocessor = TextPreprocessor()
        
    def train(self, text: str, vocab_size: int):
        """Train the BPE tokenizer"""
        # Preprocess text
        processed_text = self.preprocessor.preprocess(text)
        
        # Initialize vocabulary with characters
        self._initialize_vocab(processed_text)
        
        # Initialize token sequence
        token_ids = [self.vocab_manager.get_id(char) for char in processed_text]
        
        # Perform BPE training
        self._train_bpe(token_ids, vocab_size)

    def _initialize_vocab(self, text: str):
        """Initialize vocabulary with unique characters and special tokens"""
        # Add ASCII characters
        for i in range(256):
            self.vocab_manager.add_token(chr(i))
            
        # Add characters from text
        for char in set(text):
            self.vocab_manager.add_token(char)
            
        # Add special tokens
        for token in self.preprocessor.special_tokens:
            self.vocab_manager.add_token(token)

    def _train_bpe(self, token_ids: List[int], vocab_size: int):
        """Perform BPE training iterations"""
        while len(self.vocab_manager) < vocab_size:
            self.merge_tracker.update_frequencies(token_ids)
            pair = self.merge_tracker.get_most_frequent_pair()
            
            if not pair:
                break
                
            new_token_id = len(self.vocab_manager)
            self.merge_tracker.add_merge(pair, new_token_id)
            
            # Update token sequence
            token_ids = self._apply_merge(token_ids, pair, new_token_id)
            
            # Add merged token to vocabulary
            merged_token = (self.vocab_manager.get_token(pair[0]) + 
                          self.vocab_manager.get_token(pair[1]))
            self.vocab_manager.add_token(merged_token)

    def _apply_merge(self, token_ids: List[int], pair: Tuple[int, int], 
                    new_id: int) -> List[int]:
        """Apply a merge operation to a sequence of token IDs"""
        result = []
        i = 0
        while i < len(token_ids) - 1:
            if (token_ids[i], token_ids[i + 1]) == pair:
                result.append(new_id)
                i += 2
            else:
                result.append(token_ids[i])
                i += 1
        if i < len(token_ids):
            result.append(token_ids[i])
        return result

    def encode(self, text: str) -> List[int]:
        """Encode text into token IDs"""
        processed_text = self.preprocessor.preprocess(text)
        token_ids = []
        
        # Initial character-level tokenization
        current_token = []
        for char in processed_text:
            if char in self.preprocessor.special_tokens:
                if current_token:
                    token_ids.extend(self._tokenize_substring("".join(current_token)))
                    current_token = []
                token_ids.append(self.vocab_manager.get_id(char))
            else:
                current_token.append(char)
                
        if current_token:
            token_ids.extend(self._tokenize_substring("".join(current_token)))
            
        return token_ids

    def _tokenize_substring(self, text: str) -> List[int]:
        """Tokenize a substring using BPE merges"""
        token_ids = [self.vocab_manager.get_id(char) for char in text]
        
        while len(token_ids) > 1:
            merged = False
            new_token_ids = []
            i = 0
            
            while i < len(token_ids) - 1:
                pair = (token_ids[i], token_ids[i + 1])
                merge_id = self.merge_tracker.get_merge_id(pair)
                
                if merge_id is not None:
                    new_token_ids.append(merge_id)
                    i += 2
                    merged = True
                else:
                    new_token_ids.append(token_ids[i])
                    i += 1
                    
            if i < len(token_ids):
                new_token_ids.append(token_ids[i])
                
            if not merged:
                break
                
            token_ids = new_token_ids
            
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text"""
        tokens = []
        for token_id in token_ids:
            token = self.vocab_manager.get_token(token_id)
            if token in self.preprocessor.special_tokens:
                tokens.append(token)
            else:
                tokens.append(token)
                
        text = "".join(tokens)
        return self.preprocessor.postprocess(text)

    def save(self, vocab_path: str, merges_path: str):
        """Save tokenizer state to files"""
        # Save vocabulary
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab_manager._id_to_token, f, ensure_ascii=False, indent=2)
            
        # Save merges
        with open(merges_path, 'w', encoding='utf-8') as f:
            merges_data = [{'pair': list(pair), 'new_id': new_id}
                          for pair, new_id in self.merge_tracker._merges.items()]
            json.dump(merges_data, f, ensure_ascii=False, indent=2)

    def load(self, vocab_path: str, merges_path: str):
        """Load tokenizer state from files"""
        # Load vocabulary
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
            self.vocab_manager._id_to_token = {int(k): v for k, v in vocab_data.items()}
            self.vocab_manager._token_to_id = {v: int(k) for k, v in vocab_data.items()}
            self.vocab_manager._next_id = max(map(int, vocab_data.keys())) + 1
            
        # Load merges
        with open(merges_path, 'r', encoding='utf-8') as f:
            merges_data = json.load(f)
            self.merge_tracker._merges = {tuple(item['pair']): item['new_id'] 
                                        for item in merges_data}