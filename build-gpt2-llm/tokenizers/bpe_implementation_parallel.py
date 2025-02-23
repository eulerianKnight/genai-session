from typing import List, Dict, Tuple, Optional, Iterator
import multiprocessing as mp
import torch
from torch.nn.parallel import DataParallel
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
import numpy as np
from tqdm import tqdm
import math

@dataclass
class TokenizerConfig:
    """Configuration for parallel tokenizer"""
    vocab_size: int
    batch_size: int = 1024
    num_cpu_workers: int = mp.cpu_count()
    use_gpu: bool = torch.cuda.is_available()
    num_gpu_workers: int = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class ParallelBPETokenizer:
    """BPE Tokenizer with parallel processing support"""
    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.vocab_manager = VocabularyManager()
        self.merge_tracker = BPEMergeTracker()
        self.preprocessor = TextPreprocessor()
        
        # Initialize parallel processing components
        self.cpu_pool = ProcessPoolExecutor(max_workers=config.num_cpu_workers)
        if config.use_gpu:
            self.device = torch.device(config.device)
        else:
            self.device = torch.device("cpu")

    def _create_batches(self, data: List[str], batch_size: int) -> Iterator[List[str]]:
        """Create batches for parallel processing"""
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    def _process_batch_cpu(self, batch: List[str]) -> List[List[int]]:
        """Process a batch of text on CPU"""
        results = []
        for text in batch:
            processed = self.preprocessor.preprocess(text)
            token_ids = self._tokenize_substring(processed)
            results.append(token_ids)
        return results

    def _process_batch_gpu(self, batch: List[str]) -> torch.Tensor:
        """Process a batch of text on GPU"""
        # Convert text to tensor representation
        processed_batch = [self.preprocessor.preprocess(text) for text in batch]
        char_tensors = [torch.tensor([self.vocab_manager.get_id(c) for c in text], 
                                   device=self.device) 
                       for text in processed_batch]
        
        # Pad sequences to same length
        max_len = max(len(t) for t in char_tensors)
        padded_tensors = [torch.nn.functional.pad(t, (0, max_len - len(t))) 
                         for t in char_tensors]
        
        # Stack into batch tensor
        batch_tensor = torch.stack(padded_tensors)
        
        # Apply BPE merges in parallel on GPU
        return self._apply_merges_gpu(batch_tensor)

    def _apply_merges_gpu(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """Apply BPE merges to batch tensor on GPU"""
        # Convert merge rules to tensor format
        merge_pairs = torch.tensor([[pair[0], pair[1]] for pair in self.merge_tracker._merges.keys()], 
                                 device=self.device)
        merge_ids = torch.tensor([new_id for new_id in self.merge_tracker._merges.values()], 
                               device=self.device)
        
        # Kernel for parallel merge operations
        def merge_kernel(tokens: torch.Tensor) -> torch.Tensor:
            while True:
                # Find all possible merge locations
                matches = (tokens[:, :-1].unsqueeze(-1) == merge_pairs[:, 0]) & \
                         (tokens[:, 1:].unsqueeze(-1) == merge_pairs[:, 1])
                
                if not matches.any():
                    break
                
                # Apply first valid merge for each position
                first_matches = matches.float().argmax(dim=-1)
                valid_matches = matches.any(dim=-1)
                
                # Create new token sequence
                new_tokens = tokens.clone()
                mask = valid_matches.nonzero()
                new_tokens[mask] = merge_ids[first_matches[mask]]
                
                # Remove merged tokens
                tokens = torch.cat([new_tokens[:, ::2], new_tokens[:, 1::2]], dim=1)
                
            return tokens
            
        return merge_kernel(batch_tensor)

    def parallel_encode(self, texts: List[str]) -> List[List[int]]:
        """Encode multiple texts in parallel"""
        batches = list(self._create_batches(texts, self.config.batch_size))
        results = []
        
        # Process on GPU if available
        if self.config.use_gpu:
            for batch in tqdm(batches, desc="Processing batches on GPU"):
                batch_results = self._process_batch_gpu(batch)
                results.extend(batch_results.cpu().numpy().tolist())
        
        # Otherwise process on multiple CPU cores
        else:
            futures = [self.cpu_pool.submit(self._process_batch_cpu, batch) 
                      for batch in batches]
            
            for future in tqdm(futures, desc="Processing batches on CPU"):
                batch_results = future.result()
                results.extend(batch_results)
                
        return results

    def parallel_decode(self, token_ids_list: List[List[int]]) -> List[str]:
        """Decode multiple token sequences in parallel"""
        batches = list(self._create_batches(token_ids_list, self.config.batch_size))
        results = []
        
        def decode_batch(batch: List[List[int]]) -> List[str]:
            return [self.decode(token_ids) for token_ids in batch]
        
        # Process in parallel using CPU pool
        futures = [self.cpu_pool.submit(decode_batch, batch) for batch in batches]
        
        for future in tqdm(futures, desc="Decoding batches"):
            batch_results = future.result()
            results.extend(batch_results)
            
        return results

    def train_parallel(self, texts: List[str]):
        """Train tokenizer using parallel processing"""
        # Preprocess texts in parallel
        with ThreadPoolExecutor(max_workers=self.config.num_cpu_workers) as executor:
            processed_texts = list(executor.map(self.preprocessor.preprocess, texts))
        
        # Initialize vocabulary
        self._initialize_vocab_parallel(processed_texts)
        
        # Convert texts to token IDs
        token_ids_list = []
        for text in processed_texts:
            token_ids = [self.vocab_manager.get_id(char) for char in text]
            token_ids_list.append(token_ids)
        
        # Train BPE in parallel
        self._train_bpe_parallel(token_ids_list)

    def _initialize_vocab_parallel(self, texts: List[str]):
        """Initialize vocabulary in parallel"""
        # Collect unique characters in parallel
        def get_unique_chars(text: str) -> set:
            return set(text)
        
        with ThreadPoolExecutor(max_workers=self.config.num_cpu_workers) as executor:
            char_sets = list(executor.map(get_unique_chars, texts))
        
        # Merge character sets
        all_chars = set().union(*char_sets)
        
        # Initialize vocabulary
        for char in all_chars:
            self.vocab_manager.add_token(char)

    def _train_bpe_parallel(self, token_ids_list: List[List[int]]):
        """Train BPE using parallel processing"""
        while len(self.vocab_manager) < self.config.vocab_size:
            # Find pair frequencies in parallel
            def count_pairs(token_ids: List[int]) -> Dict[Tuple[int, int], int]:
                pairs = {}
                for i in range(len(token_ids) - 1):
                    pair = (token_ids[i], token_ids[i + 1])
                    pairs[pair] = pairs.get(pair, 0) + 1
                return pairs
            
            # Process in parallel using CPU pool
            futures = [self.cpu_pool.submit(count_pairs, token_ids) 
                      for token_ids in token_ids_list]
            
            # Merge frequencies
            total_frequencies = {}
            for future in futures:
                frequencies = future.result()
                for pair, count in frequencies.items():
                    total_frequencies[pair] = total_frequencies.get(pair, 0) + count
            
            if not total_frequencies:
                break
            
            # Find most frequent pair
            most_frequent = max(total_frequencies.items(), key=lambda x: x[1])[0]
            
            # Add new merge rule
            new_token_id = len(self.vocab_manager)
            self.merge_tracker.add_merge(most_frequent, new_token_id)
            
            # Apply merge in parallel
            token_ids_list = self.parallel_encode(
                [self.decode(token_ids) for token_ids in token_ids_list]
            )

    def __del__(self):
        """Cleanup parallel processing resources"""
        self.cpu_pool.shutdown()

# Example usage:
def main():
    # Configuration
    config = TokenizerConfig(
        vocab_size=50000,
        batch_size=1024,
        num_cpu_workers=mp.cpu_count(),
        use_gpu=torch.cuda.is_available()
    )
    
    # Initialize tokenizer
    tokenizer = ParallelBPETokenizer(config)
    
    # Example texts
    texts = [
        "Hello world!",
        "How are you?",
        "This is a test.",
        # ... more texts ...
    ]
    
    # Train tokenizer
    tokenizer.train_parallel(texts)
    
    # Encode texts in parallel
    token_ids_list = tokenizer.parallel_encode(texts)
    
    # Decode token IDs in parallel
    decoded_texts = tokenizer.parallel_decode(token_ids_list)
    
    print("Original:", texts[0])
    print("Encoded:", token_ids_list[0])
    print("Decoded:", decoded_texts[0])

if __name__ == "__main__":
    main()