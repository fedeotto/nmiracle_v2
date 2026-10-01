import re
import numpy as np
import torch

SMI_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""

class BasicSmilesTokenizer(object):
    def __init__(self, regex_pattern: str = SMI_REGEX_PATTERN):
        self.regex_pattern = regex_pattern
        self.regex = re.compile(self.regex_pattern)
        
        self.alphabet = None
        self.pad_token_id = None
        self.eos_token_id = None
        self.vocab_size = 0
        self.token_to_id = {}
        self.id_to_token = {}

    def tokenize(self, text):
        """Basic Tokenization of a SMILES.
        """
        if not text:
            return []
        tokens = [token for token in self.regex.findall(text)]
        return tokens
    
    def setup_alphabet(self, alphabet, pad_token_id=None, start_token_id=None, eos_token_id=None):
        """Setup alphabet (vocabulary) for the tokenizer"""
        self.alphabet = alphabet
        
        # Set special token IDs
        self.pad_token_id = pad_token_id if pad_token_id is not None else len(alphabet)
        self.start_token_id = start_token_id if start_token_id is not None else len(alphabet) + 1
        self.eos_token_id = eos_token_id if eos_token_id is not None else len(alphabet) + 2
        self.vocab_size = len(alphabet) + 3  # Add pad, start, and eos tokens
        
        # Create mappings
        self.token_to_id = {token: i for i, token in enumerate(alphabet)}
        self.id_to_token = {i: token for i, token in enumerate(alphabet)}
        
        # Add special tokens to mappings
        self.id_to_token[self.pad_token_id] = "[PAD]"
        self.id_to_token[self.start_token_id] = "[BOS]"  # Beginning of Sequence
        self.id_to_token[self.eos_token_id] = "[EOS]"
    
    def __call__(self, 
                text, 
                max_length=None, 
                padding="max_length", 
                truncation=True, 
                return_tensors=None):
        """
        Make the tokenizer callable like HuggingFace tokenizers
        Args:
            text: Input text to tokenize
            max_length: Maximum length for padding/truncation
            padding: Padding strategy ("max_length" or None)
            truncation: Whether to truncate to max_length
            return_tensors: Return format ("pt" for PyTorch tensors)
            
        Returns:
            TokenizerOutput object with input_ids and attention_mask
        """
        return self.encode(
            text, 
            max_length=max_length, 
            padding=padding, 
            truncation=truncation, 
            return_tensors=return_tensors
        )
    
    def encode(self, 
              text, 
              max_length=None, 
              padding="max_length", 
              truncation=True, 
              return_tensors=None):
        """
        Encode text into token IDs
        
        Args:
            text: Input text to tokenize
            max_length: Maximum length for padding/truncation
            padding: Padding strategy ("max_length" or None)
            truncation: Whether to truncate to max_length
            return_tensors: Return format ("pt" for PyTorch tensors)
            
        Returns:
            TokenizerOutput object with input_ids and attention_mask
        """
        if self.alphabet is None:
            raise ValueError("Tokenizer alphabet not initialized. Call setup_alphabet first.")
            
        # Tokenize
        tokens = self.tokenize(text)
        
        # Convert to IDs, with unknown tokens mapped to pad_token_id
        input_ids = [self.token_to_id.get(token, self.pad_token_id) for token in tokens]
        
        # Add EOS token
        # input_ids.append(self.eos_token_id)
        
        # Truncate if needed
        if truncation and max_length and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(input_ids)
        
        # Pad if requested
        if padding == "max_length" and max_length:
            padding_length = max_length - len(input_ids)
            if padding_length > 0:
                input_ids.extend([self.pad_token_id] * padding_length)
                attention_mask.extend([0] * padding_length)
        
        # Convert to tensors if requested
        if return_tensors == "pt":
            input_ids = torch.tensor([input_ids])
            attention_mask = torch.tensor([attention_mask])
        
        return TokenizerOutput(input_ids=input_ids, attention_mask=attention_mask)
    
    def decode(self, token_ids, skip_special_tokens=True):
        """
        Decode token IDs back to text
        
        Args:
            token_ids: List or tensor of token IDs
            skip_special_tokens: Whether to skip special tokens like padding
            
        Returns:
            Decoded text
        """
        if self.alphabet is None:
            raise ValueError("Tokenizer alphabet not initialized. Call setup_alphabet first.")
        
        # Convert tensor to list if needed
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        # Convert IDs to tokens
        tokens = []
        for token_id in token_ids:
            # Skip special tokens if requested
            if skip_special_tokens and token_id in [self.pad_token_id, self.start_token_id, self.eos_token_id]:
                continue
            
            # Add token if it exists in our mapping
            if token_id in self.id_to_token:
                tokens.append(self.id_to_token[token_id])
        
        # Join tokens into a string (may not be reversible for all SMILES)
        return "".join(tokens)
    
    def __len__(self):
        """Return vocabulary size"""
        if self.alphabet is None:
            return 0
        return self.vocab_size


class TokenizerOutput:
    """
    Simple container for tokenizer outputs to mimic HuggingFace's API
    """
    def __init__(self, input_ids, attention_mask=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask


# def create_alphabet_from_smiles(smiles_list, save_path=None):
#     """
#     Create a vocabulary from a list of SMILES strings
    
#     Args:
#         smiles_list: List of SMILES strings
#         save_path: Optional path to save the alphabet
        
#     Returns:
#         List of unique tokens
#     """
#     tokenizer = BasicSmilesTokenizer()
#     unique_tokens = set()
    
#     for smiles in smiles_list:
#         tokens = tokenizer.tokenize(smiles)
#         unique_tokens.update(tokens)
    
#     # Sort for deterministic ordering
#     alphabet = sorted(list(unique_tokens))
    
#     # Save if requested
#     if save_path:
#         np.save(save_path, np.array(alphabet))
    
#     return alphabet