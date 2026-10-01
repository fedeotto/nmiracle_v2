from dataclasses import asdict
import torch
from nmiracle_v2.utils.eval_utils import canonicalize_smiles, calculate_tanimoto_similarity, is_valid_smiles, compute_levenshtein_distance, calculate_maccs_similarity, calculate_rdkit_similarity
import torch.nn as nn
from nmiracle_v2.models.components.transformer_models import TransformerModel
from nmiracle_v2.models.components.graph_encoder import GraphEncoder
import torch.nn.functional as F
    
class Sub2Structure(nn.Module):
    def __init__(
        self,
        tokenizer=None,
        structure_kwargs = {},
        pretrained_structure_path=None,
        num_substructures=991,
        max_substructures_count=232,
        use_formula=False,
        use_substructures=True,
        num_elements=9,  # number of unique elements found in the dataset (excluding H)
        max_elements_count=35,  # 35 is the max number of occurrences for an element (found for carbon)
        # Graph encoder options
        use_graph=False,
        gnn_pretrained_path=None,
        gnn_num_layer=5,
        gnn_emb_dim=300,
        gnn_JK="last",
        gnn_drop_ratio=0.0,
        freeze_gnn=True,
    ):
        super().__init__()
        
        # Tokenizer and vocabulary
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size    
        self.use_formula = use_formula
        self.use_substructures = use_substructures
        self.use_graph = use_graph

        #EMBEDDINGS FOR SUBSTRUCTURES.
        if self.use_substructures:
            self.substructure_embedding = nn.Embedding(
                num_embeddings=num_substructures + 1,  # +1 for padding token (0)
                embedding_dim=asdict(structure_kwargs)['d_model'],
                padding_idx=0
            )
            
            self.substructure_count_embedding = nn.Embedding(
                num_embeddings=max_substructures_count + 1,  # +1 for counts capped at max_substructures_len
                embedding_dim=asdict(structure_kwargs)['d_model'],
                padding_idx=0
            )

            self.mlp_proj = nn.Sequential(
                nn.Linear(asdict(structure_kwargs)['d_model'], asdict(structure_kwargs)['d_model']),
                nn.GELU(),
                nn.LayerNorm(asdict(structure_kwargs)['d_model'], eps=asdict(structure_kwargs)['layer_norm_eps']),
            )

        #EMBEDDINGS FOR FORMULA
        # Embedding for element types (C, H, N, O, etc.)
        if self.use_formula:
            self.element_embedding = nn.Embedding(
                num_embeddings=num_elements + 1,  # +1 for padding token (0)
                embedding_dim=asdict(structure_kwargs)['d_model'],
                padding_idx=0
            )
        
            self.element_count_embedding = nn.Embedding(
                num_embeddings=max_elements_count + 1,
                embedding_dim=asdict(structure_kwargs)['d_model'],
                padding_idx=0
            )

        # Optional graph encoder (GraphMVP GNN → projected node tokens)
        if self.use_graph:
            self.graph_encoder = GraphEncoder(
                d_model=asdict(structure_kwargs)['d_model'],
                gnn_num_layer=gnn_num_layer,
                gnn_emb_dim=gnn_emb_dim,
                gnn_JK=gnn_JK,
                gnn_drop_ratio=gnn_drop_ratio,
                pretrained_path=gnn_pretrained_path,
                freeze=freeze_gnn,
            )

        # Structure prediction model - always needed
        self.structure_model = TransformerModel(
            vocab_size=self.vocab_size,
            pad_token=tokenizer.pad_token_id,
            **asdict(structure_kwargs)
        )
        self.structure_model.initialize_weights()  # Initialize weights for the structure model

        # Load pre-trained weights if provided
        if pretrained_structure_path:
            self._load_pretrained(self.structure_model, pretrained_structure_path, prefix="model.structure_model.")

    def _load_pretrained(self, model, path, prefix="model.structure_model."):
        """Load pre-trained weights for a component"""
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Filter keys for the specific model (e.g., structure_model)
            filtered_state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
            
            # Get the current model's state_dict
            model_dict = model.state_dict()
            
            # Filter keys to match the model's structure
            pretrained_dict = {k: v for k, v in filtered_state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            
            # Warn about missing and extra keys
            missing_keys = [k for k in model_dict.keys() if k not in pretrained_dict]
            extra_keys = [k for k in filtered_state_dict.keys() if k not in pretrained_dict]
            if missing_keys:
                print(f"Warning: {len(missing_keys)} keys are missing from the checkpoint")
            if extra_keys:
                print(f"Warning: {len(extra_keys)} extra keys in the checkpoint are ignored")
            
            # Update and load the state dictionary
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            
            # Raise an error if no parameters were loaded
            if len(pretrained_dict) == 0:
                raise ValueError("No matching parameters found in the checkpoint")
            
            print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} parameters from checkpoint")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    
    def encode(self, batch):
        """
        This method is used to extract representations after transformer encoder
        during the sub2struct task.
        """
        # Get inputs
        substructures =  batch['substructures']

        src_key_pad_mask = (substructures == 0)  # Shape: [batch_size, seq_len]
        structure_features = self.substructure_embedding(substructures)  # [batch_size, seq_len, d_model]

        device_type = 'cuda' if structure_features.is_cuda else 'cpu'

        with torch.autocast(device_type=device_type, enabled=False):
            encoded_features = self.structure_model.transformer.encoder(
                structure_features,
                src_key_padding_mask=src_key_pad_mask
            )

        return encoded_features, src_key_pad_mask


    def forward(self, batch):
        """
        Forward pass through the model.

        batch keys:
          - 'substructures'       : [B, S]  long (padded with 0)
          - 'substructure_counts' : [B, S]  long
          - 'decoder_input'       : [B, L]  long
          - 'target'              : [B, L]  long
          - 'elements'            : [B, F]  long  (if use_formula)
          - 'element_counts'      : [B, F]  long  (if use_formula)
          - 'graph'               : PyG Batch      (if use_graph)
        """
        results = {}
        views = []
        if self.use_substructures:
            substructures = batch['substructures']
            substructure_counts = batch['substructure_counts']

            # ── 1. Substructure tokens ────────────────────────────────────────────
            sub_mask = (substructures == 0)                                  # [B, S]
            sub_feats = (self.substructure_embedding(substructures)
                        + self.substructure_count_embedding(substructure_counts))  # [B, S, d]
            sub_feats = self.mlp_proj(sub_feats)

            # Collect (tokens, mask) views for Q-Former or plain concat
            views.append((sub_feats, sub_mask))

        # ── 2. Formula tokens (optional) ─────────────────────────────────────
        if self.use_formula:
            elements      = batch['elements']
            element_counts = batch['element_counts']
            formula_feats  = (self.element_embedding(elements)
                              + self.element_count_embedding(element_counts))    # [B, F, d]
            formula_mask   = (elements == 0)                                     # [B, F]
            views.append((formula_feats, formula_mask))

        # ── 3. Graph node tokens (optional) ──────────────────────────────────
        if self.use_graph and 'graph' in batch:
            node_tokens, node_mask = self.graph_encoder(batch['graph'])          # [B, N, d]
            views.append((node_tokens, node_mask))

        if not views:
            raise ValueError(
                "Sub2Structure needs at least one active input view. "
                "Enable use_substructures, use_formula, or use_graph."
            )

        # ── 4. Aggregate views → encoder input
        encoder_input = torch.cat([v[0] for v in views], dim=1)             # [B, total, d]
        encoder_mask  = torch.cat([v[1] for v in views], dim=1)             # [B, total]

        results['structure_features']      = encoder_input
        results['structure_features_mask'] = encoder_mask

        # ── 5. Decode SMILES (teacher forcing)
        decoder_input = batch['decoder_input']
        structure_logits = self.structure_model(
            encoder_input,
            decoder_input,
            src_key_padding_mask=encoder_mask,
        )
        results['structure_logits'] = structure_logits

        return results


    @torch.no_grad()
    def generate(self, structure_features=None, structure_features_mask=None, max_length=75, temperature=1.0, top_k=5, num_sequences=1):
        """
        Generate SMILES strings from substructures
        
        Args:
            substructures: Binary substructure vectors [batch_size, num_substructures]
            tokenizer: Tokenizer for decoding
            max_length: Maximum sequence length
            temperature: Temperature for sampling
            top_k: Number of top tokens to sample from
            num_sequences: Number of sequences to generate per input
            
        Returns:
            sequences: Generated token sequences [batch_size, num_sequences, max_length]
            scores: Sequence scores [batch_size, num_sequences]
        """
        # Run encoder only once
        if structure_features is None:
            raise ValueError("structure_features must be provided for generation")
        
        # Use structure_features as encoder features
        encoder_features = structure_features
        batch_size = encoder_features.size(0)
        device = encoder_features.device
        
        # Use provided padding mask or create one if not provided
        if structure_features_mask is not None:
            encoder_padding_mask = structure_features_mask
        else:
            # Fallback: create encoder padding mask - assuming padding where all features are zero
            # Shape: [batch_size, seq_len] - True for padding positions
            encoder_padding_mask = (encoder_features.abs().sum(dim=-1) == 0)
            print("Warning: No padding mask provided, creating one from zero features")
        
        # Create expanded batch for parallel generation
        expanded_encoder_features = encoder_features.repeat_interleave(num_sequences, dim=0)
        expanded_batch_size = batch_size * num_sequences
        
        if encoder_padding_mask is not None:
            expanded_encoder_padding_mask = encoder_padding_mask.repeat_interleave(num_sequences, dim=0)
        else:
            expanded_encoder_padding_mask = None
        
        # Initialize output with start token
        decoder_input = torch.full(
            (expanded_batch_size, 1), 
            self.tokenizer.start_token_id, 
            dtype=torch.long, 
            device=device
        )
        
        # Track sequence probabilities and lengths
        sequence_neg_log_probs = torch.zeros(expanded_batch_size, dtype=torch.float, device=device)
        sequence_lengths = torch.zeros(expanded_batch_size, dtype=torch.long, device=device)

        # Track which sequences are finished
        finished_sequences = torch.zeros(expanded_batch_size, dtype=torch.bool, device=device)

        # Generate tokens one by one
        for _ in range(max_length - 1):
            # Forward pass with current decoder input
            logits = self.structure_model(
                expanded_encoder_features,
                decoder_input,
                src_key_padding_mask=expanded_encoder_padding_mask
            )
            
            # Get next token logits (last position)
            next_token_logits = logits[:, -1, :]

            # Apply temperature first
            next_token_logits = next_token_logits / temperature

            # Apply softmax to ALL logits first
            full_probs = F.softmax(next_token_logits, dim=-1)

            # Then do top-k on probabilities
            top_k_probs, top_k_indices = torch.topk(full_probs, top_k, dim=-1)

            # Renormalize
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
            
            # Sample from the top-k distribution
            next_token_idx = torch.multinomial(top_k_probs, 1)
            
            # Get probability of selected token
            selected_probs = torch.gather(top_k_probs, -1, next_token_idx)
            
            # Get actual token ID
            next_token = torch.gather(top_k_indices, -1, next_token_idx)

            # For finished sequences, force padding tokens
            next_token[finished_sequences] = self.tokenizer.pad_token_id
            selected_probs[finished_sequences] = 1.0  # Don't penalize padding tokens

            # Update sequence probabilities (for non-finished sequences)
            not_finished = ~finished_sequences

            if not_finished.any():
                log_probs = torch.log(selected_probs[not_finished].view(-1) + 1e-8)
                log_probs = torch.clamp(log_probs, min=-100.0)
                sequence_neg_log_probs[not_finished] -= log_probs
            
            # Update sequence lengths
            sequence_lengths[not_finished & (next_token.squeeze(-1) != self.tokenizer.eos_token_id)] += 1
            
            # Append token to sequences
            decoder_input = torch.cat((decoder_input, next_token), dim=1)

            # Mark sequences as finished if they generated EOS
            eos_mask = (next_token.squeeze(-1) == self.tokenizer.eos_token_id) & ~finished_sequences
            finished_sequences = finished_sequences | eos_mask

            # Check if all sequences are finished
            if finished_sequences.all():
                break
        
        # Normalize scores by length
        sequence_lengths = torch.clamp(sequence_lengths, min=1)
        normalized_scores = sequence_neg_log_probs / sequence_lengths.float()
        
        # Pad sequences to max_length if needed
        if decoder_input.size(1) < max_length:
            padding = torch.full(
                (expanded_batch_size, max_length - decoder_input.size(1)),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
                device=device
            )
            decoder_input = torch.cat((decoder_input, padding), dim=1)
        else:
            decoder_input = decoder_input[:, :max_length]
        
        # Reshape output
        all_sequences = decoder_input.view(batch_size, num_sequences, max_length)
        all_scores = normalized_scores.view(batch_size, num_sequences)
        
        return all_sequences, all_scores

    @torch.no_grad()
    def compute_generation_metrics(self, 
                                   structure_features, 
                                   structure_features_mask, 
                                   true_smiles, 
                                   temperature=1.0,
                                   top_k=5,
                                   num_sequences=15,
                                   max_samples=None,
                                   max_molecule_len=45, 
                                   log_prefix='val'):
        """Comprehensive evaluation of generated molecules with multiple metrics"""        
        # Handle empty case
        if not true_smiles:
            print("No molecules found")
            return {}
            
        # Take a subset to save time (first few examples)
        batch_size = len(true_smiles)
        
        if max_samples is None:
            sample_size = batch_size
        else:
            sample_size = min(max_samples, batch_size)
        
        sample_indices = list(range(sample_size))
        
        # Sample the structure_features and true_smiles to match the sample size
        sampled_structure_features = structure_features[sample_indices] if structure_features is not None else None
        sampled_structure_features_mask = structure_features_mask[sample_indices] if structure_features_mask is not None else None

        sample_true_smiles = [str(true_smiles[i]) for i in sample_indices]

        # Initialize metrics storage
        metrics = {
            'valid_rate': 0.0,
            'tanimoto_similarity': 0.0,
            'rdkit_similarity': 0.0,
            'maccs_similarity': 0.0,
            'exact_match_rate': 0.0,
            'levenshtein_distance': 0.0,
            'top_1_accuracy': 0.0,
            'top_3_accuracy': 0.0,
            'top_5_accuracy': 0.0,
            'top_10_accuracy': 0.0,
            'top_15_accuracy': 0.0
        }
        
        try:
            # Generate molecules using the model's generate method with precomputed features
            generated_ids, neg_log_probs = self.generate(
                structure_features=sampled_structure_features,
                structure_features_mask= sampled_structure_features_mask,
                max_length=max_molecule_len,  # Adjust as needed
                temperature=temperature,
                top_k=top_k,
                num_sequences=num_sequences
            )
            
            # Reshape generated_ids into [batch_size, num_sequences, seq_len]
            batch_size, num_seqs, seq_len = generated_ids.shape
            
            # Decode all generated sequences for each sample
            generated_smiles_lists = []
            generated_scores_lists = []  # Store the scores (negative log probs)

            for i in range(batch_size):
                sample_ids = generated_ids[i]
                sample_scores = neg_log_probs[i]
                decoded_smiles = []
                
                for j, seq_ids in enumerate(sample_ids):
                    # Extract tokens until EOS or end of sequence
                    tokens = []
                    for token in seq_ids:
                        if token.item() == self.tokenizer.eos_token_id:
                            break
                        if token.item() in [self.tokenizer.pad_token_id, self.tokenizer.start_token_id]:
                            continue
                        tokens.append(token.item())
                    # Decode to SMILES
                    try:
                        smiles = self.tokenizer.decode(tokens)
                        decoded_smiles.append((smiles, sample_scores[j].item()))  # Store (SMILES, score) tuples
                    except Exception:
                        decoded_smiles.append(("", float('inf')))  # Use infinity for invalid SMILES
                
                # Sort by score (lower is better for negative log probs)
                sorted_smiles = sorted(decoded_smiles, key=lambda x: x[1])
                
                # Store only the SMILES strings, keeping the sorted order
                generated_smiles_lists.append([s[0] for s in sorted_smiles])
                # Store the scores separately
                generated_scores_lists.append([s[1] for s in sorted_smiles])
            
            # Calculate various metrics
            valid_count = 0
            tanimoto_scores = []
            rdkit_scores = []
            maccs_scores = []
            exact_match_count = 0
            levenshtein_distances = []
            top_1_hits = 0
            top_3_hits = 0
            top_5_hits = 0
            top_10_hits = 0
            top_15_hits = 0
            
            # Store top-1 generated molecules for visualization
            visualization_generated_smiles = []
            visualization_true_smiles = []
            
            # If testing, store all the generated vs true molecules
            if log_prefix == "test":
                all_canon_generated = []
                all_canon_true = []
            
            for i, (gen_smiles_list, true_smiles) in enumerate(zip(generated_smiles_lists, sample_true_smiles)):
                # Canonicalize all generated SMILES and filter invalid ones
                canon_gen_smiles = []
                
                for gen_smiles in gen_smiles_list:
                    canon_gen = canonicalize_smiles(gen_smiles)
                    if canon_gen and is_valid_smiles(canon_gen):
                        canon_gen_smiles.append(canon_gen)
                
                # Get canonical version of true SMILES
                canon_true = canonicalize_smiles(true_smiles)
                
                # Store first generated SMILES for visualization
                if gen_smiles_list:
                    visualization_generated_smiles.append(gen_smiles_list[0])
                    visualization_true_smiles.append(true_smiles)
                
                # If testing, store all generated SMILES
                if log_prefix == "test":
                    all_canon_true.append(canon_true)
                    all_canon_generated.append(canon_gen_smiles)
                
                # Check if we have any valid molecules
                if canon_gen_smiles:
                    valid_count += 1
                    
                    # Calculate similarities and distances for each valid generated molecule
                    best_tanimoto = 0
                    best_rdkit = 0
                    best_maccs = 0
                    best_levenshtein = float('inf')
                    exact_match = False
                    
                    for j, canon_gen in enumerate(canon_gen_smiles):
                        # Check for exact match
                        if canon_gen == canon_true:
                            exact_match = True
                            
                        # Calculate Tanimoto similarity (Morgan fingerprints)
                        tanimoto = calculate_tanimoto_similarity(canon_gen, canon_true)
                        if tanimoto > best_tanimoto:
                            best_tanimoto = tanimoto
                        
                        # Calculate RDKit fingerprint similarity
                        rdkit_sim = calculate_rdkit_similarity(canon_gen, canon_true)
                        if rdkit_sim > best_rdkit:
                            best_rdkit = rdkit_sim
                        
                        # Calculate MACCS fingerprint similarity
                        maccs_sim = calculate_maccs_similarity(canon_gen, canon_true)
                        if maccs_sim > best_maccs:
                            best_maccs = maccs_sim
                        
                        # Calculate Levenshtein distance
                        levenshtein = compute_levenshtein_distance(canon_gen, canon_true)
                        if levenshtein < best_levenshtein:
                            best_levenshtein = levenshtein
                    
                    # Add best scores to overall metrics
                    tanimoto_scores.append(best_tanimoto)
                    rdkit_scores.append(best_rdkit)
                    maccs_scores.append(best_maccs)
                    levenshtein_distances.append(best_levenshtein)
                    if exact_match:
                        exact_match_count += 1
                    
                    # Calculate top-k accuracy
                    if canon_true in canon_gen_smiles[:1]:
                        top_1_hits += 1
                    if canon_true in canon_gen_smiles[:3]:
                        top_3_hits += 1
                    if canon_true in canon_gen_smiles[:5]:
                        top_5_hits += 1
                    if canon_true in canon_gen_smiles[:10]:
                        top_10_hits += 1
                    if canon_true in canon_gen_smiles[:15]:
                        top_15_hits += 1
            
            # Calculate overall metrics
            metrics['valid_rate'] = valid_count / sample_size
            metrics['exact_match_rate'] = exact_match_count / sample_size
            metrics['top_1_accuracy'] = top_1_hits / sample_size
            metrics['top_3_accuracy'] = top_3_hits / sample_size
            metrics['top_5_accuracy'] = top_5_hits / sample_size
            metrics['top_10_accuracy'] = top_10_hits / sample_size
            metrics['top_15_accuracy'] = top_15_hits / sample_size
            
            # Average metrics across valid molecules
            if tanimoto_scores:
                metrics['tanimoto_similarity'] = sum(tanimoto_scores) / len(tanimoto_scores)
            if rdkit_scores:
                metrics['rdkit_similarity'] = sum(rdkit_scores) / len(rdkit_scores)
            if maccs_scores:
                metrics['maccs_similarity'] = sum(maccs_scores) / len(maccs_scores)
            if levenshtein_distances:
                metrics['levenshtein_distance'] = sum(levenshtein_distances) / len(levenshtein_distances)
            
            # Add test data for later analysis
            if log_prefix == "test": #and hasattr(self, 'test_results'):
                metrics["all_canon_generated"] = all_canon_generated
                metrics["all_canon_true"] = all_canon_true
                
            return metrics
        
        except Exception as e:
            print(f"Error during generation evaluation: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e
