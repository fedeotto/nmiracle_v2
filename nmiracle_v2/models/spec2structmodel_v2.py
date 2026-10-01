import torch
import torch.nn as nn
from nmiracle_v2.models.components.multispectra_encoder import MultiSpectralEncoder
from nmiracle_v2.models.components.mmspectrum_token_encoder import MMSpectrumTokenEncoder
from nmiracle_v2.utils.eval_utils import canonicalize_smiles, calculate_tanimoto_similarity, is_valid_smiles, compute_levenshtein_distance, calculate_maccs_similarity, calculate_rdkit_similarity
import torch.nn.functional as F
from nmiracle_v2.models.components.transformer_models import TransformerModel

class BaseSpectra2Structure(nn.Module):
    """Base class for spectra-to-structure models"""
    def __init__(
        self,
        tokenizer=None,
        pretrained_structure_path=None,
        structure_model_kwargs=None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size

        if pretrained_structure_path is not None:
            print(f"Loading pretrained structure model from {pretrained_structure_path}")
            checkpoint = torch.load(pretrained_structure_path, map_location='cpu', weights_only=False)
            
            structure_model_kwargs = checkpoint['hyper_parameters']['model']['structure_model']
            self.structure_model = TransformerModel(
                vocab_size=self.vocab_size,
                pad_token=tokenizer.pad_token_id,
                **structure_model_kwargs
            )
            
            # Load pretrained weights by removing the prefix
            if 'state_dict' in checkpoint:
                # Filter and rename keys to match structure model
                structure_model_dict = {}
                for key, value in checkpoint['state_dict'].items():
                    if key.startswith('model.structure_model.'):
                        new_key = key.replace('model.structure_model.', '')
                        structure_model_dict[new_key] = value
            
                # Load the state dict
                missing, unexpected = self.structure_model.load_state_dict(structure_model_dict, strict=True)
                
                if missing:
                    print(f"Warning: {len(missing)} keys are missing from checkpoint:")
                    print(missing[:5], "..." if len(missing) > 5 else "")
                
                if unexpected:
                    print(f"Warning: {len(unexpected)} unexpected keys in checkpoint:")
                    print(unexpected[:5], "..." if len(unexpected) > 5 else "")
                
                print(f"Loaded structure model from structure checkpoint ✅")
        else:
            # Structure prediction model - always needed
            self.structure_model = TransformerModel(
                vocab_size=self.vocab_size,
                pad_token=tokenizer.pad_token_id,
                **structure_model_kwargs
                )

            self.structure_model.initialize_weights()  # Initialize weights for the structure model

    def forward(self, batch):
        pass

    def compute_loss(self, outputs, batch):
        pass

    @torch.no_grad()
    def generate(self, structure_features=None, structure_features_mask=None, max_length=75,
                 temperature=1.0, top_k=5, num_sequences=1, decode_strategy="beam", length_penalty=1.0):
        """
        Generate SMILES strings from substructures.

        Args:
            structure_features / structure_features_mask: encoder outputs to condition on
            max_length: Maximum sequence length
            temperature, top_k: only used when decode_strategy == "sampling"
            num_sequences: number of sequences to return per input (beam width when
                decode_strategy == "beam", number of independent samples when "sampling")
            decode_strategy: "beam" (default) or "sampling"
            length_penalty: only used when decode_strategy == "beam"; divides the final
                cumulative log-prob by length ** length_penalty when ranking beams

        Returns:
            sequences: Generated token sequences [batch_size, num_sequences, max_length]
            scores: Sequence scores [batch_size, num_sequences] (negative log-prob, lower is better)
        """
        if decode_strategy == "beam":
            return self._generate_beam(
                structure_features=structure_features,
                structure_features_mask=structure_features_mask,
                max_length=max_length,
                num_beams=num_sequences,
                length_penalty=length_penalty,
            )
        elif decode_strategy == "sampling":
            return self._generate_sampling(
                structure_features=structure_features,
                structure_features_mask=structure_features_mask,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                num_sequences=num_sequences,
            )
        else:
            raise ValueError(f"Unknown decode_strategy: {decode_strategy!r} (expected 'beam' or 'sampling')")

    @torch.no_grad()
    def _generate_beam(self, structure_features=None, structure_features_mask=None, max_length=75,
                        num_beams=15, length_penalty=1.0):
        """
        Standard batched beam search: at every step, every beam is expanded over
        the full vocabulary and only the top-`num_beams` cumulative-log-prob
        hypotheses survive (as opposed to _generate_sampling, where each of the
        num_sequences hypotheses evolves independently via stochastic top-k sampling).
        Finished beams (EOS already emitted) are frozen by forcing PAD with 0 log-prob
        cost, exactly mirroring _generate_sampling's finished_sequences handling.

        Returns sequences/scores in the same shape/convention as _generate_sampling:
        scores are length-normalized negative log-prob, lower is better.
        """
        if structure_features is None:
            raise ValueError("structure_features must be provided for generation")

        encoder_features = structure_features
        batch_size = encoder_features.size(0)
        device = encoder_features.device
        vocab_size = self.vocab_size
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id

        if structure_features_mask is not None:
            encoder_padding_mask = structure_features_mask
        else:
            encoder_padding_mask = (encoder_features.abs().sum(dim=-1) == 0)
            print("Warning: No padding mask provided, creating one from zero features")

        expanded_batch_size = batch_size * num_beams
        expanded_encoder_features = encoder_features.repeat_interleave(num_beams, dim=0)
        expanded_encoder_padding_mask = (
            encoder_padding_mask.repeat_interleave(num_beams, dim=0)
            if encoder_padding_mask is not None else None
        )

        decoder_input = torch.full(
            (expanded_batch_size, 1), self.tokenizer.start_token_id, dtype=torch.long, device=device
        )

        # Only the first beam of each batch item starts "active" (score 0); the rest
        # start at -inf so that at step 0 the top-k selection picks num_beams DIFFERENT
        # tokens from the single true start-of-sequence distribution, rather than
        # num_beams copies of the single argmax token (all beams are identical at step 0).
        beam_scores = torch.full((batch_size, num_beams), -1e9, dtype=torch.float, device=device)
        beam_scores[:, 0] = 0.0
        beam_scores = beam_scores.view(-1)

        finished = torch.zeros(expanded_batch_size, dtype=torch.bool, device=device)
        lengths = torch.ones(expanded_batch_size, dtype=torch.long, device=device)

        finished_fill = torch.full((vocab_size,), -1e9, device=device)
        finished_fill[pad_id] = 0.0

        for _ in range(max_length - 1):
            logits = self.structure_model(
                expanded_encoder_features, decoder_input, src_key_padding_mask=expanded_encoder_padding_mask
            )
            next_token_logits = logits[:, -1, :]
            log_probs = F.log_softmax(next_token_logits, dim=-1)

            if finished.any():
                log_probs = torch.where(finished.unsqueeze(-1), finished_fill.unsqueeze(0), log_probs)

            cumulative = beam_scores.unsqueeze(-1) + log_probs  # [expanded_batch, vocab]
            cumulative = cumulative.view(batch_size, num_beams * vocab_size)

            topk_scores, topk_indices = torch.topk(cumulative, num_beams, dim=-1)
            beam_idx = torch.div(topk_indices, vocab_size, rounding_mode="floor")  # [batch, num_beams]
            token_idx = topk_indices % vocab_size

            batch_offset = (torch.arange(batch_size, device=device) * num_beams).unsqueeze(-1)
            global_beam_idx = (beam_idx + batch_offset).view(-1)  # [expanded_batch]

            decoder_input = decoder_input[global_beam_idx]
            expanded_encoder_features = expanded_encoder_features[global_beam_idx]
            if expanded_encoder_padding_mask is not None:
                expanded_encoder_padding_mask = expanded_encoder_padding_mask[global_beam_idx]
            finished = finished[global_beam_idx]
            lengths = lengths[global_beam_idx]

            next_token = token_idx.reshape(-1, 1)
            next_token[finished] = pad_id  # paranoia: keep frozen beams literally frozen

            lengths = lengths + (~finished).long()
            decoder_input = torch.cat((decoder_input, next_token), dim=1)
            beam_scores = topk_scores.view(-1)

            eos_mask = (next_token.squeeze(-1) == eos_id) & ~finished
            finished = finished | eos_mask

            if finished.all():
                break

        lengths = torch.clamp(lengths, min=1).float()
        normalized_scores = (-beam_scores) / lengths.pow(length_penalty)
        normalized_scores = normalized_scores.view(batch_size, num_beams)

        if decoder_input.size(1) < max_length:
            padding = torch.full(
                (expanded_batch_size, max_length - decoder_input.size(1)),
                pad_id, dtype=torch.long, device=device,
            )
            decoder_input = torch.cat((decoder_input, padding), dim=1)
        else:
            decoder_input = decoder_input[:, :max_length]

        all_sequences = decoder_input.view(batch_size, num_beams, max_length)

        # topk already sorts by raw cumulative log-prob; re-sort by the length-normalized
        # score (ascending, lower is better) since length normalization can reorder beams.
        sorted_scores, sort_idx = normalized_scores.sort(dim=1, descending=False)
        all_sequences = torch.gather(
            all_sequences, 1, sort_idx.unsqueeze(-1).expand(-1, -1, max_length)
        )
        return all_sequences, sorted_scores

    @torch.no_grad()
    def _generate_sampling(self, structure_features=None, structure_features_mask=None, max_length=75, temperature=1.0, top_k=5, num_sequences=1):
        """
        Generate SMILES strings via independent top-k sampling per sequence
        (each of the num_sequences hypotheses evolves independently and stochastically --
        no shared/pruned beam, unlike _generate_beam).

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
    def generate_smiles(self,
                        structure_features,
                        structure_features_mask=None,
                        true_smiles=None,
                        max_length=75,
                        temperature=1.0,
                        top_k=5,
                        num_sequences=15,
                        max_samples=None,
                        decode_strategy="beam",
                        length_penalty=1.0):
        """
        Wrapper around self.generate() that decodes token ids to SMILES and canonicalizes them.

        Returns dict with:
          - raw_generated: list(batch) of list(num_sequences) raw SMILES strings (decoded)
          - canonical_generated: list(batch) of list(num_sequences) canonical SMILES or None
          - scores: list(batch) of list(num_sequences) scores (neg log prob)
          - canonical_true: list(batch) of canonicalized true SMILES or None (if true_smiles given)
        """
        if structure_features is None:
            raise ValueError("structure_features must be provided")

        # call the core generator
        generated_ids, neg_log_probs = self.generate(
            structure_features=structure_features,
            structure_features_mask=structure_features_mask,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            num_sequences=num_sequences,
            decode_strategy=decode_strategy,
            length_penalty=length_penalty,
        )

        # optionally limit samples
        batch_size = generated_ids.size(0)
        if max_samples is not None:
            batch_size = min(batch_size, int(max_samples))
            generated_ids = generated_ids[:batch_size]
            neg_log_probs = neg_log_probs[:batch_size]
            if true_smiles is not None:
                true_smiles = list(true_smiles)[:batch_size]

        canon_gen_list = []
        scores_list = []

        for i in range(batch_size):
            seqs = generated_ids[i]            # [num_sequences, seq_len]
            scores = neg_log_probs[i]         # [num_sequences]
            # sort by score (lower is better)
            order = torch.argsort(scores).tolist()
            sample_raw = []
            sample_canon = []
            sample_scores = []

            for idx in order:
                seq = seqs[idx]
                # collect token ids up to EOS, skip PAD/START
                token_ids = []
                for t in seq:
                    tid = int(t.item()) if hasattr(t, "item") else int(t)
                    if tid == self.tokenizer.eos_token_id:
                        break
                    if tid in (self.tokenizer.pad_token_id, self.tokenizer.start_token_id):
                        continue
                    token_ids.append(tid)
                # decode
                try:
                    raw_smiles = self.tokenizer.decode(token_ids)
                except Exception:
                    raw_smiles = ""
                # canonicalize
                canon = None
                if raw_smiles:
                    try:
                        c = canonicalize_smiles(raw_smiles)
                        if c and is_valid_smiles(c):
                            canon = c
                    except Exception:
                        canon = None

                sample_raw.append(raw_smiles)
                sample_canon.append(canon)
                sample_scores.append(float(scores[idx].item()))

            canon_gen_list.append(sample_canon)
            scores_list.append(sample_scores)

        canonical_true = None
        if true_smiles is not None:
            canonical_true = []
            for s in true_smiles:
                try:
                    s_str = str(s) if s is not None else ""
                    c = canonicalize_smiles(s_str) if s_str else None
                    canonical_true.append(c if (c and is_valid_smiles(c)) else None)
                except Exception:
                    canonical_true.append(None)

        return {
            "canonical_generated": canon_gen_list,
            "scores": scores_list,
            "canonical_true": canonical_true
        }

    def _decode_generated_ids(self, generated_ids):
        decoded = []
        for seq in generated_ids:
            token_ids = []
            for token in seq:
                token_id = int(token.item()) if hasattr(token, "item") else int(token)
                if token_id == self.tokenizer.eos_token_id:
                    break
                if token_id in (self.tokenizer.pad_token_id, self.tokenizer.start_token_id):
                    continue
                token_ids.append(token_id)
            try:
                decoded.append(self.tokenizer.decode(token_ids) if token_ids else "")
            except Exception:
                decoded.append("")
        return decoded

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
                                   log_prefix='val',
                                   decode_strategy="beam",
                                   length_penalty=1.0):
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
            # 'mces_similarity': 0.0,
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
                num_sequences=num_sequences,
                decode_strategy=decode_strategy,
                length_penalty=length_penalty,
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
            # mces_scores = []
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
                
                # Check if we have at least one valid molecule
                if canon_gen_smiles:
                    valid_count += 1
                    
                    # Calculate similarities and distances for each valid generated molecule
                    best_tanimoto = 0
                    best_rdkit = 0
                    best_maccs = 0
                    best_levenshtein = float('inf')
                    # best_mces = 0
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
                        
                        # Calculate MCES similarity
                        # mces_sim = compute_mces_similarity(canon_gen, canon_true)
                        # if mces_sim > best_mces:
                        #     best_mces = mces_sim
                    
                    # Add best scores to overall metrics
                    tanimoto_scores.append(best_tanimoto)
                    rdkit_scores.append(best_rdkit)
                    maccs_scores.append(best_maccs)
                    levenshtein_distances.append(best_levenshtein)
                    # mces_scores.append(best_mces)
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
            # if mces_scores:
            #     metrics['mces_similarity'] = sum(mces_scores) / len(mces_scores)
            
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


class Spectra2Structure(BaseSpectra2Structure):
    """Spectra to Structure model"""
    def __init__(self,
                 tokenizer=None,
                 pretrained_spectrum_encoder_path=None,
                 pretrained_structure_path=None,
                 freeze_spectrum_encoder=False,
                 multispectra_encoder_kwargs=None,
                 use_tokenized_spectrum_encoder=False,
                 tokenized_spectrum_encoder_kwargs=None,
                 use_formula=False,
                 formula_baked_into_tokens=False,
                 num_elements=9,
                 max_elements_count=35,
                 structure_model_kwargs=None,
                 label_smoothing=0.0):

        super().__init__(
            tokenizer=tokenizer,
            pretrained_structure_path=pretrained_structure_path,
            structure_model_kwargs=structure_model_kwargs
        )

        self.structure_loss_fn = nn.CrossEntropyLoss(
            ignore_index=tokenizer.pad_token_id, label_smoothing=label_smoothing
        )
        # Formula prior: an additional view fused with the spectral tokens
        # before decoding, independent of which spectrum encoder is active --
        # EXCEPT when formula_baked_into_tokens=True (use_alberts_baseline_tokens),
        # where the dataset already includes formula in the token string, so
        # building a separate embedding here would represent it twice.
        self.use_formula = use_formula and not formula_baked_into_tokens
        if self.use_formula:
            structure_dim = self.structure_model.d_model
            self.element_embedding = nn.Embedding(
                num_embeddings=num_elements + 1, embedding_dim=structure_dim, padding_idx=0,
            )
            self.element_count_embedding = nn.Embedding(
                num_embeddings=max_elements_count + 1, embedding_dim=structure_dim, padding_idx=0,
            )
            # Best-effort: a sub2struct checkpoint (loaded above by BaseSpectra2Structure
            # for structure_model) has these same-named, same-mapping embeddings under
            # 'model.element_embedding.'/'model.element_count_embedding.' -- reuse them if present.
            if pretrained_structure_path is not None:
                ckpt = torch.load(pretrained_structure_path, map_location='cpu', weights_only=False)
                state_dict = ckpt.get('state_dict', {})
                for attr_name, prefix in [
                    ('element_embedding', 'model.element_embedding.'),
                    ('element_count_embedding', 'model.element_count_embedding.'),
                ]:
                    sub_state = {
                        k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)
                    }
                    if sub_state:
                        getattr(self, attr_name).load_state_dict(sub_state, strict=True)
                        print(f"Loaded pretrained {attr_name} from sub2struct checkpoint ✅")
                    else:
                        print(f"No {attr_name} found in sub2struct checkpoint -- initialized fresh.")

        # Load and initialize spectrum encoder from alignment checkpoint
        if pretrained_spectrum_encoder_path is not None:
            print(f"Loading pretrained spectrum encoder model from {pretrained_spectrum_encoder_path}")
            checkpoint = torch.load(pretrained_spectrum_encoder_path, map_location='cpu', weights_only=False)
            
            encoder_kwargs = checkpoint['hyper_parameters'].model.multispectra_encoder
            self.spectrum_encoder = MultiSpectralEncoder(**encoder_kwargs)

            # Load pretrained weights
            spectrum_encoder_dict = {}
            for key, value in checkpoint['state_dict'].items():
                if key.startswith('model.spectrum_encoder.'):
                    new_key = key.replace('model.spectrum_encoder.', '')
                    spectrum_encoder_dict[new_key] = value

            # Load the state dict
            missing_spec, unexpected_spec = self.spectrum_encoder.load_state_dict(spectrum_encoder_dict, strict=True)
            if missing_spec:
                print(f"Warning: {len(missing_spec)} keys are missing from checkpoint:")
                print(missing_spec[:5], "..." if len(missing_spec) > 5 else "")
            
            if unexpected_spec:
                print(f"Warning: {len(unexpected_spec)} unexpected keys in checkpoint:")
                print(unexpected_spec[:5], "..." if len(unexpected_spec) > 5 else "")
            else:
                print(f"Loaded spectrum encoder from checkpoint ✅")
        elif use_tokenized_spectrum_encoder:
            print("Initializing tokenized spectrum encoder (see AlbertsDatasetConfig for which ablation).")
            self.spectrum_encoder = MMSpectrumTokenEncoder(**tokenized_spectrum_encoder_kwargs)
        else:
            print("No pretrained alignment model provided, initializing spectrum encoder with default provided parameters.")
            self.spectrum_encoder = MultiSpectralEncoder(**multispectra_encoder_kwargs)

        # Freeze spectrum encoder if specifiedx
        if freeze_spectrum_encoder:
            for param in self.spectrum_encoder.parameters():
                param.requires_grad = False

        # Create adapter if dimensions don't match
        spectrum_dim = self.spectrum_encoder.d_model
        structure_dim = self.structure_model.d_model
        
        if spectrum_dim != structure_dim:
            print(f"Creating adapter from dimension {spectrum_dim} to {structure_dim}")
            self.spectrum_to_structure_adapter = nn.Linear(spectrum_dim, structure_dim)
        else:
            print(f"No adapter needed, dimensions match: {spectrum_dim}")
            self.spectrum_to_structure_adapter = nn.Identity()

    def forward(self, batch):
        outputs = {}
        # Process spectral data
        spectra = batch['spectra']
        spectrum_features, spectrum_mask = self.spectrum_encoder(spectra)

        # Adapt spectrum features to structure space if needed
        structure_features = self.spectrum_to_structure_adapter(spectrum_features)
        structure_features_mask = spectrum_mask

        # Formula prior (optional): fuse as extra tokens the decoder cross-attends
        # to, same pattern as Sub2Structure's multi-view fusion.
        if self.use_formula:
            elements = batch['elements']
            element_counts = batch['element_counts']
            formula_feats = (self.element_embedding(elements)
                            + self.element_count_embedding(element_counts))
            formula_mask = (elements == 0)
            structure_features = torch.cat([structure_features, formula_feats], dim=1)
            structure_features_mask = torch.cat([structure_features_mask, formula_mask], dim=1)

        # Store features for generation later
        outputs['structure_features'] = structure_features
        outputs['structure_features_mask'] = structure_features_mask
        
        # Get decoder inputs for teacher forcing
        decoder_input = batch['decoder_input']
        
        # Forward with teacher forcing using pre-created decoder input
        structure_logits = self.structure_model(
            structure_features,
            decoder_input,
            src_key_padding_mask=structure_features_mask
        )
        outputs['structure_logits'] = structure_logits
        
        return outputs

    def compute_loss(self, outputs, batch):
        total_loss = 0.0
        loss_info = {}

        target = batch['target']
        target = target.view(-1)

        structure_logits = outputs['structure_logits']
        structure_logits = structure_logits.view(-1, structure_logits.size(-1))

        structure_loss = self.structure_loss_fn(structure_logits, target)
        total_loss += structure_loss
        loss_info['structure_loss'] = structure_loss.item()

        return total_loss, loss_info
