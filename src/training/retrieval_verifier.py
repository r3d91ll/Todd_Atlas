"""
Retrieval Verifier - Verifies that stored content is actually retrieved.

Critical component for episodic memory training. Records what was stored
during storage phase and verifies retrieval during retrieval phase.
"""

import time
import hashlib
from typing import Dict, Any, Optional, Tuple
from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class StorageRecord:
    """Record of what was stored during a storage phase."""
    targets: torch.Tensor  # Target tokens
    memory_snapshot: torch.Tensor  # Memory state at storage time
    timestamp: float
    batch_hash: str


class RetrievalVerifier:
    """
    Verifies that episodic memory storage and retrieval are working.

    During storage phase:
    - Records target tokens and memory state

    During retrieval phase:
    - Compares model output to stored targets
    - Computes retrieval accuracy metrics
    - Applies heavy penalty for retrieval failure
    """

    def __init__(
        self,
        max_buffer_size: int = 100,
        retrieval_loss_weight: float = 5.0,
        device: str = 'cuda',
    ):
        """
        Args:
            max_buffer_size: Maximum number of storage records to keep
            retrieval_loss_weight: Weight for retrieval loss (heavy penalty)
            device: Device for tensor operations
        """
        self.max_buffer_size = max_buffer_size
        self.retrieval_loss_weight = retrieval_loss_weight
        self.device = device

        # Storage buffer: batch_hash -> StorageRecord
        # Using OrderedDict to maintain insertion order for LRU eviction
        self._storage_buffer: OrderedDict[str, StorageRecord] = OrderedDict()

        # Statistics
        self._total_storage_calls = 0
        self._total_retrieval_calls = 0
        self._successful_retrievals = 0

    def compute_batch_hash(self, batch: Dict[str, torch.Tensor]) -> str:
        """
        Compute a hash for a batch to match storage and retrieval.

        Uses input_ids to create a unique identifier.
        """
        input_ids = batch.get('input_ids', batch.get('inputs'))
        if input_ids is None:
            # Fallback to timestamp-based hash
            return hashlib.md5(str(time.time()).encode()).hexdigest()[:16]

        # Use first few tokens as hash (efficient)
        tokens = input_ids.flatten()[:100].cpu().numpy().tobytes()
        return hashlib.md5(tokens).hexdigest()[:16]

    def record_storage(
        self,
        batch_hash: str,
        target_tokens: torch.Tensor,
        memory_state: torch.Tensor,
    ) -> None:
        """
        Record what was stored during storage phase.

        Args:
            batch_hash: Unique identifier for this batch
            target_tokens: Target tokens that should be retrievable
            memory_state: Current memory state snapshot
        """
        self._total_storage_calls += 1

        # Evict oldest if buffer full
        while len(self._storage_buffer) >= self.max_buffer_size:
            self._storage_buffer.popitem(last=False)

        # Store record
        self._storage_buffer[batch_hash] = StorageRecord(
            targets=target_tokens.detach().clone(),
            memory_snapshot=memory_state.detach().clone(),
            timestamp=time.time(),
            batch_hash=batch_hash,
        )

    def get_stored_record(self, batch_hash: str) -> Optional[StorageRecord]:
        """Get a stored record by batch hash."""
        return self._storage_buffer.get(batch_hash)

    def verify_retrieval(
        self,
        batch_hash: str,
        model_logits: torch.Tensor,
        current_memory: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Verify retrieval from stored content.

        Args:
            batch_hash: Batch identifier (must match a storage record)
            model_logits: Model output logits [batch, seq, vocab]
            current_memory: Current memory state

        Returns:
            Dictionary of retrieval metrics
        """
        self._total_retrieval_calls += 1

        stored = self._storage_buffer.get(batch_hash)
        if stored is None:
            return {
                'retrieval_error': 1.0,
                'retrieval_token_accuracy': 0.0,
                'retrieval_exact_match': 0.0,
                'memory_retention_similarity': 0.0,
            }

        stored_targets = stored.targets.to(model_logits.device)
        stored_memory = stored.memory_snapshot.to(current_memory.device)

        # Get predictions from logits
        predictions = model_logits.argmax(dim=-1)

        # Ensure shapes match
        min_len = min(predictions.size(-1), stored_targets.size(-1))
        predictions = predictions[..., :min_len]
        stored_targets = stored_targets[..., :min_len]

        # Token-level accuracy
        token_matches = (predictions == stored_targets).float()
        token_accuracy = token_matches.mean().item()

        # Exact match rate (entire sequence matches)
        if predictions.dim() > 1:
            exact_matches = token_matches.all(dim=-1).float()
            exact_match_rate = exact_matches.mean().item()
        else:
            exact_match_rate = 1.0 if token_accuracy == 1.0 else 0.0

        # Memory retention similarity
        # Flatten and compute cosine similarity
        stored_flat = stored_memory.flatten().float()
        current_flat = current_memory.flatten().float()

        # Ensure same size
        min_size = min(stored_flat.size(0), current_flat.size(0))
        stored_flat = stored_flat[:min_size]
        current_flat = current_flat[:min_size]

        memory_similarity = F.cosine_similarity(
            stored_flat.unsqueeze(0),
            current_flat.unsqueeze(0),
            dim=1
        ).item()

        # Track successful retrievals
        if token_accuracy > 0.5:
            self._successful_retrievals += 1

        return {
            'retrieval_error': 0.0,
            'retrieval_token_accuracy': token_accuracy,
            'retrieval_exact_match': exact_match_rate,
            'memory_retention_similarity': memory_similarity,
            'retrieval_time_delta': time.time() - stored.timestamp,
        }

    def compute_retrieval_loss(
        self,
        model_logits: torch.Tensor,
        stored_targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute retrieval loss - heavy penalty for retrieval failure.

        Args:
            model_logits: Model output logits [batch, seq, vocab]
            stored_targets: Target tokens from storage [batch, seq]

        Returns:
            Weighted cross-entropy loss
        """
        # Flatten for cross entropy
        logits_flat = model_logits.view(-1, model_logits.size(-1))

        # Ensure targets match logits sequence length
        min_len = min(model_logits.size(1), stored_targets.size(-1))
        targets_flat = stored_targets[..., :min_len].contiguous().view(-1)
        logits_flat = model_logits[:, :min_len, :].contiguous().view(-1, model_logits.size(-1))

        # Cross entropy loss with heavy weight
        loss = F.cross_entropy(logits_flat, targets_flat)

        return loss * self.retrieval_loss_weight

    def compute_retrieval_loss_from_hash(
        self,
        batch_hash: str,
        model_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute retrieval loss for a batch hash.

        Args:
            batch_hash: Batch identifier
            model_logits: Model output logits

        Returns:
            Tuple of (loss tensor, verification metrics)
        """
        stored = self._storage_buffer.get(batch_hash)
        if stored is None:
            # No stored content - return zero loss
            return (
                torch.tensor(0.0, device=model_logits.device),
                {'retrieval_error': 1.0}
            )

        stored_targets = stored.targets.to(model_logits.device)
        loss = self.compute_retrieval_loss(model_logits, stored_targets)

        # Also compute verification metrics
        # (Use dummy memory state since we don't have access here)
        dummy_memory = stored.memory_snapshot.to(model_logits.device)
        metrics = self.verify_retrieval(batch_hash, model_logits, dummy_memory)

        return loss, metrics

    def get_statistics(self) -> Dict[str, Any]:
        """Get verifier statistics."""
        return {
            'total_storage_calls': self._total_storage_calls,
            'total_retrieval_calls': self._total_retrieval_calls,
            'successful_retrievals': self._successful_retrievals,
            'success_rate': (
                self._successful_retrievals / max(self._total_retrieval_calls, 1)
            ),
            'buffer_size': len(self._storage_buffer),
            'buffer_capacity': self.max_buffer_size,
        }

    def clear(self) -> None:
        """Clear storage buffer and reset statistics."""
        self._storage_buffer.clear()
        self._total_storage_calls = 0
        self._total_retrieval_calls = 0
        self._successful_retrievals = 0

    def flush_old_records(self, max_age_seconds: float = 300) -> int:
        """
        Remove storage records older than max_age_seconds.

        Returns number of records removed.
        """
        current_time = time.time()
        to_remove = []

        for hash_key, record in self._storage_buffer.items():
            if current_time - record.timestamp > max_age_seconds:
                to_remove.append(hash_key)

        for key in to_remove:
            del self._storage_buffer[key]

        return len(to_remove)
