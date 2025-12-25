"""
RNN Models for Patient Outcome Prediction.

Provides LSTM, GRU, and BiLSTM architectures for sequence classification.
"""

from typing import Optional, Tuple, Dict, Any

try:
    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .config import Config, get_config, ModelConfig


if HAS_TORCH:

    class PatientRNN(nn.Module):
        """
        Base RNN model for patient outcome prediction.

        Supports LSTM, GRU, and bidirectional variants.
        """

        def __init__(
            self,
            vocab_size: int,
            embedding_dim: int = 128,
            hidden_size: int = 256,
            num_layers: int = 2,
            dropout: float = 0.3,
            rnn_type: str = 'lstm',
            bidirectional: bool = False,
            num_classes: int = 1,
            pad_idx: int = 0
        ):
            """
            Initialize the model.

            Args:
                vocab_size: Size of the event vocabulary
                embedding_dim: Dimension of event embeddings
                hidden_size: RNN hidden state size
                num_layers: Number of RNN layers
                dropout: Dropout probability
                rnn_type: 'lstm' or 'gru'
                bidirectional: Use bidirectional RNN
                num_classes: Number of output classes (1 for binary)
                pad_idx: Padding token index
            """
            super().__init__()

            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.bidirectional = bidirectional
            self.num_directions = 2 if bidirectional else 1
            self.rnn_type = rnn_type.lower()

            # Embedding layer
            self.embedding = nn.Embedding(
                vocab_size,
                embedding_dim,
                padding_idx=pad_idx
            )

            # RNN layer
            rnn_class = nn.LSTM if self.rnn_type == 'lstm' else nn.GRU
            self.rnn = rnn_class(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )

            # Dropout
            self.dropout = nn.Dropout(dropout)

            # Fully connected layers
            fc_input_size = hidden_size * self.num_directions
            self.fc1 = nn.Linear(fc_input_size, hidden_size // 2)
            self.fc2 = nn.Linear(hidden_size // 2, num_classes)

            # Activation
            self.relu = nn.ReLU()

        def forward(
            self,
            sequences: torch.Tensor,
            lengths: torch.Tensor
        ) -> torch.Tensor:
            """
            Forward pass.

            Args:
                sequences: Padded sequences [batch, max_len]
                lengths: Original sequence lengths [batch]

            Returns:
                Output logits [batch, num_classes]
            """
            batch_size = sequences.size(0)

            # Embed sequences
            embedded = self.embedding(sequences)  # [batch, max_len, embed_dim]
            embedded = self.dropout(embedded)

            # Sort by length for packing (required by pack_padded_sequence)
            lengths_cpu = lengths.cpu()
            sorted_lengths, sort_idx = lengths_cpu.sort(descending=True)
            sorted_embedded = embedded[sort_idx]

            # Pack sequences
            packed = pack_padded_sequence(
                sorted_embedded,
                sorted_lengths.tolist(),
                batch_first=True
            )

            # Run RNN
            if self.rnn_type == 'lstm':
                packed_output, (hidden, cell) = self.rnn(packed)
            else:
                packed_output, hidden = self.rnn(packed)

            # Get final hidden state
            # hidden: [num_layers * num_directions, batch, hidden_size]
            if self.bidirectional:
                # Concatenate forward and backward final hidden states
                hidden_fwd = hidden[-2, :, :]  # Last layer forward
                hidden_bwd = hidden[-1, :, :]  # Last layer backward
                final_hidden = torch.cat([hidden_fwd, hidden_bwd], dim=1)
            else:
                final_hidden = hidden[-1, :, :]  # Last layer

            # Unsort to original order
            _, unsort_idx = sort_idx.sort()
            final_hidden = final_hidden[unsort_idx]

            # Fully connected layers
            out = self.dropout(final_hidden)
            out = self.relu(self.fc1(out))
            out = self.dropout(out)
            out = self.fc2(out)

            return out

        def predict_proba(
            self,
            sequences: torch.Tensor,
            lengths: torch.Tensor
        ) -> torch.Tensor:
            """
            Get probability predictions.

            Args:
                sequences: Padded sequences [batch, max_len]
                lengths: Original sequence lengths [batch]

            Returns:
                Probabilities [batch] (for binary classification)
            """
            logits = self.forward(sequences, lengths)
            return torch.sigmoid(logits).squeeze(-1)

        @classmethod
        def from_config(
            cls,
            vocab_size: int,
            config: Optional[ModelConfig] = None,
            pad_idx: int = 0
        ) -> 'PatientRNN':
            """
            Create model from configuration.

            Args:
                vocab_size: Vocabulary size
                config: Model configuration
                pad_idx: Padding index

            Returns:
                PatientRNN model
            """
            if config is None:
                config = get_config().model

            return cls(
                vocab_size=vocab_size,
                embedding_dim=config.embedding_dim,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                dropout=config.dropout,
                rnn_type=config.type,
                bidirectional=config.bidirectional,
                pad_idx=pad_idx
            )


    class PatientLSTM(PatientRNN):
        """LSTM model for patient outcome prediction."""

        def __init__(self, vocab_size: int, **kwargs):
            super().__init__(vocab_size, rnn_type='lstm', **kwargs)


    class PatientGRU(PatientRNN):
        """GRU model for patient outcome prediction."""

        def __init__(self, vocab_size: int, **kwargs):
            super().__init__(vocab_size, rnn_type='gru', **kwargs)


    class PatientBiLSTM(PatientRNN):
        """Bidirectional LSTM model for patient outcome prediction."""

        def __init__(self, vocab_size: int, **kwargs):
            super().__init__(vocab_size, rnn_type='lstm', bidirectional=True, **kwargs)


    class AttentionPatientRNN(nn.Module):
        """
        RNN with attention mechanism for patient outcome prediction.

        Attention helps identify which events are most predictive.
        """

        def __init__(
            self,
            vocab_size: int,
            embedding_dim: int = 128,
            hidden_size: int = 256,
            num_layers: int = 2,
            dropout: float = 0.3,
            rnn_type: str = 'lstm',
            bidirectional: bool = False,
            num_classes: int = 1,
            pad_idx: int = 0
        ):
            super().__init__()

            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.bidirectional = bidirectional
            self.num_directions = 2 if bidirectional else 1
            self.rnn_type = rnn_type.lower()

            # Embedding
            self.embedding = nn.Embedding(
                vocab_size,
                embedding_dim,
                padding_idx=pad_idx
            )

            # RNN
            rnn_class = nn.LSTM if self.rnn_type == 'lstm' else nn.GRU
            self.rnn = rnn_class(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )

            # Attention
            rnn_output_size = hidden_size * self.num_directions
            self.attention = nn.Sequential(
                nn.Linear(rnn_output_size, hidden_size // 2),
                nn.Tanh(),
                nn.Linear(hidden_size // 2, 1)
            )

            # Output layers
            self.dropout = nn.Dropout(dropout)
            self.fc1 = nn.Linear(rnn_output_size, hidden_size // 2)
            self.fc2 = nn.Linear(hidden_size // 2, num_classes)
            self.relu = nn.ReLU()

        def forward(
            self,
            sequences: torch.Tensor,
            lengths: torch.Tensor
        ) -> torch.Tensor:
            """
            Forward pass with attention.

            Args:
                sequences: Padded sequences [batch, max_len]
                lengths: Original sequence lengths [batch]

            Returns:
                Output logits [batch, num_classes]
            """
            batch_size, max_len = sequences.size()

            # Embed
            embedded = self.embedding(sequences)
            embedded = self.dropout(embedded)

            # Sort by length
            lengths_cpu = lengths.cpu()
            sorted_lengths, sort_idx = lengths_cpu.sort(descending=True)
            sorted_embedded = embedded[sort_idx]

            # Pack and run RNN
            packed = pack_padded_sequence(
                sorted_embedded,
                sorted_lengths.tolist(),
                batch_first=True
            )

            if self.rnn_type == 'lstm':
                packed_output, _ = self.rnn(packed)
            else:
                packed_output, _ = self.rnn(packed)

            # Unpack
            rnn_output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)

            # Unsort
            _, unsort_idx = sort_idx.sort()
            rnn_output = rnn_output[unsort_idx]

            # Get actual max length from RNN output
            actual_max_len = rnn_output.size(1)

            # Compute attention weights
            attn_scores = self.attention(rnn_output).squeeze(-1)  # [batch, actual_max_len]

            # Mask padding positions (use actual_max_len, not max_len)
            mask = self._create_mask(lengths, actual_max_len).to(sequences.device)
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

            # Softmax
            attn_weights = torch.softmax(attn_scores, dim=1)  # [batch, max_len]

            # Weighted sum
            context = torch.bmm(
                attn_weights.unsqueeze(1),  # [batch, 1, max_len]
                rnn_output                   # [batch, max_len, hidden*dirs]
            ).squeeze(1)  # [batch, hidden*dirs]

            # Output layers
            out = self.dropout(context)
            out = self.relu(self.fc1(out))
            out = self.dropout(out)
            out = self.fc2(out)

            return out

        def _create_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
            """Create attention mask from lengths."""
            batch_size = lengths.size(0)
            mask = torch.arange(max_len).unsqueeze(0).expand(batch_size, -1)
            mask = mask < lengths.unsqueeze(1)
            return mask

        def get_attention_weights(
            self,
            sequences: torch.Tensor,
            lengths: torch.Tensor
        ) -> torch.Tensor:
            """
            Get attention weights for interpretation.

            Args:
                sequences: Padded sequences [batch, max_len]
                lengths: Original sequence lengths [batch]

            Returns:
                Attention weights [batch, max_len]
            """
            batch_size, max_len = sequences.size()

            # Forward pass through embedding and RNN
            embedded = self.embedding(sequences)

            lengths_cpu = lengths.cpu()
            sorted_lengths, sort_idx = lengths_cpu.sort(descending=True)
            sorted_embedded = embedded[sort_idx]

            packed = pack_padded_sequence(
                sorted_embedded,
                sorted_lengths.tolist(),
                batch_first=True
            )

            if self.rnn_type == 'lstm':
                packed_output, _ = self.rnn(packed)
            else:
                packed_output, _ = self.rnn(packed)

            rnn_output, _ = pad_packed_sequence(packed_output, batch_first=True)

            _, unsort_idx = sort_idx.sort()
            rnn_output = rnn_output[unsort_idx]

            # Get actual max length from RNN output
            actual_max_len = rnn_output.size(1)

            # Compute attention
            attn_scores = self.attention(rnn_output).squeeze(-1)
            mask = self._create_mask(lengths, actual_max_len).to(sequences.device)
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
            attn_weights = torch.softmax(attn_scores, dim=1)

            # Pad to original max_len for consistent output
            if actual_max_len < max_len:
                padding = torch.zeros(batch_size, max_len - actual_max_len, device=sequences.device)
                attn_weights = torch.cat([attn_weights, padding], dim=1)

            return attn_weights


    def create_model(
        model_type: str,
        vocab_size: int,
        config: Optional[Config] = None,
        pad_idx: int = 0
    ) -> nn.Module:
        """
        Factory function to create models.

        Args:
            model_type: 'lstm', 'gru', 'bilstm', or 'attention'
            vocab_size: Vocabulary size
            config: Configuration object
            pad_idx: Padding index

        Returns:
            Model instance
        """
        if config is None:
            config = get_config()

        model_config = config.model
        kwargs = {
            'embedding_dim': model_config.embedding_dim,
            'hidden_size': model_config.hidden_size,
            'num_layers': model_config.num_layers,
            'dropout': model_config.dropout,
            'pad_idx': pad_idx
        }

        model_type = model_type.lower()

        if model_type == 'lstm':
            return PatientLSTM(vocab_size, **kwargs)
        elif model_type == 'gru':
            return PatientGRU(vocab_size, **kwargs)
        elif model_type == 'bilstm':
            return PatientBiLSTM(vocab_size, **kwargs)
        elif model_type == 'attention':
            return AttentionPatientRNN(vocab_size, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
