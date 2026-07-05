import abc

import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
            - The first output result[:, 0, 0] does not depend on any input
            - The second output result[:, 0, 1] depends only on x[:, 0, 0]
            - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        self.n_tokens = n_tokens
        self.embedding = torch.nn.Embedding(n_tokens, d_latent)
        self.start_token = torch.nn.Parameter(torch.zeros(1, 1, d_latent))
        self.transformer = torch.nn.TransformerEncoderLayer(
            d_model=d_latent, nhead=4, dim_feedforward=512, batch_first=True
        )
        self.head = torch.nn.Linear(d_latent, n_tokens)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, h, w = x.shape
        seq_len = h * w
        
        x_flat = x.view(B, seq_len)
        x_emb = self.embedding(x_flat)
        start = self.start_token.expand(B, 1, -1)
        x_emb = torch.cat([start, x_emb[:, :-1, :]], dim=1)
        mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        out = self.transformer(x_emb, src_mask=mask, is_causal=True)
        logits = self.head(out)
        return logits.view(B, h, w, self.n_tokens), {}

    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:
        seq_len = h * w
        tokens = torch.zeros(B, 0, dtype=torch.long, device=device)
        for _ in range(seq_len):
            x_emb = self.embedding(tokens) if tokens.shape[1] > 0 else torch.zeros(B, 0, self.embedding.embedding_dim, device=device)
            start = self.start_token.expand(B, 1, -1)
            x_emb = torch.cat([start, x_emb], dim=1)  # (B, current_len+1, d_latent)
            cur_len = x_emb.shape[1]
            mask = torch.nn.Transformer.generate_square_subsequent_mask(cur_len, device=device)
            out = self.transformer(x_emb, src_mask=mask, is_causal=True)
            logits = self.head(out[:, -1, :])  # (B, n_tokens) — only last position
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1)  # (B, 1)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens.view(B, h, w)
