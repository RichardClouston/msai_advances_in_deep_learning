from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image

from .autoregressive import Autoregressive
from .bsq import Tokenizer

SCALE = 1 << 16

def _probs_to_cdf(probs):
    counts = np.maximum(1, (probs * SCALE).astype(np.int64))
    cdf = np.zeros(len(probs) + 1, dtype=np.int64)
    cdf[1:] = np.cumsum(counts)
    return cdf

class ArithmeticEncoder:
    HALF = 1 << 31
    QUARTER = 1 << 30
    THREE_QUARTER = 3 << 30

    def __init__(self):
        self.low = 0
        self.high = (1 << 32) - 1
        self.pending = 0
        self.bits = []

    def encode(self, sym, cdf):
        total = cdf[-1]
        rng = self.high - self.low + 1
        self.high = self.low + rng * cdf[sym + 1] // total - 1
        self.low = self.low + rng * cdf[sym] // total
        self._normalize()

    def _normalize(self):
        while True:
            if self.high < self.HALF:
                self.bits.append(0)
                for _ in range(self.pending): self.bits.append(1)
                self.pending = 0
                self.low <<= 1
                self.high = (self.high << 1) | 1
            elif self.low >= self.HALF:
                self.bits.append(1)
                for _ in range(self.pending): self.bits.append(0)
                self.pending = 0
                self.low = (self.low - self.HALF) << 1
                self.high = ((self.high - self.HALF) << 1) | 1
            elif self.low >= self.QUARTER and self.high < self.THREE_QUARTER:
                self.pending += 1
                self.low = (self.low - self.QUARTER) << 1
                self.high = ((self.high - self.QUARTER) << 1) | 1
            else:
                break

    def flush(self):
        self.pending += 1
        if self.low < self.QUARTER:
            self.bits.append(0)
            for _ in range(self.pending): self.bits.append(1)
        else:
            self.bits.append(1)
            for _ in range(self.pending): self.bits.append(0)
        while len(self.bits) % 8:
            self.bits.append(0)
        result = bytearray()
        for i in range(0, len(self.bits), 8):
            b = 0
            for j in range(8): b = (b << 1) | self.bits[i + j]
            result.append(b)
        return bytes(result)

class ArithmeticDecoder:
    HALF = 1 << 31
    QUARTER = 1 << 30
    THREE_QUARTER = 3 << 30

    def __init__(self, data):
        self._gen = self._bit_gen(data)
        self.low = 0
        self.high = (1 << 32) - 1
        self.value = 0
        for _ in range(32):
            self.value = (self.value << 1) | next(self._gen, 0)

    def _bit_gen(self, data):
        for byte in data:
            for j in range(7, -1, -1):
                yield (byte >> j) & 1

    def decode(self, cdf):
        total = cdf[-1]
        rng = self.high - self.low + 1
        scaled = ((self.value - self.low + 1) * total - 1) // rng
        sym = int(np.searchsorted(cdf[1:], scaled, side='right'))
        sym = min(sym, len(cdf) - 2)
        self.high = self.low + rng * cdf[sym + 1] // total - 1
        self.low = self.low + rng * cdf[sym] // total
        self._normalize()
        return sym

    def _normalize(self):
        while True:
            if self.high < self.HALF:
                self.low <<= 1
                self.high = (self.high << 1) | 1
                self.value = (self.value << 1) | next(self._gen, 0)
            elif self.low >= self.HALF:
                self.low = (self.low - self.HALF) << 1
                self.high = ((self.high - self.HALF) << 1) | 1
                self.value = (self.value - self.HALF) << 1 | next(self._gen, 0)
            elif self.low >= self.QUARTER and self.high < self.THREE_QUARTER:
                self.low = (self.low - self.QUARTER) << 1
                self.high = ((self.high - self.QUARTER) << 1) | 1
                self.value = (self.value - self.QUARTER) << 1 | next(self._gen, 0)
            else:
                break

class Compressor:
    def __init__(self, tokenizer: Tokenizer, autoregressive: Autoregressive):
        super().__init__()
        self.tokenizer = tokenizer
        self.autoregressive = autoregressive

    def compress(self, x: torch.Tensor) -> bytes:
        with torch.no_grad():
            tokens = self.tokenizer.encode_index(x.unsqueeze(0))
            _, h, w = tokens.shape
            n_tokens = 1024

            logits, _ = self.autoregressive(tokens)
            probs = torch.softmax(logits, dim=-1)[0].view(-1, n_tokens).cpu().numpy()
            tokens_flat = tokens[0].view(-1).cpu().numpy()

            coder = ArithmeticEncoder()
            for i in range(len(tokens_flat)):
                cdf = _probs_to_cdf(probs[i])
                coder.encode(int(tokens_flat[i]), cdf)
            return coder.flush()

    def decompress(self, data: bytes) -> torch.Tensor:
        device = next(self.autoregressive.parameters()).device
        h, w, n_tokens = 20, 30, 1024

        dec = ArithmeticDecoder(data)
        tokens = torch.zeros(1, h, w, dtype=torch.long, device=device)

        with torch.no_grad():
            for i in range(h * w):
                logits, _ = self.autoregressive(tokens)
                probs_i = torch.softmax(logits[0].view(-1, n_tokens)[i], dim=-1).cpu().numpy()
                sym = dec.decode(_probs_to_cdf(probs_i))
                tokens[0].view(-1)[i] = sym

        return self.tokenizer.decode_index(tokens)[0]

def compress(tokenizer: Path, autoregressive: Path, image: Path, compressed_image: Path):
    """
    Compress images using a pre-trained model.

    tokenizer: Path to the tokenizer model.
    autoregressive: Path to the autoregressive model.
    images: Path to the image to compress.
    compressed_image: Path to save the compressed image tensor.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    x = torch.tensor(np.array(Image.open(image)), dtype=torch.uint8, device=device)
    cmp_img = cmp.compress(x.float() / 255.0 - 0.5)
    with open(compressed_image, "wb") as f:
        f.write(cmp_img)


def decompress(tokenizer: Path, autoregressive: Path, compressed_image: Path, image: Path):
    """
    Decompress images using a pre-trained model.

    tokenizer: Path to the tokenizer model.
    autoregressive: Path to the autoregressive model.
    compressed_image: Path to the compressed image tensor.
    images: Path to save the image to compress.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    with open(compressed_image, "rb") as f:
        cmp_img = f.read()

    x = cmp.decompress(cmp_img)
    img = Image.fromarray(((x + 0.5) * 255.0).clamp(min=0, max=255).byte().cpu().numpy())
    img.save(image)


if __name__ == "__main__":
    from fire import Fire

    Fire({"compress": compress, "decompress": decompress})
