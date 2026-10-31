"""
Universal Global Coding Model Trainer (Decoder-Only Causal Transformer)
Trains a high-performance Causal Transformer model on the Universal Global Coding Dataset.
Features: Scaled Dot-Product Attention (SDPA), Cross-Entropy loss, perplexity tracking,
and code generation inference.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Workspace paths
BASE_DIR = Path(__file__).resolve().parent
TRAIN_FILE = BASE_DIR / "universal-global-coding-dataset" / "train" / "train.jsonl"
VAL_FILE = BASE_DIR / "universal-global-coding-dataset" / "validation" / "validation.jsonl"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

# Special Tokens
PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
FIM_PREFIX = 3
FIM_SUFFIX = 4
FIM_MIDDLE = 5
NUM_SPECIAL = 6
VOCAB_SIZE = 256 + NUM_SPECIAL  # 256 byte tokens + special tokens

class ByteTokenizer:
    """UTF-8 Byte-level tokenizer ensuring 100% vocabulary coverage without out-of-vocab issues."""

    @staticmethod
    def encode(text: str) -> List[int]:
        raw_bytes = text.encode("utf-8", errors="ignore")
        return [b + NUM_SPECIAL for b in raw_bytes]

    @staticmethod
    def decode(tokens: List[int]) -> str:
        byte_list = []
        for t in tokens:
            if t >= NUM_SPECIAL:
                byte_list.append(t - NUM_SPECIAL)
        return bytes(byte_list).decode("utf-8", errors="replace")


class CodeDataset(Dataset):
    """Dataset streaming tokenized code chunks."""

    def __init__(self, file_path: Path, max_length: int = 256):
        self.samples: List[torch.Tensor] = []
        self.max_length = max_length
        self.tokenizer = ByteTokenizer()

        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    content = item.get("content", "")
                    if not content:
                        continue
                    token_ids = [BOS_TOKEN] + self.tokenizer.encode(content) + [EOS_TOKEN]
                    # Chunk into windows of max_length
                    for i in range(0, len(token_ids) - 1, max_length):
                        chunk = token_ids[i:i + max_length + 1]
                        if len(chunk) > 4:  # At least 4 tokens
                            self.samples.append(torch.tensor(chunk, dtype=torch.long))

        if not self.samples:
            # Fallback dummy chunk if dataset is empty
            self.samples.append(torch.tensor([BOS_TOKEN, 10, 20, 30, EOS_TOKEN], dtype=torch.long))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.samples[idx]
        if len(chunk) < self.max_length + 1:
            # Pad
            padded = torch.full((self.max_length + 1,), PAD_TOKEN, dtype=torch.long)
            padded[:len(chunk)] = chunk
            chunk = padded
        else:
            chunk = chunk[:self.max_length + 1]

        x = chunk[:-1]
        y = chunk[1:]
        return x, y


class CausalSelfAttention(nn.Module):
    """Multi-Head Self-Attention with causal masking using PyTorch SDPA."""

    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Transformer Decoder Block: LayerNorm -> Attention -> Residual -> LayerNorm -> MLP -> Residual."""

    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class CodingTransformerLM(nn.Module):
    """
    State-of-the-Art Decoder-Only Causal Language Model for Code.
    Architecture: Token + Learned Positional Embeddings -> Stack of Blocks -> LayerNorm -> LM Head.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        max_seq_len: int = 512,
        n_embd: int = 256,
        n_layer: int = 4,
        n_head: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, n_embd, padding_idx=PAD_TOKEN)
        self.pos_emb = nn.Embedding(max_seq_len, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.tok_emb.weight

        # Parameter initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(pos)
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Flatten for cross-entropy, ignore padding
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=PAD_TOKEN
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, prompt_tokens: List[int], max_new_tokens: int = 50, temperature: float = 0.8) -> List[int]:
        self.eval()
        idx = torch.tensor([prompt_tokens], dtype=torch.long, device=self.tok_emb.weight.device)
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
            if next_token.item() == EOS_TOKEN:
                break
        return idx[0].tolist()


def train_model(
    epochs: int = 2,
    batch_size: int = 4,
    learning_rate: float = 5e-4,
    max_steps: Optional[int] = 50,
    seq_len: int = 128,
    n_embd: int = 128,
    n_layer: int = 3,
    n_head: int = 4,
    device_str: Optional[str] = None
):
    print("=====================================================================")
    print("UNIVERSAL CODING MODEL TRAINING ENGINE")
    print("=====================================================================")

    # Select hardware device
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Utilizing hardware compute device: {device}")

    # Load Train and Validation splits
    print(f"[Data] Loading training dataset from: {TRAIN_FILE}")
    train_dataset = CodeDataset(TRAIN_FILE, max_length=seq_len)
    val_dataset = CodeDataset(VAL_FILE, max_length=seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"[Data] Training sequences: {len(train_dataset):,} | Validation sequences: {len(val_dataset):,}")

    # Initialize Model
    model = CodingTransformerLM(
        vocab_size=VOCAB_SIZE,
        max_seq_len=seq_len,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Architecture: Decoder-Only Causal Transformer | Parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    step = 0
    start_time = time.time()
    best_val_loss = float("inf")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[Training] Starting gradient updates...")
    print("---------------------------------------------------------------------")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for x, y in train_loader:
            step += 1
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

            if step % 10 == 0 or step == 1:
                ppl = math.exp(min(loss.item(), 20.0))
                elapsed = time.time() - start_time
                tok_per_sec = (step * batch_size * seq_len) / max(elapsed, 0.001)
                print(f"Epoch [{epoch}/{epochs}] | Step {step:3d} | Loss: {loss.item():.4f} | PPL: {ppl:.2f} | Speed: {tok_per_sec:.0f} tok/s")

            if max_steps and step >= max_steps:
                break

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                _, v_loss = model(vx, vy)
                val_loss += v_loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        val_ppl = math.exp(min(avg_val_loss, 20.0))
        print(f"\n---> Epoch {epoch} Evaluation | Val Loss: {avg_val_loss:.4f} | Val Perplexity: {val_ppl:.2f}")

        # Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = CHECKPOINT_DIR / "best_coding_model.pt"
            torch.save({
                "epoch": epoch,
                "step": step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "config": {
                    "vocab_size": VOCAB_SIZE,
                    "max_seq_len": seq_len,
                    "n_embd": n_embd,
                    "n_layer": n_layer,
                    "n_head": n_head
                }
            }, ckpt_path)
            print(f"---> [Checkpoint] Best model checkpoint saved to: {ckpt_path}\n")

        if max_steps and step >= max_steps:
            print(f"[Training] Reached max steps limit ({max_steps}). Training complete.")
            break

    total_time = time.time() - start_time
    print("---------------------------------------------------------------------")
    print(f"[Training Summary] Total Time: {total_time:.1f}s | Steps: {step} | Best Val Loss: {best_val_loss:.4f}")

    # Inference Demonstration
    print("\n[Inference Demonstration] Testing generation from trained model checkpoint...")
    sample_prompt = "def binary_search(arr, target):"
    prompt_tokens = ByteTokenizer.encode(sample_prompt)
    generated_tokens = model.generate(prompt_tokens, max_new_tokens=40, temperature=0.7)
    decoded_text = ByteTokenizer.decode(generated_tokens)
    clean_text = decoded_text.encode("ascii", errors="backslashreplace").decode("ascii")
    print(f"Prompt: {sample_prompt}")
    print(f"Generated Completion:\n{clean_text}")
    print("=====================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal Coding Model Trainer")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per step")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=40, help="Maximum gradient update steps")
    parser.add_argument("--seq-len", type=int, default=128, help="Context length sequence window")
    parser.add_argument("--device", type=str, default=None, help="Device: 'cpu' or 'cuda'")
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        seq_len=args.seq_len,
        device_str=args.device
    )
