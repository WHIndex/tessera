from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def resolve_model_dir(path: str) -> Path:
    p = Path(path)
    if (p / "config.json").exists():
        return p
    snapshots = p / "snapshots"
    if snapshots.exists() and snapshots.is_dir():
        cands = sorted([x for x in snapshots.iterdir() if x.is_dir()])
        for cand in reversed(cands):
            if (cand / "config.json").exists():
                return cand
        if cands:
            return cands[-1]
    raise FileNotFoundError(path)


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return torch.sum(last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


class NVEmbedModel:
    def __init__(self, model):
        self.model = model
        self._tessera_backend = "nvembed"

    def eval(self):
        self.model.eval()
        return self


def _embedding_backend() -> str:
    return os.environ.get("TESSERA_EMBED_BACKEND", "hf").strip().lower()


def load_e5(model_dir: str, device: str | None = None):
    resolved = resolve_model_dir(model_dir)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    backend = _embedding_backend()
    trust_remote = backend in {"nvembed", "nv-embed", "nv_embed"} or os.environ.get(
        "TESSERA_HF_TRUST_REMOTE_CODE", "0"
    ).strip() in {"1", "true", "yes"}
    tokenizer = AutoTokenizer.from_pretrained(str(resolved), trust_remote_code=trust_remote)
    kwargs = {"trust_remote_code": trust_remote}
    if backend in {"nvembed", "nv-embed", "nv_embed"} and device.startswith("cuda"):
        kwargs["torch_dtype"] = torch.float16
    device_map = os.environ.get("TESSERA_NV_DEVICE_MAP", "").strip()
    if backend in {"nvembed", "nv-embed", "nv_embed"} and device_map:
        kwargs["device_map"] = device_map
    model = AutoModel.from_pretrained(str(resolved), **kwargs)
    if not device_map:
        model = model.to(device)
    if backend in {"nvembed", "nv-embed", "nv_embed"}:
        if hasattr(model, "tokenizer"):
            try:
                model.tokenizer.padding_side = os.environ.get("TESSERA_NV_PADDING_SIDE", "right")
            except Exception:
                pass
        model = NVEmbedModel(model)
    model.eval()
    return tokenizer, model, device, resolved


def _encode_nvembed_batch(
    batch: list[str],
    model,
    instruction: str,
    max_length: int,
) -> torch.Tensor | np.ndarray:
    encoder = model.model if isinstance(model, NVEmbedModel) else model
    kwargs = {
        "instruction": instruction,
        "max_length": int(max_length),
    }
    try:
        return encoder.encode(batch, **kwargs)
    except TypeError:
        if instruction:
            return encoder.encode([instruction + x for x in batch], max_length=int(max_length))
        return encoder.encode(batch, max_length=int(max_length))


def encode_texts(
    texts: List[str],
    tokenizer,
    model,
    device: str,
    batch_size: int = 64,
    max_length: int = 512,
    progress_every_batches: int = 20,
    pooling_mode: str = "mean",
    query_prefix: str = "",
) -> np.ndarray:
    if getattr(model, "_tessera_backend", "") == "nvembed":
        vecs = []
        instruction = query_prefix
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                emb = _encode_nvembed_batch(batch, model, instruction, max_length)
                if isinstance(emb, torch.Tensor):
                    emb_t = emb.detach()
                else:
                    emb_t = torch.as_tensor(emb)
                emb_t = torch.nn.functional.normalize(emb_t.float(), p=2, dim=1)
                vecs.append(emb_t.cpu().numpy().astype(np.float32))
                if (i // batch_size) % progress_every_batches == 0:
                    print(f"[encode:nvembed] {min(i + batch_size, len(texts))}/{len(texts)}")
        if not vecs:
            return np.zeros((0, 1), dtype=np.float32)
        return np.concatenate(vecs, axis=0)

    vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            if query_prefix:
                batch = [query_prefix + t for t in batch]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state
            if pooling_mode == "cls":
                emb = hidden[:, 0, :]
            else:
                emb = mean_pool(hidden, inputs["attention_mask"])
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            vecs.append(emb.cpu().numpy())
            if (i // batch_size) % progress_every_batches == 0:
                print(f"[encode] {min(i + batch_size, len(texts))}/{len(texts)}")

    return np.concatenate(vecs, axis=0)
