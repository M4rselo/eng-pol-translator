"""
webapp/loader.py
=================
Model loading + inference for the Gender-Context Translator web app.

Design notes (why this file looks the way it does):

- Checkpoints in ./appdata/checkpoints contain only {'epoch', 'model_state'}
  (no optimizer/scaler state) -- so we build the model ourselves and load
  weights with torch.load + load_state_dict. We never use
  modules.Trainer.TrainerModule.load_checkpoint (that path expects a full
  training checkpoint and a CUDA GradScaler, neither of which applies here).

- Beam search is NOT reimplemented here. `modules.Predict.PredictionModule`
  / `PredictionModuleRef` are used directly and unmodified: their `__init__`
  only does cheap dict/attribute setup (no I/O), so we build a fresh,
  disposable instance per request with whatever `alpha` was requested,
  instead of mutating a shared cached object. `alpha` therefore never
  touches any shared state, and there is exactly one beam-search
  implementation in the whole project (yours).

- Confidence is always computed the same way (geometric mean of the actual
  per-token probabilities of the sequence that was chosen), regardless of
  `alpha`. `alpha` only ever influences *which* sequence beam search picks,
  never how confident we report being in it afterwards.

- Source-sentence encoding reuses `BPEEncoder.encode_word` directly, unwrapped
  -- it now falls back to `<unk>` itself for characters outside the training
  vocabulary (fixed in modules/BPE_tokenizer.py, not patched around here).
  `_safe_encode` only adds the explicit truncation to the model's max
  sequence length (a real, previously observed crash on ordinary,
  non-adversarial input -- the original encoder never truncates).
"""
import math
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

WEBAPP_DIR = Path(__file__).resolve().parent
ROOT_DIR = WEBAPP_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules import Model_ref, Predict
from modules.BPE_tokenizer import tokenize_eng

DEVICE = "cpu"

# Hyperparameters per version, taken from research/training_notebooks/training-translator-v1_[2-5].ipynb
VERSION_CONFIG: Dict[str, Dict[str, Any]] = {
    "v1_2": dict(checkpoint="translator_v1_2-15.pt", artifacts="translator_v1_2",
                 num_hiddens=512, ffn_num_hiddens=1024, num_heads=8, num_blks=4, dropout=0.3,
                 max_seq=37, n_ref=1),
    "v1_3": dict(checkpoint="translator_v1_3-15.pt", artifacts="translator_v1_3",
                 num_hiddens=512, ffn_num_hiddens=1024, num_heads=8, num_blks=4, dropout=0.3,
                 max_seq=37, n_ref=1),
    "v1_4": dict(checkpoint="translator_v1_4-13.pt", artifacts="translator_v1_4",
                 num_hiddens=512, ffn_num_hiddens=1024, num_heads=8, num_blks=4, dropout=0.3,
                 max_seq=27, n_ref=2),
    "v1_5": dict(checkpoint="translator_v1_5-13.pt", artifacts="translator_v1_5",
                 num_hiddens=512, ffn_num_hiddens=1024, num_heads=8, num_blks=4, dropout=0.3,
                 max_seq=27, n_ref=2),
}

SELF_OPTS = ["na", "f", "m"]
ADDR_OPTS = ["na", "f", "m", "p"]

CHECKPOINT_DIR = ROOT_DIR / "appdata" / "checkpoints"
ARTIFACT_DIR = ROOT_DIR / "appdata" / "model_reference"

_BUNDLE_CACHE: Dict[str, Dict[str, Any]] = {}


def normalize_version(version: Optional[str]) -> str:
    v = str(version or "v1_5").strip().lower()
    if v not in VERSION_CONFIG:
        raise ValueError(f"Unsupported translator version '{version}'. Valid: {list(VERSION_CONFIG)}")
    return v


def ui_config(version: str) -> Dict[str, Any]:
    cfg = VERSION_CONFIG[normalize_version(version)]
    return {"version": version, "has_addr": cfg["n_ref"] > 1,
            "self_opts": SELF_OPTS, "addr_opts": ADDR_OPTS if cfg["n_ref"] > 1 else [],
            "max_tokens": cfg["max_seq"] - 1}


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_bundle(version: str) -> Dict[str, Any]:
    """Load (and cache) the expensive, immutable parts: model weights + tokenizers.
    Nothing request-specific (alpha, num_k, ref tokens) lives in here."""
    v = normalize_version(version)
    if v in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[v]

    cfg = VERSION_CONFIG[v]
    art_dir = ARTIFACT_DIR / cfg["artifacts"]
    tokenizer_eng = _load_pickle(art_dir / "tokenizer_eng.pkl")
    tokenizer_pol = _load_pickle(art_dir / "tokenizer_pol.pkl")
    encoder_eng = _load_pickle(art_dir / "encoder_eng.pkl")
    encoder_pol = _load_pickle(art_dir / "encoder_pol.pkl")

    encoder = Model_ref.TransformerEncoder(
        tokenizer_eng.vocab_size, cfg["num_hiddens"], cfg["ffn_num_hiddens"],
        cfg["num_heads"], cfg["num_blks"], cfg["dropout"], cfg["max_seq"])
    decoder = Model_ref.TransformerDecoder(
        tokenizer_pol.vocab_size, cfg["num_hiddens"], cfg["ffn_num_hiddens"],
        cfg["num_heads"], cfg["num_blks"], cfg["dropout"], cfg["max_seq"])
    model = Model_ref.Seq2Seq(encoder=encoder, decoder=decoder, lr=1e-4, pad_id=0,
                              n_ref=cfg["n_ref"], device=DEVICE)

    # Checkpoints here hold only {'epoch', 'model_state'} -- not a full TrainerModule
    # checkpoint -- so we load weights directly instead of Trainer.load_checkpoint().
    ckpt = torch.load(CHECKPOINT_DIR / cfg["checkpoint"], map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    bundle = {
        "version": v, "cfg": cfg, "model": model,
        "tokenizer_eng": tokenizer_eng, "tokenizer_pol": tokenizer_pol,
        "encoder_eng": encoder_eng, "encoder_pol": encoder_pol,
        "eos_id": tokenizer_eng.vocab.get("<eos>", 2),
    }
    _BUNDLE_CACHE[v] = bundle
    return bundle


def _build_predicter(bundle: Dict[str, Any], alpha: float):
    """A fresh, disposable PredictionModule(Ref) per request. Construction is
    cheap (dict/attribute assignment only, no I/O) so there is no reason to
    share -- and therefore mutate -- one instance across requests with
    different `alpha`."""
    cfg = bundle["cfg"]
    cls = Predict.PredictionModuleRef if cfg["n_ref"] == 1 else Predict.PredictionModule
    return cls(bundle["tokenizer_eng"], bundle["tokenizer_pol"],
               bundle["encoder_eng"], bundle["encoder_pol"], bundle["model"],
               alpha=alpha, max_seq=cfg["max_seq"] - 1)


def _encode_raw(bundle: Dict[str, Any], text: str) -> Tuple[List[int], List[str]]:
    """Encode source text with the project's own BPEEncoder.encode_word,
    unwrapped -- it now maps characters outside the training vocabulary to
    <unk> itself (see modules/BPE_tokenizer.py). Not truncated -- callers
    decide whether they need the truncated (for the model) or full (for the
    token counter) version."""
    encoder_eng = bundle["encoder_eng"]
    ids: List[int] = []
    for word in tokenize_eng(text):
        if not word:
            continue
        ids.extend(encoder_eng.encode_word(word))

    rev_eng = {v: k for k, v in encoder_eng.vocab_encoder.items()}
    tokens = [rev_eng.get(i, "<unk>").rstrip("_") for i in ids]
    return ids, tokens


def _safe_encode(bundle: Dict[str, Any], text: str) -> Tuple[List[int], List[str]]:
    """Full encoding for actual translation: truncates to the model's max
    sequence length (the original does not, and overflows the positional
    encoding buffer on long input) and appends <eos>."""
    max_len = bundle["cfg"]["max_seq"] - 1  # leave room for the trailing <eos>
    ids, tokens = _encode_raw(bundle, text)
    ids, tokens = ids[:max_len], tokens[:max_len]
    ids.append(bundle["eos_id"])
    return ids, tokens


def tokenize_preview(version: str, text: str) -> Dict[str, Any]:
    """Cheap, model-free tokenization used for the live token counter --
    no encoder/decoder forward pass, just the BPE tokenizer, so it can be
    called far more eagerly than a full translation."""
    bundle = _load_bundle(version)
    max_tokens = bundle["cfg"]["max_seq"] - 1
    ids, tokens = _encode_raw(bundle, (text or "").strip())
    return {
        "tokens": [{"text": t, "id": i, "over_limit": idx >= max_tokens}
                   for idx, (t, i) in enumerate(zip(tokens, ids))],
        "count": len(ids),
        "max_tokens": max_tokens,
    }


def _build_init_seq(predicter, self_ref: str, addr_ref: str) -> torch.Tensor:
    if isinstance(predicter, Predict.PredictionModuleRef):
        prefix_ids = [predicter.self_vocab[self_ref], predicter.bos_id]
    else:
        prefix_ids = [predicter.ref_vocab[f"self_{self_ref}"],
                      predicter.ref_vocab[f"addr_{addr_ref}"], predicter.bos_id]
    return torch.tensor(prefix_ids, device=predicter.device).view(1, -1)


def _analyze(model, predicter, X_enc, valid_len, init_seq, best_seq, want_attention: bool):
    """One extra, clean forward pass (teacher forcing) over the sequence beam
    search already chose. This is the only place per-token probabilities and
    attention weights come from -- never from mid-search bookkeeping, and
    never re-run with a different alpha, so both stay independent of it."""
    prefix_len = init_seq.shape[1]
    with torch.inference_mode():
        model.eval()
        enc_out = model.encoder(X_enc, valid_len)
        dec_state = model.decoder.init_state(enc_out, valid_len)
        if len(best_seq) > 1:
            tail = torch.tensor([best_seq[:-1]], device=predicter.device)
            full_dec_in = torch.cat([init_seq, tail], dim=1)
        else:
            full_dec_in = init_seq
        Y_dec, _ = model.decoder(full_dec_in, dec_state)
        Y_log = F.log_softmax(Y_dec[:, prefix_len - 1:, :], dim=2)

        step_log_probs = [
            float(Y_log[0, i, tok].item()) for i, tok in enumerate(best_seq) if i < Y_log.shape[1]
        ]

        attention = None
        prefix_attention = None
        if want_attention:
            cross = torch.stack(model.decoder._attention_weights[1], dim=0)  # (blocks, heads, q, k)
            gen = cross[:, :, prefix_len - 1:, :]
            attention = gen.mean(dim=(0, 1)).cpu().tolist()  # (tgt_len, src_len), avg over blocks & heads

            # Decoder SELF-attention, but only the columns that are the
            # reference-token prefix itself (self/[addr]/bos): how much did
            # each generated word "look back" at the chosen gender tokens,
            # as opposed to the source sentence.
            self_attn = torch.stack(model.decoder._attention_weights[0], dim=0)  # (blocks, heads, q, k)
            gen_self = self_attn[:, :, prefix_len - 1:, :prefix_len]
            prefix_attention = gen_self.mean(dim=(0, 1)).cpu().tolist()  # (tgt_len, prefix_len)

    return step_log_probs, attention, prefix_attention


def _group_words(tokens: List[str], log_probs: List[float]) -> List[Dict[str, Any]]:
    """Group BPE sub-tokens (word boundary = trailing '_') into display words
    with a per-word confidence, merging bare punctuation into the previous
    word instead of leaving it as its own token."""
    words: List[Dict[str, Any]] = []
    cur_tokens: List[str] = []
    cur_lps: List[float] = []

    def flush():
        if not cur_tokens:
            return
        text = "".join(cur_tokens).replace("_", "")
        mean_lp = sum(cur_lps) / len(cur_lps)
        conf = round(math.exp(mean_lp) * 100.0, 1)
        if words and re.fullmatch(r"[,.!?:;]+", text):
            words[-1]["word"] += text
        else:
            words.append({"word": text, "confidence": conf})

    for tok, lp in zip(tokens, log_probs):
        if tok == "<eos>":
            continue
        cur_tokens.append(tok)
        cur_lps.append(lp)
        if tok.endswith("_"):
            flush()
            cur_tokens, cur_lps = [], []
    flush()

    if words:
        words[0]["word"] = words[0]["word"].capitalize()
    return words


def translate(version: str, text: str, self_ref: str = "na", addr_ref: str = "na",
              alpha: float = 0.6, num_k: int = 5, want_attention: bool = False) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {"translation": "", "confidence": 0.0, "words": [], "src_tokens": [],
                "tgt_tokens": [], "subwords": [], "attention": None, "prefix_attention": None}

    bundle = _load_bundle(version)
    predicter = _build_predicter(bundle, alpha)

    eng_ids, src_tokens = _safe_encode(bundle, text)
    init_seq = _build_init_seq(predicter, self_ref, addr_ref)
    X_enc, valid_len = predicter.get_enc_input(eng_ids)  # reused from Predict.py, unmodified

    best_seq = predicter.predict_ids(X_enc, valid_len, init_seq, num_k)  # reused, unmodified beam search

    raw_tgt = [predicter.rev_pol.get(t, "<unk>") for t in best_seq]
    translation = predicter.out_handler(raw_tgt[:-1] if raw_tgt and raw_tgt[-1] == "<eos>" else raw_tgt)

    step_log_probs, attention, prefix_attention = _analyze(
        bundle["model"], predicter, X_enc, valid_len, init_seq, best_seq, want_attention)

    lexical_lps = step_log_probs[:-1] if len(step_log_probs) > 1 else step_log_probs
    confidence = round(math.exp(sum(lexical_lps) / max(1, len(lexical_lps))) * 100.0, 1)
    words = _group_words(raw_tgt, step_log_probs)
    has_eos = bool(raw_tgt) and raw_tgt[-1] == "<eos>"
    tgt_tokens = [t.rstrip("_") for t in raw_tgt if t != "<eos>"]

    result = {
        "translation": translation,
        "confidence": confidence,
        "words": words,
        "src_tokens": src_tokens,
        "tgt_tokens": tgt_tokens,
        # Raw sub-word pieces + their vocabulary ids, for the "show tokens/ids" view.
        "subwords": [{"text": t.rstrip("_"), "id": i}
                     for t, i in zip(raw_tgt[:-1] if has_eos else raw_tgt, best_seq[:-1] if has_eos else best_seq)],
    }
    if want_attention:
        # Both attention matrices have one row per best_seq position
        # (including the row that produced <eos>); tgt_tokens has <eos>
        # filtered out, so drop the matching trailing row to keep rows
        # aligned 1:1 with their labels.
        matrix = attention[:-1] if has_eos and attention else attention
        result["attention"] = {"matrix": matrix, "src_tokens": src_tokens, "tgt_tokens": tgt_tokens}

        prefix_labels = [f"<self_{self_ref}>"]
        if isinstance(predicter, Predict.PredictionModule):
            prefix_labels.append(f"<addr_{addr_ref}>")
        prefix_labels.append("<bos>")
        pmatrix = prefix_attention[:-1] if has_eos and prefix_attention else prefix_attention
        result["prefix_attention"] = {"matrix": pmatrix, "prefix_tokens": prefix_labels, "tgt_tokens": tgt_tokens}
    return result


def compare_all(version: str, text: str, alpha: float = 0.6, num_k: int = 5) -> Dict[str, Any]:
    """Translate `text` for every reference-token combination valid for this
    version. `alpha`/`num_k` are shared across all combinations (one
    disposable predicter is enough, since only the decoder prefix differs
    between iterations) -- but each combination still runs its own beam
    search and encoder pass, deliberately: sharing the encoder output across
    combinations would require re-implementing beam search's internals
    (see notes above), and measured against the actual cost of the
    autoregressive decode loop, that saving turned out to be negligible."""
    text = (text or "").strip()
    bundle = _load_bundle(version)
    cfg = bundle["cfg"]
    if not text:
        return {"version": version, "has_addr": cfg["n_ref"] > 1, "rows": []}

    predicter = _build_predicter(bundle, alpha)
    eng_ids, _ = _safe_encode(bundle, text)
    X_enc, valid_len = predicter.get_enc_input(eng_ids)

    addr_opts = ADDR_OPTS if cfg["n_ref"] > 1 else ["na"]
    rows = []
    for self_ref in SELF_OPTS:
        for addr_ref in addr_opts:
            init_seq = _build_init_seq(predicter, self_ref, addr_ref)
            best_seq = predicter.predict_ids(X_enc, valid_len, init_seq, num_k)
            raw_tgt = [predicter.rev_pol.get(t, "<unk>") for t in best_seq]
            translation = predicter.out_handler(raw_tgt[:-1] if raw_tgt and raw_tgt[-1] == "<eos>" else raw_tgt)

            step_log_probs, _, _ = _analyze(bundle["model"], predicter, X_enc, valid_len,
                                             init_seq, best_seq, want_attention=False)
            lexical_lps = step_log_probs[:-1] if len(step_log_probs) > 1 else step_log_probs
            confidence = round(math.exp(sum(lexical_lps) / max(1, len(lexical_lps))) * 100.0, 1)

            rows.append({"self_ref": self_ref, "addr_ref": addr_ref if cfg["n_ref"] > 1 else None,
                         "translation": translation, "confidence": confidence})

    return {"version": version, "has_addr": cfg["n_ref"] > 1, "rows": rows}
