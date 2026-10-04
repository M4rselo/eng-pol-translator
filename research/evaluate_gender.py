"""
Gender-control evaluation on a clean held-out set (inference only, no retraining).

Run from the repo root:

    python research/evaluate_gender.py \
        --data local_data/translator_v2/data_final_opensub.pkl \
        --versions v1_4 v1_5 --n 200

What it does
------------
1. Reproduces the exact training sample of every evaluated version
   (same tokenizers, same length trimming, same `sample(1_500_000, random_state=42)`
   as in research/training_notebooks/training-translator-v1_[45].ipynb) and prints
   sanity checks against the numbers logged in those notebooks.
2. Builds a held-out set from rows that were never sampled for training, then also
   drops every row whose English or Polish text occurs anywhere in a training sample
   (OpenSubtitles has many exact duplicates).
3. Measures what the project actually claims, using simple morphological detectors
   (past-tense / conditional endings):
     A. speaker control   - does <self_f>/<self_m> produce feminine/masculine forms?
     B. neutral stability - on first-person sentences with no gendered form in the
                            reference, does the output stay the same for f/m/na?
     C. addressee control - does <addr_f>/<addr_m>/<addr_p> produce the right forms?
     D. general quality   - corpus chrF / BLEU (sacrebleu, lowercased), if installed.
4. Writes research/results/eval_summary.md (+ JSON metrics and a CSV with every output).

Detector caveat: only forms like -łam/-łem, -łabym/-łbym, -łaś/-łeś, -liście/-łyście
are detected. Test sentences are filtered so that the reference contains such a form,
which makes the numbers measurable but biased towards past tense / conditional.
"""
import argparse
import csv
import json
import math
import pickle
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "webapp"))

import pandas as pd  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

import loader  # noqa: E402  (webapp/loader.py - model + tokenizer loading)
from modules.BPE_tokenizer import tokenize_eng, tokenize_pol  # noqa: E402

# Constants copied from the v1_4 / v1_5 training notebooks.
MAX_LEN = 26
TRAIN_SAMPLE = 1_500_000
TRAIN_SEED = 42
EXPECTED_DROPPED = 5825
EXPECTED_COUNTS = {"v1_5": {"NA_OTHER": 1079339, "NA": 313358, "M": 75493, "F": 31810}}

TOKEN_OF_LABEL = {"F": "f", "M": "m", "P": "p"}

# ---------------------------------------------------------------------------
# Morphological detectors
# ---------------------------------------------------------------------------
SPK_F_SUFFIX = ("łam", "łabym")
SPK_M_SUFFIX = ("łem", "łbym")
SPK_F_EDGE = re.compile(r"(?:\w*działam|złam|połam|odłam|rozłam|przełam)")
# Nouns in the instrumental case that end in -łem (stołem, aniołem, Pawłem, ...)
SPK_M_EDGE = re.compile(
    r"(?:materia|cia|z|ko|czo|gard|krzes|mas|źród|god|skrzyd|siod|dzie|szk|do|ty|anio|tytu|"
    r"artyku|kościo|zespo|szczegó|sto|udzia|wydzia|oddzia|dzia|genera|admira|py|diab|kana|"
    r"kryszta|myd|zapa|strza|sygna|kardyna|\w*mys|mu|peda|kowad|zwierciad|popychad|aposto|"
    r"protoko|paw|micha|or|os|koz|węz|popio|kot|rzemios|wios|wo)łem"
)
ADDR_SUFFIX = {"F": ("łaś", "łabyś"), "M": ("łeś", "łbyś"),
               "P": ("liście", "łyście", "libyście", "łybyście")}
WORD_RE = re.compile(r"\w+")


def _words(text):
    return WORD_RE.findall(str(text).lower())


def _has(words, suffixes, edge=None):
    return any(w.endswith(suffixes) and len(w) > len(min(suffixes, key=len)) + 1
               and not (edge and edge.fullmatch(w)) for w in words)


def detect_speaker(text):
    w = _words(text)
    f, m = _has(w, SPK_F_SUFFIX, SPK_F_EDGE), _has(w, SPK_M_SUFFIX, SPK_M_EDGE)
    return "MIXED" if f and m else "F" if f else "M" if m else "NONE"


def detect_addr(text):
    w = _words(text)
    found = [lbl for lbl, suf in ADDR_SUFFIX.items() if _has(w, suf)]
    return found[0] if len(found) == 1 else ("MIXED" if found else "NONE")


# ---------------------------------------------------------------------------
# Held-out construction
# ---------------------------------------------------------------------------
def _token_lengths(texts, tok_fn, encoder, desc):
    cache, out = {}, []
    for t in tqdm(texts, desc=desc, leave=False):
        n = 0
        for w in tok_fn(t):
            l = cache.get(w)
            if l is None:
                l = cache[w] = sum(1 for _ in encoder.encode_word(w))
            n += l
        out.append(n)
    return pd.Series(out, index=texts.index)


def reproduce_training_rows(df, version):
    bundle = loader._load_bundle(version)
    n_ref = bundle["cfg"]["n_ref"]
    eng_len = _token_lengths(df["eng_text"], tokenize_eng, bundle["encoder_eng"], f"{version} eng lengths") + 1
    pol_len = _token_lengths(df["pol_text"], tokenize_pol, bundle["encoder_pol"], f"{version} pol lengths") + n_ref + 2
    keep = (eng_len <= MAX_LEN) & (pol_len <= MAX_LEN)

    kept_positions = df.index[keep].to_numpy()
    trimmed = df[keep].reset_index(drop=True)
    sampled = trimmed.sample(TRAIN_SAMPLE, random_state=TRAIN_SEED).index.to_numpy()
    train_rows = kept_positions[sampled]

    dropped = int((~keep).sum())
    counts = trimmed.loc[sampled, "self_ref"].value_counts().to_dict()
    ok = dropped == EXPECTED_DROPPED
    if version in EXPECTED_COUNTS:
        ok = ok and all(counts.get(k) == v for k, v in EXPECTED_COUNTS[version].items())
    print(f"[{version}] dropped by length: {dropped} (notebook: {EXPECTED_DROPPED}) | "
          f"sample self_ref counts: {counts} | reproduction check: {'OK' if ok else 'MISMATCH'}")
    if not ok:
        print(f"  WARNING: [{version}] training sample could not be verified - held-out may overlap training.")
    return set(train_rows.tolist()), keep, ok


def build_heldout(df, versions):
    train_rows, keep_all, checks = set(), pd.Series(True, index=df.index), {}
    for v in versions:
        rows, keep, ok = reproduce_training_rows(df, v)
        train_rows |= rows
        keep_all &= keep
        checks[v] = ok

    in_train = df.index.isin(list(train_rows))
    norm_eng = df["eng_text"].str.lower().str.strip()
    norm_pol = df["pol_text"].str.lower().str.strip()
    seen_eng, seen_pol = set(norm_eng[in_train]), set(norm_pol[in_train])

    mask = (~in_train) & keep_all & ~norm_eng.isin(seen_eng) & ~norm_pol.isin(seen_pol)
    held = df[mask].assign(_norm=norm_eng[mask]).drop_duplicates("_norm").drop(columns="_norm")
    print(f"Held-out pool after removing training rows and any duplicate of them: {len(held):,} rows")
    return held, checks


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------
class Translator:
    def __init__(self, version, beam, alpha):
        self.bundle = loader._load_bundle(version)
        if self.bundle["cfg"]["n_ref"] != 2:
            raise ValueError(f"{version} has no addressee token - only v1_4 / v1_5 are supported.")
        self.pred = loader._build_predicter(self.bundle, alpha)
        self.beam, self.cache = beam, {}

    def __call__(self, text, self_ref, addr_ref):
        key = (text, self_ref, addr_ref)
        if key not in self.cache:
            ids, _ = loader._safe_encode(self.bundle, text)
            X_enc, valid_len = self.pred.get_enc_input(ids)
            init_seq = loader._build_init_seq(self.pred, self_ref, addr_ref)
            best = self.pred.predict_ids(X_enc, valid_len, init_seq, self.beam)
            toks = [self.pred.rev_pol.get(t, "<unk>") for t in best]
            if toks and toks[-1] == "<eos>":
                toks = toks[:-1]
            self.cache[key] = self.pred.out_handler(toks)
        return self.cache[key]


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def fmt(k, n):
    p, lo, hi = wilson(k, n)
    return "n/a" if n == 0 else f"{p * 100:.1f}% ({lo * 100:.0f}–{hi * 100:.0f})"


def sample_rows(df, n, seed):
    return df.sample(min(n, len(df)), random_state=seed) if len(df) else df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def run_version(version, held, args, records):
    tr = Translator(version, args.beam, args.alpha)
    m = {}

    def addr_tok(row):
        return TOKEN_OF_LABEL.get(row["addr_ref"], "na")

    def self_tok(row):
        return TOKEN_OF_LABEL.get(row["self_ref"], "na") if row["self_ref"] in ("F", "M") else "na"

    # A. speaker control
    ref_spk = held["pol_text"].map(detect_speaker)
    spk = pd.concat([sample_rows(held[(held["self_ref"] == lbl) & (ref_spk == lbl)], args.n, args.seed)
                     for lbl in ("F", "M")])
    out = {"f": Counter(), "m": Counter(), "na": Counter()}
    flips = 0
    for _, row in tqdm(spk.iterrows(), total=len(spk), desc=f"{version} A speaker"):
        det = {}
        for tok in ("f", "m", "na"):
            hyp = tr(row["eng_text"], tok, addr_tok(row))
            det[tok] = detect_speaker(hyp)
            out[tok][det[tok]] += 1
            records.append([version, "A_speaker", row["self_ref"], tok, addr_tok(row),
                            row["eng_text"], row["pol_text"], hyp, det[tok]])
        flips += det["f"] == "F" and det["m"] == "M"
    n = len(spk)
    m["A"] = {"n": n, "f_correct": out["f"]["F"], "m_correct": out["m"]["M"], "flip": flips,
              "na_dist": dict(out["na"]), "f_dist": dict(out["f"]), "m_dist": dict(out["m"])}

    # B. neutral stability
    neu = sample_rows(held[(held["self_ref"] == "NA") & (ref_spk == "NONE")], args.n, args.seed)
    same, introduced = 0, 0
    for _, row in tqdm(neu.iterrows(), total=len(neu), desc=f"{version} B neutral"):
        hyps = {}
        for tok in ("f", "m", "na"):
            hyps[tok] = tr(row["eng_text"], tok, addr_tok(row))
            records.append([version, "B_neutral", "NA", tok, addr_tok(row),
                            row["eng_text"], row["pol_text"], hyps[tok], detect_speaker(hyps[tok])])
        same += len(set(h.lower() for h in hyps.values())) == 1
        introduced += detect_speaker(hyps["f"]) in ("F", "MIXED") or detect_speaker(hyps["m"]) in ("M", "MIXED")
    m["B"] = {"n": len(neu), "identical": same, "introduced": introduced}

    # C. addressee control
    ref_addr = held["pol_text"].map(detect_addr)
    adr = pd.concat([sample_rows(held[(held["addr_ref"] == lbl) & (ref_addr == lbl)], args.n, args.seed)
                     for lbl in ("F", "M", "P")])
    conf = {tok: Counter() for tok in ("f", "m", "p", "na")}
    for _, row in tqdm(adr.iterrows(), total=len(adr), desc=f"{version} C addressee"):
        for tok in ("f", "m", "p", "na"):
            hyp = tr(row["eng_text"], self_tok(row), tok)
            d = detect_addr(hyp)
            conf[tok][d] += 1
            records.append([version, "C_addressee", row["addr_ref"], self_tok(row), tok,
                            row["eng_text"], row["pol_text"], hyp, d])
    m["C"] = {"n": len(adr), "confusion": {k: dict(v) for k, v in conf.items()}}

    # D. general quality
    gen = sample_rows(held[held["self_ref"] == "NA_OTHER"], args.n_general, args.seed)
    hyps, refs = [], []
    for _, row in tqdm(gen.iterrows(), total=len(gen), desc=f"{version} D general"):
        hyp = tr(row["eng_text"], "na", "na")
        hyps.append(hyp.lower())
        refs.append(str(row["pol_text"]).lower())
        records.append([version, "D_general", "NA_OTHER", "na", "na", row["eng_text"], row["pol_text"], hyp, ""])
    try:
        import sacrebleu
        m["D"] = {"n": len(gen), "chrF": round(sacrebleu.corpus_chrf(hyps, [refs]).score, 1),
                  "BLEU": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 1)}
    except ImportError:
        m["D"] = {"n": len(gen), "chrF": None, "BLEU": None}
        print("sacrebleu not installed - skipping chrF/BLEU (pip install sacrebleu)")
    return m


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_report(metrics, checks, args, held_size, path):
    vs = list(metrics)
    head = "| | " + " | ".join(f"`{v}`" for v in vs) + " |\n|---|" + "---|" * len(vs) + "\n"

    def row(label, fn):
        return f"| {label} | " + " | ".join(fn(metrics[v]) for v in vs) + " |\n"

    def share(dist, key, n):
        return fmt(dist.get(key, 0), n)

    s = [f"# Gender-control evaluation\n\n"
         f"Held-out pool: {held_size:,} sentence pairs never sampled for training (and not duplicates of any "
         f"training pair). Beam size {args.beam}, length penalty α = {args.alpha}. Percentages are shown with "
         f"95% Wilson confidence intervals. Training-sample reproduction check: "
         + ", ".join(f"`{v}` {'OK' if ok else 'NOT VERIFIED'}" for v, ok in checks.items()) + ".\n\n"]

    s.append("## A. Speaker control\n\nSentences whose reference contains a gendered first-person form "
             "(-łam/-łem, -łabym/-łbym). Each is translated with every speaker token.\n\n" + head)
    s.append(row("`<self_f>` → feminine form", lambda m: fmt(m["A"]["f_correct"], m["A"]["n"])))
    s.append(row("`<self_m>` → masculine form", lambda m: fmt(m["A"]["m_correct"], m["A"]["n"])))
    s.append(row("**both correct on the same sentence**", lambda m: fmt(m["A"]["flip"], m["A"]["n"])))
    s.append(row("`<self_na>` → feminine", lambda m: share(m["A"]["na_dist"], "F", m["A"]["n"])))
    s.append(row("`<self_na>` → masculine", lambda m: share(m["A"]["na_dist"], "M", m["A"]["n"])))
    s.append(row("`<self_na>` → no gendered form", lambda m: share(m["A"]["na_dist"], "NONE", m["A"]["n"])))
    s.append(row("n", lambda m: str(m["A"]["n"])))

    s.append("\n## B. Neutral stability\n\nFirst-person sentences whose reference has *no* gendered form "
             "(e.g. *Chcę z tobą porozmawiać*). The speaker token should not change anything.\n\n" + head)
    s.append(row("identical output for f / m / na", lambda m: fmt(m["B"]["identical"], m["B"]["n"])))
    s.append(row("gendered form introduced by f or m token", lambda m: fmt(m["B"]["introduced"], m["B"]["n"])))
    s.append(row("n", lambda m: str(m["B"]["n"])))

    s.append("\n## C. Addressee control\n\nSentences whose reference contains a gendered/plural second-person "
             "past or conditional form (-łaś, -łeś, -liście/-łyście). Rows: token given; columns: form detected "
             "in the output. *Plural is measured on wy-forms only (not pan/pani).*\n")
    for v in vs:
        c, n = metrics[v]["C"]["confusion"], metrics[v]["C"]["n"]
        s.append(f"\n**`{v}`** (n = {n})\n\n| token | feminine | masculine | plural | none / mixed |\n|---|---|---|---|---|\n")
        for tok in ("f", "m", "p", "na"):
            d = c[tok]
            s.append(f"| `<addr_{tok}>` | {fmt(d.get('F', 0), n)} | {fmt(d.get('M', 0), n)} | "
                     f"{fmt(d.get('P', 0), n)} | {fmt(d.get('NONE', 0) + d.get('MIXED', 0), n)} |\n")

    s.append("\n## D. General translation quality\n\nSentences without first/second-person context, "
             "`<self_na> <addr_na>`, lowercased, sacreBLEU defaults.\n\n" + head)
    s.append(row("chrF", lambda m: str(m["D"]["chrF"])))
    s.append(row("BLEU", lambda m: str(m["D"]["BLEU"])))
    s.append(row("n", lambda m: str(m["D"]["n"])))

    s.append("\n## Caveats\n\n- The detectors only see past-tense and conditional endings, so test sentences are "
             "filtered to those forms; present-tense agreement (e.g. adjectives) is not measured.\n"
             "- \"No gendered form\" in A can also mean a valid paraphrase (e.g. present tense).\n"
             "- Held-out sentences come from the same OpenSubtitles distribution (in-domain).\n"
             "- Every individual output is in `eval_outputs.csv` for manual inspection.\n")
    path.write_text("".join(s), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "local_data/translator_v2/data_final_opensub.pkl"))
    ap.add_argument("--versions", nargs="+", default=["v1_5"])
    ap.add_argument("--n", type=int, default=200, help="sentences per category")
    ap.add_argument("--n-general", type=int, default=300)
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "research/results"))
    args = ap.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.data, "rb") as f:
        df = pickle.load(f).reset_index(drop=True)
    print(f"Loaded {len(df):,} rows from {args.data}")

    t0 = time.time()
    held, checks = build_heldout(df, args.versions)

    metrics, records = {}, []
    for v in args.versions:
        metrics[v] = run_version(v, held, args, records)

    (out_dir / "eval_metrics.json").write_text(
        json.dumps({"args": vars(args), "checks": checks, "heldout_size": len(held), "metrics": metrics},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out_dir / "eval_outputs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["version", "test", "ref_label", "self_tok", "addr_tok", "eng", "pol_ref", "pol_hyp", "detected"])
        w.writerows(records)
    write_report(metrics, checks, args, len(held), out_dir / "eval_summary.md")
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min -> {out_dir / 'eval_summary.md'}")


if __name__ == "__main__":
    main()
