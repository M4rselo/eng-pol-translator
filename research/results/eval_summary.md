# Gender-control evaluation

Held-out pool: 1,132,758 sentence pairs never sampled for training (and not duplicates of any training pair). Beam size 5, length penalty α = 0.6. Percentages are shown with 95% Wilson confidence intervals. Training-sample reproduction check: `v1_4` OK, `v1_5` OK.

## A. Speaker control

Sentences whose reference contains a gendered first-person form (-łam/-łem, -łabym/-łbym). Each is translated with every speaker token.

| | `v1_4` | `v1_5` |
|---|---|---|
| `<self_f>` → feminine form | 80.8% (77–84) | 76.5% (72–80) |
| `<self_m>` → masculine form | 88.5% (85–91) | 86.8% (83–90) |
| **both correct on the same sentence** | 78.5% (74–82) | 75.5% (71–79) |
| `<self_na>` → feminine | 7.5% (5–11) | 35.8% (31–41) |
| `<self_na>` → masculine | 6.2% (4–9) | 26.8% (23–31) |
| `<self_na>` → no gendered form | 86.2% (83–89) | 36.0% (31–41) |
| n | 400 | 400 |

## B. Neutral stability

First-person sentences whose reference has *no* gendered form (e.g. *Chcę z tobą porozmawiać*). The speaker token should not change anything.

| | `v1_4` | `v1_5` |
|---|---|---|
| identical output for f / m / na | 48.0% (41–55) | 74.5% (68–80) |
| gendered form introduced by f or m token | 12.5% (9–18) | 6.5% (4–11) |
| n | 200 | 200 |

## C. Addressee control

Sentences whose reference contains a gendered/plural second-person past or conditional form (-łaś, -łeś, -liście/-łyście). Rows: token given; columns: form detected in the output. *Plural is measured on wy-forms only (not pan/pani).*

**`v1_4`** (n = 600)

| token | feminine | masculine | plural | none / mixed |
|---|---|---|---|---|
| `<addr_f>` | 73.0% (69–76) | 1.2% (1–2) | 0.2% (0–1) | 25.7% (22–29) |
| `<addr_m>` | 0.0% (0–1) | 72.3% (69–76) | 0.0% (0–1) | 27.7% (24–31) |
| `<addr_p>` | 0.7% (0–2) | 1.0% (0–2) | 61.7% (58–65) | 36.7% (33–41) |
| `<addr_na>` | 3.3% (2–5) | 5.7% (4–8) | 1.3% (1–3) | 89.7% (87–92) |

**`v1_5`** (n = 600)

| token | feminine | masculine | plural | none / mixed |
|---|---|---|---|---|
| `<addr_f>` | 62.0% (58–66) | 3.0% (2–5) | 0.3% (0–1) | 34.7% (31–39) |
| `<addr_m>` | 0.7% (0–2) | 62.8% (59–67) | 0.2% (0–1) | 36.3% (33–40) |
| `<addr_p>` | 5.2% (4–7) | 1.0% (0–2) | 60.3% (56–64) | 33.5% (30–37) |
| `<addr_na>` | 23.5% (20–27) | 28.5% (25–32) | 0.3% (0–1) | 47.7% (44–52) |

## D. General translation quality

Sentences without first/second-person context, `<self_na> <addr_na>`, lowercased, sacreBLEU defaults.

| | `v1_4` | `v1_5` |
|---|---|---|
| chrF | 41.3 | 41.4 |
| BLEU | 20.6 | 21.0 |
| n | 300 | 300 |

## Caveats

- The detectors only see past-tense and conditional endings, so test sentences are filtered to those forms; present-tense agreement (e.g. adjectives) is not measured.
- "No gendered form" in A can also mean a valid paraphrase (e.g. present tense).
- Held-out sentences come from the same OpenSubtitles distribution (in-domain).
- Every individual output is in `eval_outputs.csv` for manual inspection.
