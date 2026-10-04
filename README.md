# Gender Context English-Polish Translator

A sequence-to-sequence neural machine translator built from scratch in *PyTorch*, without using any pre-trained models or high-level NLP libraries - with explicit control over the speaker's and addressee's gender and number in the Polish output.

---

## Motivation

Most translation tools we use work sentence by sentence, with no memory of who's speaking or who's being spoken to. That's rarely a problem in English, but Polish is a heavily gendered language: verbs, adjectives, and even some nouns change form depending on the speaker's and listener's gender and number.

Polish is also going through a genuinely interesting linguistic moment right now: feminine forms of nouns (*feminatywy*), such as **członkini** instead of **członek** are becoming more common - and, depending on who you ask, more accepted... This isn't entirely new territory either: forms like **nauczycielka** ("female teacher") have long been completely standard and uncontroversial.

None of this is something a context-free translator can reason about, and the results can range from mildly inaccurate to unintentionally funny. </br>
<br>

#### As an example, here's a real chain of attempts with&nbsp;<a href="https://translate.google.com"><img src="https://img.shields.io/badge/Google%20Translate-white?logo=googletranslate&logoColor=4285F4" height="32" align="top"></a>&nbsp;, trying to coax a single sentence into coming out right:
---

```diff
─┬─ Google Translate ───────────────────────────────────
 │ [EN] >> I knew you liked me.
+│ [PL] >> Wiedziałem, że mnie lubisz.
─┴──────────────────────────────────────────────────────
```

- Looks right. But what if "you" is meant to be plural? </br>
<br>

```diff
─┬─ Google Translate ───────────────────────────────────
 │ [EN] >> I knew you (guys) liked me.
+│ [PL] >> Wiedziałem, że mnie lubicie.
─┴──────────────────────────────────────────────────────
```

- Perfect. Now, what if the speaker is female? </br>
<br>

```diff
─┬─ Google Translate ───────────────────────────────────
 │ [EN] >> I (girl) knew you (guys) liked me.
-│ [PL] >> Wiedziałem, że mi się podobam (dziewczyno).
─┴──────────────────────────────────────────────────────
```

- Hmm..., now it treated *"(girl)"* as a part of the sentence, not as a context - "wiedziałem" is still the masculine form. I'm sure dropping  *"(guys)"* will do the job!</br>
<br>

```diff
─┬─ Google Translate ───────────────────────────────────
 │ [EN] >> I (girl) knew you liked me.
-│ [PL] >> Wiedziałem, że ci się podobam.
─┴──────────────────────────────────────────────────────
```

- Nope, same problem, still masculine. One more try, rephrasing more aggressively this time. </br>
<br>

```diff
─┬─ Google Translate ───────────────────────────────────
 │ [EN] >> I (as a girl) knew you guys liked me.
+│ [PL] >> Jako dziewczyna wiedziałam, że mnie lubicie.
─┴──────────────────────────────────────────────────────
```

- There it is - correct speaker gender, correct addressee plurality. But getting there took mangling the English sentence into something nobody would naturally write, purely to smuggle context through a translator that has no way to ask for it directly. </br>
<br>

Of course, this particular example is trivial, and you could just fix "wiedziałem" to "wiedziałam" by hand in a second. But scale it up to a whole document, a story, a hundred-line chat log with this happening throughout, and manual correction stops being just a "quick fix". That's the actual problem this project explores: ***What if the translator itself took the speaker's gender, and whether the listener is a man, a woman, or a group, as explicit input, instead of guessing?***

---
## Approach

*NOTE: This problem is absolutely solvable more easily by prompting an LLM, or by fine-tuning an existing SOTA translation model. This repo
deliberately does neither. The goal wasn't to build the most practical tool, but to understand the problem end to end - from raw subtitle data to a model
that can be told who's speaking and who's being spoken to.*

### 1. Context as part of the target sequence
---
The model is a standard encoder-decoder Transformer. The English sentence goes into the encoder unchanged - the context is injected on the **decoder
side**, as two reference tokens placed before `<bos>`:

```
<self_f> <addr_p> <bos> Wiedziałam, że się mylicie. <eos>
```
<br>
  
There are seven reference tokens in total - three for the speaker and four for the listener: <br>
  
<div align="center">
    
| Token | Meaning |
|---|---|
| `<self_f>` · `<self_m>` · `<self_na>` | female · male · unspecified speaker |
| `<addr_f>` · `<addr_m>` · `<addr_p>` · `<addr_na>` | female · male · plural · unspecified listener |
   
</div>

<br>
During inference, these two tokens are fed in as a fixed prefix and the model generates the rest of the sentence conditioned on them. Since the context
lives outside the English input, it can be changed without touching a single word of the source.

### 2. Labelling the data by grammar
---
No gender-annotated English-Polish corpus exists, so the labels were derived from the Polish side of *OpenSubtitles*. First and second-person sentences
were classified by their grammatical endings - *-łam / -łem*, *-łabym / -łbym* for the speaker, *-łaś / -łeś*, *pan / pani* and *wy*-forms for the
listener, with additional sentence-structure checks to filter out false matches.

The first-person extraction alone produced **~68k feminine** and **~161k masculine** sentences. The imbalance was even stronger for profession nouns
(*jestem lekarzem*, *jestem nauczycielem*), so a large set of them was converted to feminine forms (*lekarzem → lekarką*, *prawnikiem → prawniczką*),
teaching the model the feminine forms the raw subtitles barely contain. <br>

> [!TIP]
  > The extraction process, including the exact patterns and filtering rules, can be found in
  [`research/Gender_Pronouns_1st_Person.ipynb`](research/Gender_Pronouns_1st_Person.ipynb) (speaker) and
  [`research/Gender_Pronouns_2nd_Person.ipynb`](research/Gender_Pronouns_2nd_Person.ipynb) (listener).

### 3. Knowing when not to change anything
---
Most sentences don't express gender at all, e.g., *Lubię chodzić do kina.* (*I like going to the cinema.*). Training only on gendered examples made the model overuse gendered forms everywhere
- an early version translated *"I want to talk to you."* with `<self_f>` as *"Chciałam z tobą porozmawiać"* (past tense, just to make it feminine).

The fix is a **reference-token augmentation** applied on the fly during training *(translator-v1_5)*. Whenever a sentence has no gender label for the speaker or the listener, its context token is replaced with one drawn at random each time the sample is loaded - `<self_f>`, `<self_m>` or `<self_na>` for the speaker, and `<addr_f>`, `<addr_m>` or `<addr_na>` for the listener. Across epochs, the same sentence is seen under different tokens, but always with the same Polish target.

---

The same sentence from the Google Translate example, this time without rewriting a single word:

```diff
─┬─ Translator-v1_5 ────────────────────────────────────
 │ [EN]  >> I knew you liked me.
 │ [TOK] >> speaker: <self_f> · listener: <addr_p>
+│ [PL]  >> Wiedziałam, że mnie lubicie.
─┴──────────────────────────────────────────────────────
```

The full story of how the gender handling evolved, including evaluation and failure cases, is documented in
[`research/gender_agreement.md`](research/gender_agreement.md).

<br>

---
## Try it

<div align="center">
  <img src="TODO/path/to/screenshot.png" alt="Web demo" width="800">
</div>

<br>

The repo comes with a small web app for playing with the model: pick the speaker and listener context, compare all token combinations side by side, and
inspect what the model is actually doing - per-word confidence, attention maps and the BPE tokenization of the input.

### 1. Install

```bash
git clone https://github.com/TODO/eng-pol-translator.git
cd eng-pol-translator
pip install -r requirements.txt
```

### 2. Download the model

The trained weights are too large for the repo and are hosted on [Hugging Face](TODO-link). Download them and place them in `appdata/`:

```
appdata/
├── checkpoints/
│   └── translator_v1_5-13.pt
└── model_reference/
└── translator_v1_5/
```

```bash
TODO: download command
```

### 3. Run

```bash
python webapp/app.py
```

The app will be available at [http://localhost:5000](http://localhost:5000). Inference runs on the CPU - no GPU required.

---

