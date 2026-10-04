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
 │ [EN] >> I knew you were wrong.
+│ [PL] >> Wiedziałem, że się mylisz.
─┴──────────────────────────────────────────────────────
```

- Looks right. But what if "you" is meant to be plural? </br>
<br>

```diff
─┬─ Google Translate ───────────────────────────────────
 │ [EN] >> I knew you (guys) were wrong.
+│ [PL] >> Wiedziałem, że się mylicie.
─┴──────────────────────────────────────────────────────
```

- Perfect. Now, what if the speaker is female? </br>
<br>

```diff
─┬─ Google Translate ───────────────────────────────────
 │ [EN] >> I (girl) knew you (guys) were wrong.
-│ [PL] >> Wiedziałem (dziewczyny), że się mylicie.
─┴──────────────────────────────────────────────────────
```

- Hmm..., now it treated *"(girl)"* as a part of the sentence, not as a context - "wiedziałem" is still the masculine form. I'm sure dropping  *"(guys)"* will do the job!</br>
<br>

```diff
─┬─ Google Translate ───────────────────────────────────
 │ [EN] >> I (girl) knew you were wrong.
-│ [PL] >> Wiedziałem (dziewczyno), że się mylisz.
─┴──────────────────────────────────────────────────────
```

- Nope, same problem, still masculine. One more try, rephrasing more aggressively this time. </br>
<br>

```diff
─┬─ Google Translate ───────────────────────────────────
 │ [EN] >> I (as a girl) knew you (guys) were wrong.
+│ [PL] >> Jako dziewczyna, wiedziałam, że się mylicie.
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

Everything below is implemented from scratch in *PyTorch*, with no pre-trained weights:

- **Data** — English-Polish pairs from *OpenSubtitles*, plus first- and second-person sentences extracted and labelled by grammatical gender using Polish
verb endings (*-łam / -łem*, *-łaś / -łeś*, *pan / pani*, *wy*-forms).
- **Tokenizer** — a custom BPE tokenizer, trained separately for English and Polish.
- **Model** — an encoder-decoder Transformer (multi-head attention, positional encoding, residual connections, layer norm).
- **Context control** — two control tokens are prefixed to the decoder input, one for the speaker and one for the addressee:
  `<self_f | self_m | self_na>` `<addr_f | addr_m | addr_p | addr_na>` `<bos> ...`
  During training, sentences without a gender label get a random context token, so the model learns to change the output *only when Polish actually has a
grammatical choice to make*.
- **Inference** — a custom beam search with a length penalty and a decoder cache.

The full story of how the gender handling evolved (v1 → v3), including evaluation and failure cases, is documented in
[`research/gender_agreement.md`](research/gender_agreement.md).

---

## Results

The same English sentence, translated under different speaker contexts:

```
EN: I would like to talk to you.
─────────────────────────────────────────────
<self_f>   Chciałabym z tobą porozmawiać.
<self_m>   Chciałbym z tobą porozmawiać.
<self_na>  Chcę z tobą porozmawiać.
```

```
EN: I went to the store, bought bread, and came back home.
─────────────────────────────────────────────
<self_f>   Pojechałam do sklepu, kupiłam chleb i wróciłam do domu.
<self_m>   Poszedłem do sklepu, kupiłem chleb i wróciłem do domu.
```

And the sentence from the motivation, this time with no rephrasing needed:

```
EN: I knew you were wrong.
─────────────────────────────────────────────
<self_f> <addr_m>   TODO
<self_f> <addr_p>   TODO
<self_m> <addr_f>   TODO
<self_m> <addr_p>   TODO
```

Compared with the first version, the refined data in v2 roughly tripled the exact-match rate on gendered sentences (e.g. `<self_f>`: 29 → 88 / 596) and
raised mean BLEU from 0.22 to 0.43. Full tables are in [`research/gender_agreement.md`](research/gender_agreement.md).

