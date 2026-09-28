# Gender Context English-Polish Translator

A sequence-to-sequence neural machine translator built from scratch in *PyTorch*, without using any pre-trained models or high-level NLP libraries — with explicit control over the speaker's and addressee's gender and number in the Polish output.

---

## Motivation

Most translation tools we use day to day work sentence by sentence, with no memory of who's speaking or who's being spoken to. That's rarely a problem in English, but Polish is a heavily gendered language: verbs, adjectives, and even some nouns change form depending on the speaker's and listener's gender and number.

Polish is also going through a genuinely interesting linguistic moment right now: feminine forms of nouns (*feminatywy*), such as **członkini** instead of the traditionally default **członek**, are becoming more common — and, depending on who you ask, more accepted. This isn't entirely new territory either: forms like **nauczycielka** ("female teacher") have long been completely standard and uncontroversial. What's changing is how far that pattern extends into roles and titles that used to default to the masculine form.

None of this is something a context-free translator can reason about, and the results can range from mildly inaccurate to unintentionally funny. Here's a real chain of attempts with Google Translate, trying to coax a single sentence into coming out right:

```
EN: I knew you were wrong.
PL: Wiedziałem, że się mylisz.
```

Looks right. But what if "you" means more than one person?

```
EN: I knew you (guys) were wrong.
PL: Wiedziałem, że się mylicie.
```

Still right. Now, what if the speaker is a girl?

```
EN: I (girl) knew you (guys) were wrong.
PL: Wiedziałem (dziewczyny), że się mylicie.
```

Not quite — "wiedziałem" is still the masculine form, and Google just stapled "(dziewczyny)" onto the sentence without doing anything with it. Maybe dropping "(guys)" fixes it?

```
EN: I (girl) knew you were wrong.
PL: Wiedziałem (dziewczyno), że się mylisz.
```

Nope, same problem, still masculine. One more try, rephrasing more aggressively this time:

```
EN: I (as a girl) knew you (guys) were wrong.
PL: Jako dziewczyna, wiedziałam, że się mylicie.
```

There it is — correct speaker gender, correct addressee plurality. But getting there took mangling the English sentence into something nobody would naturally write, purely to smuggle context through a translator that has no way to ask for it directly.

This particular example is trivial — you could just fix "wiedziałem" to "wiedziałam" by hand in five seconds. But scale it up to a whole document, a story, a hundred-line chat log with this happening throughout, and manual correction stops being trivial fast. That's the actual problem this project explores: what if the translator itself took speaker/addressee gender and number as explicit input, instead of guessing — or not even trying?

*A quick disclaimer: this problem is absolutely solvable more easily by prompting an LLM, or by fine-tuning an existing state-of-the-art translation model. Neither is what this repo does. Everything here — tokenizer, model, beam search — is built from scratch in PyTorch, without any pre-trained weights, for the sake of understanding the problem end to end rather than reaching for the most practical tool. It isn't meant to be a revolutionary MT system; it's a from-scratch exploration of a genuinely interesting corner of the problem.*
