# English-Polish Neural Machine Translator

A sequence-to-sequence neural machine translator built from scratch in *PyTorch*, without using any pre-trained models or high-level NLP libraries.

---

## About

- **Transformer architecture** - encoder-decoder with multi-head attention, positional encoding, residual connections, and layer normalization
- **BPE tokenizer** - Byte Pair Encoding implemented using a priority queue, trained separately for English (24k vocab) and Polish (48k vocab)
- **Training pipeline** - custom training loop with mixed precision (fp16), gradient clipping, Xavier weight initialization, and checkpoint saving per epoch
- **Beam search** - custom implementation with length penalty (α = 0.6) and decoder cache for faster inference

**The project is currently in active development. Core translation functionality works well; additional features and improvements are ongoing.**

---

## Architecture
*Current model - base translator (v1)*
| Parameter | Value |
|---|---|
| Model type | Encoder-Decoder Transformer |
| Hidden size (d_model) | 512 |
| Encoder/Decoder blocks | 4 |
| Attention heads | 8 |
| FFN hidden size | 1024 |
| Dropout | 0.3 |
| English vocab size | 24,000 |
| Polish vocab size | 48,000 |
| Max sequence length | 51 tokens |

---
## Data

Trained on 882,569 pairs / validated on 109,082 pairs. 

| Dataset | Style | Source |
|---|---|---|
| Europarl | Formal | European Parliament proceedings |
| OpenSubtitles | Colloquial | Movie and TV subtitles |

**No test set yet - evaluation metrics (BLEU) are planned for a future update.**

---
## Examples

```
Eng-Pol Translator ready. Type 'quit' to exit.

EN: Although it was raining, we decided to go for a walk.
PL: Chociaż padało, postanowiliśmy iść na spacer.

EN: My mother-in-law visited us last weekend.
PL: Moja teściowa odwiedziła nas w miniony weekend.

EN: Member states are required to implement the directive within two years.
PL: Państwa członkowskie muszą wdrożyć dyrektywę w ciągu dwóch lat.

EN: We need couple things: - a tent, - a map, - and water.
PL: Potrzebujemy kilku rzeczy: namiotu, mapy i wody.

EN: - No, I am not ready yet, - he replied quietly.
PL: - nie, jeszcze nie jestem gotowy, odpowiedział spokojnie.

EN: We must ensure that human rights are protected across all member states.
PL: Musimy zapewnić ochronę praw człowieka we wszystkich państwach członkowskich.

EN: quit
```

---

## Setup & Usage

**Requirements:** Python 3.10+, PyTorch, pandas, numpy, tqdm (see `requirements.txt`)

**1. Clone the repository**
```bash
git clone https://github.com/M4rselo/eng-pol-translator
cd eng-pol-translator
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download model weights**

Download `predicter_1.pkl` (serialized model + tokenizers) from [Google Drive](https://drive.google.com/file/d/1aGExk4IfDADB9nFfD842lkzHs7UGq7Ye/view?usp=sharing) and place it in `data/predicter/`.

**4. Run the translator**
```
python modules/translate_v1.py
```

```
Eng-Pol Translator ready. Type 'quit' to exit.

EN: Have you ever been to Paris?
PL: Byłeś kiedyś w paryżu?

EN: quit
```

---

## Planned & In Progress

- **Data analysis notebook** - notebook walking through the full data pipeline, from raw TSV files to model-ready tensors: what the data looked like, what got cleaned and why, how tokenization affects vocabulary coverage

- **BLEU evaluation** - automated scoring on the validation set, broken down by sentence type (simple, complex, formal)

- **Handling typos** - right now a misspelled word often breaks the translation. planning to fix this with data augmentation during training so the model learns to deal with imperfect input

- **Gender agreement** - the model struggles with speaker/addressee gender in Polish ("szłam" vs "szedłem"). improving this requires better context inference from the surrounding sentence

- **Style & tone conditioning** - the idea is to let the user provide a few example sentences and have the model pick up on the style and apply it to the translation, without any retraining

- **Unique sequence handling** - URLs, long numbers, email addresses - things that don't fit standard vocabulary and currently get mangled

---

## Known Limitations

- **Typos** - a single misspelled word can significantly affect translation quality
```
EN: I visited my grandfathher.
PL: Odwiedziłem mój grnkundl.

EN: It is neccessary to sign this form.
PL: To eceuccio, aby podpisać tę formularz.
```
- **Grammatical gender** - the model does not consistently infer the gender of the speaker or subject from context. This affects verb and adjective agreement in Polish, particularly in first-person statements and sentences involving proper names
```
EN: John gave Mary a book.
PL: John dała mary książkę.

EN: If I had known about the problem earlier, I would have fixed it.
PL: Gdybym wiedziała wcześniej o problemie, naprawiłbym go.
```
- **Idioms** - model struggles with non-literal expressions, translating them word-for-word
```
EN: I'll speak straight from the shoulder.
PL: Porozmawiam prosto z ramię.

EN: He feels under the weather.
PL: Czuje się pod wodą.
```
- **Unique sequences** - numbers, URLs, and other rare character sequences outside standard vocabulary tend to get mangled or omitted during translation
```
EN: It costs 235660$.
PL: Kosztuje 25656000 $.

EN: My nickname is M4rselo.
PL: Nazywam się m4see.
```
---
## References
P. Lison and J. Tiedemann, 2016, <a href="http://stp.lingfil.uu.se/~joerg/paper/opensubs2016.pdf"><i>OpenSubtitles2016: Extracting Large Parallel Corpora from Movie and TV Subtitles.</i></a> In Proceedings of the 10th International Conference on Language Resources and Evaluation (LREC 2016)<br/> 
<br/> P. Koehn, 2005, Europarl: A Parallel Corpus for Statistical Machine Translation. MT Summit
