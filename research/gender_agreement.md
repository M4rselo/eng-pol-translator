# Gender Agreement
 
## Problem
 
In Polish, verbs, adjectives, and participles need to match the subject’s gender. The model often has no reliable way to infer gender from context, so the agreement can end up inconsistent or completely wrong.

#### Three recurring patterns can be observed:

**1. Gender agreement within a single sentence** - Without a clear he/she marker, the model does not connect gender-specific words across the sentence.

> **EN:** I am a woman and I am proud of it.   
> **Model:** Jestem kobietą i jestem z tego **dumny**.  
> ***Expected:*** Jestem kobietą i jestem z tego **dumna**. 

> **EN:** As a husband, I believe that I am beautiful.   
> **Model:** Jako mąż uważam, że jestem **piękna**.   
> ***Expected:*** Jako mąż uważam, że jestem **piękny**.   

**2. Proper names treated as gender-neutral tokens** - Polish names carry strong gender information (Anna = female, John = male), but the model cannot reliably associate a name with a gender.

> **EN:** John gave Mary a book.      
> **Model:** John **dała** mary książkę.  
> ***Expected:*** John **dał** Mary książkę.  

> **EN:** Anna told her friend the truth.   
> **Model:** Anno **powiedział** jej przyjacielowi prawdę.   
> ***Expected:*** Anna **powiedziała** swojej przyjaciółce prawdę.   

> **EN:** Kate decided to leave early.   
> **Model:** Kate **zdecydowali** się wyjechać wcześnie.   
> ***Expected:*** Kate **zdecydowała** się wyjść wcześniej.   

> **EN:** Tom and Sarah got married last year.      
> **Model:** Tom i sarah **wyszła** za mąż w zeszłym roku.   
> ***Expected:*** Tom i Sarah **wzięli ślub** w zeszłym roku.   

> **EN:** The teacher asked her students to sit down.    
> **Model:** **Nauczyciel** poprosił jej studentów, by **usiadła**.    
> ***Expected:*** **Nauczycielka** poprosiła swoich uczniów, żeby usiedli.    

**3. First-person sentences without gender context** -
When there is no he/she or name to infer gender from, the output gender is unpredictable - it reflects training data distribution rather than any actual decision.

> **EN:** I have never been to Paris.   
> **Model:** Nigdy nie **byłam** w paryżu.   
> ***Expected:*** ambiguous - depends on the speaker   

> **EN:** I decided to quit my job.   
> **Model:** Postanowi**łem** przestać wykonywać swoją pracę.   
> ***Expected:*** ambiguous - but should at least be consistent with the sentence above   

---
 
## Solutions

### Solution v1 (Initial Implementation)
The first working solution focuses on first-person sentences, where grammatical gender cannot be inferred from the source sentence alone.

A corpus of first-person sentences was collected from *OpenSubtitles* dataset and manually classified into three groups. Feminine and masculine examples were identified based on grammatical endings (e.g. *-łam*, *-łabym* *-łem*, *-łbym*) together with sentence structure analysis. Sentences that could not be reliably assigned to either gender, or where grammatical gender is not expressed in Polish (e.g. *Lubię chodzić do kina.*/*Jesteś dla mnie niemiły.*), were placed into a separate neutral category.

The resulting dataset consisted of approximately:

- **20k** feminine first-person sentences,
- **20k** masculine first-person sentences,
- **60k** gender-neutral or unassigned first-person sentences.

Three special control tokens were introduced:

- `<self_f>` – feminine speaker
- `<self_m>` – masculine speaker
- `<self_na>` – gender-neutral

The control token is prefixed to the input sequence before the `<bos>` token, allowing the model to explicitly condition the translation on the speaker's grammatical gender when necessary.

#### Results and Limitations:
- The model correctly conditions its output on the provided gender token, selecting the appropriate grammatical forms.
```python
predicter.translate_snt("I have never been to Paris.", 'f')
> 'Nigdy nie byłam w paryżu.'
------------------------------------------------------------
predicter.translate_snt("I have never been to Paris.", 'm')  
> 'Nigdy nie byłem w paryżu.'
============================================================
predicter.translate_snt("I shouldn't be here.", 'f')
> 'Nie powinnam tu zostać.'
------------------------------------------------------------
predicter.translate_snt("I shouldn't be here.", 'm')
> 'Nie powinienem tu zostać.'
```
- The model appears to struggle with sentences where grammatical gender is not expressed in Polish: </br>
</br>This behavior is likely due to the lack of gender-conditioned examples where grammatical gender should remain unspecified. Since the model receives an explicit gender signal, it may overuse gender-specific forms even in contexts where Polish naturally uses a gender-neutral construction.
```python
predicter.translate_snt("I want to talk to you.", 'f')
> 'Chciałam z tobą porozmawiać.'
[Expected: Chcę z tobą porozmawiać]
------------------------------------------------------------
predicter.translate_snt("I want to talk to you.", 'm')  
> 'Chciałbym z tobą porozmawiać.'
[Partially correct, but the meaning shifts closer to "I would like to ..."]
------------------------------------------------------------
predicter.translate_snt("I want to talk to you.", 'na')
> 'Chcę z tobą porozmawiać.'
```
#### Sample Evaluation

A quick test was performed on three manually prepared test subsets (500 sentences each), selected before training from the data assigned to each reference token category. The table shows the number of sentences where the model generated the expected grammatical forms.

| Token | Test size | Correct sentences |
|-------|----------:|------------------:|
| `<self_f>` | 500 | 30 / 500 |
| `<self_m>` | 500 | 18 / 500 |
| `<self_na>` | 500 | 50 / 500 |

Due to limitations in the current training data structure, the model struggles with gender-specific tokens. `<self_na>` achieves better results, while `<self_f>` and especially `<self_m>` remain less reliable.



