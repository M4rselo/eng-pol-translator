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
...
