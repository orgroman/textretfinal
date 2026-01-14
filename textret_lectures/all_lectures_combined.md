


================================================================================
FILE: Entity-Based Relevance Feedback for Document Retrieval.md
================================================================================


Below is a **full, faithful Markdown conversion** of
**“Entity-Based Relevance Feedback for Document Retrieval”**
(Lecture 11 – Based on ICTIR paper).
I did **not omit any content**; equations are rendered in **LaTeX**, figures are described textually, and slide structure is preserved.
Source: uploaded PDF 

---

# Text Retrieval and Search Engines

## Lecture 11 – Entity-Based Relevance Feedback

*Semester 1 – 2026*
*Based on work presented at the ICTIR conference*

---

## What Is an Entity?

**Example document:**

```
<DOCNO> FT921-7107 </DOCNO>
It will allow astronomers to explore three
quarters of the universe and study galaxies
perhaps as far as 14bn light years away.
```

### Entity Annotations

* **Entity name**: Universe

  * Entity ID: 31880
  * ( \Phi_{\text{universe}} = 0.217 )

* **Entity name**: Research

  * Entity ID: 25524
  * ( \Phi_{\text{study}} = 0.085 )

Where:

* ( \Phi ) = **confidence score** of the entity linker

*(Slide figure shows highlighted mentions “universe”, “study galaxies”, “light years” with Wikipedia entity panels.)*

---

## Our Relevance Feedback Framework

### Definitions

* ( L_{\text{init}} ): initial list of top-ranked ( N ) documents
* ( q ): query
* ( w ): term
* ( e ): entity

### Conceptual Flow

```
Information Need
      ↓
      q
      ↓
 L_init = {d1, d2, ..., dN}
```

---

## Extended Framework: Terms and Entities

From ( L_{\text{init}} ):

* Extract **terms**
* Extract **entities**

We assess relevance at two levels:

* **Term relevance**
  ( w_1, w_2, w_3, \dots )

* **Entity relevance**
  ( e_1, e_2, e_3, \dots )

Both are fed into a retrieval method to produce the **final document ranking**.

---

## STLM: Soft Confidence-Level Thresholding Language Model

*(Raviv et al., 2016)*

### Purpose

STLM is used to produce the initial ranked list ( L_{\text{init}} ).

---

### Token Space

Define the token set:

[
T \triangleq V \cup E
]

Where:

* ( V ): set of terms
* ( E ): set of entities

---

### Pseudo-Count Definition

For token ( t ) in text ( x ):

[
pc(t, x) \triangleq
\begin{cases}
\lambda \cdot tf(t, x) & \text{if } t \in V \
(1 - \lambda)\sum_{m \in M_x : e_m = t} \Phi(m) & \text{if } t \in E
\end{cases}
]

Where:

* ( tf(t, x) ): term frequency
* ( M_x ): set of entity mentions in ( x )
* ( e_m ): entity mapped by mention ( m )
* ( \Phi(m) ): confidence score
* ( \lambda ): interpolation parameter

---

### Pseudo-Length

[
pl(x) \triangleq \sum_{t \in T} pc(t, x)
]

---

## Token Selection Strategies

### T-RM#3 — Term-Based Relevance Model

* ( \lambda = 1 )
* Only terms are used

[
pc(t, x) = tf(t, x)
]

---

### E-RM#3 — Entity-Based Relevance Model

* ( \lambda = 0 )
* Only entities are used

[
pc(t, x) = \sum_{m \in M_x : e_m = t} \Phi(m)
]

---

## Presentation Approaches

### Term Presentation

* Term only
  *(Kelly & Fu ’06; Tan et al. ’07)*
* Term + sentence from ( L_{\text{init}} )
  *(Kelly & Fu ’06)*

### Entity Presentation

* Entity only
* Entity + sentence from ( L_{\text{init}} )
* Entity + abstract from the entity’s Wikipedia page

---

### Entity Linking

* **Toolkit**: TagMe
  *(Ferragina & Scaiella ’10)*
* **Data**: Wikipedia dump (July 2014)

---

### Human Relevance Judgments

* Platform: Amazon Mechanical Turk (HIT)
* Majority vote: **3 votes per query–token pair**
* Quality checks applied

---

## Example HIT

**ROBUST query #607**:

> *“human genetic code”*

Annotators are asked:

> *Are the entities relevant to the following topic?*

Examples shown:

* Entity: Blood → Relevant / Not relevant
* Entity: DNA profiling → Relevant / Not relevant
* Entity: Virus → Relevant / Not relevant

---

## The ETReF Dataset

**Entity and Term Relevance Feedback Dataset**
🔗 [https://github.com/Eilons/ETReF](https://github.com/Eilons/ETReF)

### Statistics

| Collection  | Presentation | #Tokens | % Relevant |
| ----------- | ------------ | ------- | ---------- |
| ROBUST      | Term only    | 12,740  | 39.0       |
| ROBUST      | Entity only  | 12,968  | 49.9       |
| ClueWeb09-B | Term only    | 10,156  | 35.1       |
| ClueWeb09-B | Entity only  | 10,352  | 30.6       |

**Examples**

* Query: *“nobel prize winners”*

  * Term: *award*, *professor*
  * Entity: *Particle physics*, *Buckingham Palace*

---

## Interpolated Token Feedback (ITF)

### Definitions

* ( T_r ): set of relevant tokens
* ( \delta_t = 1 ) if ( t \in T_r ), else 0

### ITF Query Model

[
p_q^{\text{ITF}}(t)
===================

\beta \frac{pc(t, q)}{pl(q)}
+
(1 - \beta)\frac{\delta_t}{|T_r|}
]

Where:

* ( \beta ): free parameter

---

### T-ITF (Term-Based ITF)

* ( \lambda = 1 )
* ( T_r ): relevant terms
* ( pc(t, x) = tf(t, x) )

---

### E-ITF (Entity-Based ITF)

* ( \lambda = 0 )
* ( T_r ): relevant entities
* ( pc(t, x) = \sum_{m \in M_x : e_m = t} \Phi(m) )

---

## Reciprocal Rank Fusion (RRF)

### Motivation

Combine **term** and **entity** feedback.

### Fusion Formula

[
F!-!X(q, d)
===========

\lambda \cdot \text{score}*{L*{T-X}}(d)
+
(1 - \lambda) \cdot \text{score}*{L*{E-X}}(d)
]

Where:

* ( X \in { RM#3, ITF } )
* Example: **F-ITF** = fusion of T-ITF and E-ITF

---

### Rank-to-Score Transformation

*(Cormack et al., 2009)*

[
\text{Score}_L(d) \triangleq \frac{1}{v + r_L(d)}
]

Where:

* ( r_L(d) ): rank of ( d ) in list ( L )
* ( v ): free parameter

---

## Pseudo vs. True Relevance Feedback

### Findings (ClueWeb, MAP)

1. **True relevance feedback** outperforms automatic query expansion
2. **Term relevance feedback** > automatic term expansion
3. **Entity relevance feedback** ≈ automatic entity expansion

*(Asterisks denote statistically significant differences.)*

---

## Effect of Feedback Token

### Findings

1. Entity relevance feedback is **inferior to term feedback**
2. Integrating term + entity feedback yields **significant gains**

---

## Init vs. True Relevance Feedback

### Results (ClueWeb, MAP)

* T-ITF **outperforms Init**
* E-ITF **underperforms Init**
* **F-ITF is best overall**

---

## Difficult Queries Analysis

**Queries with no relevant doc in top-10 Init**
(45 ClueWeb queries)

### Finding

> Entity relevance feedback is **especially effective for difficult queries**

Metrics shown:

* MAP
* P@5

---

## Boosters vs. Poison Pills

### Definitions

* **Booster**: relevant token that improves performance when added alone
* **Poison pill**: relevant token that degrades performance when added alone

### Statistics (ClueWeb)

| Tokens   | % Poison Pills | % Boosters |
| -------- | -------------- | ---------- |
| Terms    | 26.5           | 73.5       |
| Entities | 23.3           | 76.7       |

Aligned with prior work:

* Anick ’03
* Nemeth et al. ’04
* Ruthven ’95
* Tan et al. ’07

---

## Summary

* First use of **true entity relevance feedback** for document retrieval
* Introduced the **ETReF dataset**
* Developed LM-based relevance feedback methods:

  * Term feedback > entity feedback (overall)
  * Entity feedback **very effective for difficult queries**
* **Integrating term and entity feedback is highly effective**

---

If you want next, I can:

* Map **this paper directly to RM3 / PRF** you’re using in the Kaggle HW
* Extract **implementation recipes** (what is feasible without true relevance)
* Show how to **approximate entity feedback** using LLMs or NER in your Streamlit app

---

## Kaggle-Feasible Variants (No True Relevance)

Constraints: no external data or APIs; use only the provided Pyserini index, Wikipedia corpus, and the fixed Llama-3.2-1B model.

1. **Init (Pyserini BM25/QL)** — Use BM25 or QL with Dirichlet to build L_init; keep `k1`/`b` defaults or mild grid search.
  *runtime: fast · complexity: simple · rules: compliant*

2. **T-RM3 (Terms Only)** — Pyserini RM3 over top-20 docs; re-rank with feedback terms only (no entities); tune `fbDocs`, `fbTerms`, `originalQueryWeight`.
  *runtime: medium · complexity: simple · rules: compliant*

3. **Entity-Lite RM3** — Approximate entities by capitalized noun phrases or Wikipedia titles appearing in top docs; treat them as tokens in RM3 feedback; cap phrases to 1–3 words to avoid noise.
  *runtime: medium · complexity: medium · rules: compliant*

4. **F-RM3 (Term + Entity-Lite Fusion)** — Build separate RM3 lists: terms-only and entity-lite; fuse with Reciprocal Rank Fusion (small `v`, `lambda≈0.5`).
  *runtime: medium · complexity: medium · rules: compliant*

5. **ITF-Lite (Token Interpolation)** — From the top-20 docs, keep the top-k capitalized phrases plus top-k terms; interpolate counts with `beta` to balance query terms vs. feedback tokens.
  *runtime: medium · complexity: medium · rules: compliant*

6. **Booster Filtering** — Before feedback, drop feedback tokens that never appear in the top-5 docs (proxy for poison pills); optionally upweight tokens shared by ≥2 of the top-5.
  *runtime: fast · complexity: simple · rules: compliant*



================================================================================
FILE: IR Challenge-Intro.md
================================================================================


<!-- Slide number: 1 -->
1
IR Challenge

Goal: achieving the best MAP performance
1000 documents per query

ROBUST collection
Newswire collection
Considered relatively homogeneous

249 queries
50 – train + qrels
199 – test

| Topics | # of Docs | TREC disks | Collection |
| --- | --- | --- | --- |
| 301-450 601-700 | 528,155 | Disks 4&5-CR | ROBUST |

<!-- Slide number: 2 -->
2
IR Challenge

Submission
Up to 3 runs
Standard TREC 6 columns format

![תמונה 8](תמונה8.jpg)

| Method Name | Score | Rank | Doc No | Iteration(Unused) | Query ID |
| --- | --- | --- | --- | --- | --- |
| indri | -5.0 | 1 | 1410 | Q0 | 1 |
| indri | -5.5 | 2 | 1572 | Q0 | 1 |



================================================================================
FILE: Lecture_10_RM.md
================================================================================


Below is a **clean Markdown conversion** of **Lecture 10 – Relevance Models**, with **LaTeX used for all math**, structured for study/reference.
(Source: uploaded PDF )

---

# Text Retrieval and Search Engines

## Lecture 10 – Relevance Models (RM)

*Semester 1 – 2026*
(Some slides adapted from *Introduction to Information Retrieval* — Manning, Raghavan, Schütze; and Victor Lavrenko)

---

## Motivation: The Probabilistic Ranking Principle (PRP)

The **PRP** states that documents should be ranked by their probability of relevance:

[
p(R \mid D, q) \propto \frac{P(R \mid D, q)}{P(NR \mid D, q)}
]

Under standard independence assumptions, this yields:

[
\frac{P(R \mid D, q)}{P(NR \mid D, q)}
= \prod_{w \in D}
\frac{p_w (1 - r_w)}{r_w (1 - p_w)}
]

Where:

* ( p_w ): probability of term ( w ) in the **background model**
* ( r_w ): probability of term ( w ) in the **relevance model**

---

## Background vs. Relevance Models

* **Background model**

  * Estimated from the *entire document collection*
* **Relevance model**

  * Estimated from a *set of known relevant documents*

### Problem

What if **no relevant documents** are available?

---

## Key Assumption

When no relevance judgments are available:

> **Assume the query ( q ) is a random sample from the relevance model ( R ).**

Example query:

> *“hubble telescope achievements”*

---

## Estimation: The Sampling Game

Assumptions:

* There exists an **unknown relevance language model** ( R )
* Both:

  * Query terms ( q_1, \dots, q_n )
  * Relevant document terms
    are samples from ( R )

### Question

Given observed query terms:
[
q = (\text{hubble}, \text{telescope}, \text{achievements})
]

What is the probability that the next sampled word is *earth*?

[
P(\text{earth} \mid R)
\approx
P(\text{earth} \mid \text{hubble}, \text{telescope}, \text{achievements})
]

---

## Estimating the Conditional Probability

[
P(w \mid R)
\approx
\frac{P(w, q_1, \dots, q_n)}{P(q_1, \dots, q_n)}
]

So the task reduces to estimating the **joint probability**:

[
P(w, q_1, \dots, q_n)
]

---

## Estimating the Joint Distribution

Assumptions:

* Words are sampled **i.i.d.** from a unigram distribution
* There exists a finite universe ( \mathcal{M} ) of unigram models

Sampling process:

1. Sample a model ( M \in \mathcal{M} ) with probability ( P(M) )
2. Sample ( n+1 ) words from ( M )

Thus:

[
P(w, q_1, \dots, q_n)
=====================

\sum_{M \in \mathcal{M}} P(M),P(w, q_1, \dots, q_n \mid M)
]

---

## I.I.D. Simplification

Under i.i.d. sampling:

[
P(w, q_1, \dots, q_n \mid M)
============================

P(w \mid M)\prod_{i=1}^{n} P(q_i \mid M)
]

So:

[
P(w, q_1, \dots, q_n)
=====================

\sum_{M \in \mathcal{M}}
P(M),P(w \mid M)\prod_{i=1}^{n} P(q_i \mid M)
]

---

## Practical Shortcut

Instead of summing over all ( M ), approximate using an **initial retrieved set**:

[
\mathcal{D}_{init}
]

Each document ( d \in \mathcal{D}_{init} ) defines a language model ( M_d ).

---

## The Relevance Model (RM1)

Using the approximation:

[
P(w \mid R)
\approx
\sum_{d \in \mathcal{D}_{init}}
P(w \mid M_d),P(M_d \mid q)
]

Where:

[
P(M_d \mid q)
=============

\frac{P(q \mid M_d)P(M_d)}
{\sum_{d' \in \mathcal{D}*{init}} P(q \mid M*{d'})P(M_{d'})}
]

This is known as **RM1**.

---

## Interpolated Relevance Model (RM3)

To reduce query drift, interpolate RM1 with the original query model:

[
P(w \mid R)
\approx
\beta,P_{\text{MLE}}(w \mid q)
+
(1-\beta)
\sum_{d \in \mathcal{D}_{init}}
P(w \mid M_d),P(M_d \mid q)
]

* Introduced by **Abdul-Jaleel et al. (2004)**
* Known as **RM3**

---

## Ranking with Relevance Models

Documents are ranked by **KL divergence** between the relevance model and document model:

[
\text{Score}(d)
===============

* D_{\mathrm{KL}}(P(\cdot \mid R) ;|; P(\cdot \mid M_d))
  ]

Expanded:

[
\text{Score}(d)
===============

*

\sum_{w}
P(w \mid R)
\log
\frac{P(w \mid R)}{P(w \mid M_d)}
]

---

## Empirical Performance

Empirical results (ROBUST, WT10G collections) show:

* RM3 consistently outperforms:

  * Initial retrieval
  * RM1
* Gains observed in:

  * MAP
  * P@5
  * NDCG@5

(See performance table on slide ~18.)

---

## Relevance Feedback

### General Idea

Allows the system to **update the query representation**:

* Can be **iterative**
* Can be **implicit** (e.g., clicks)

---

## Pseudo-Relevance Feedback (PRF)

Procedure:

1. Retrieve ( n ) documents
2. Assume top ( m ) documents are relevant
3. Induce relevance model

[
P(M_d \mid q)
=============

\frac{P(q \mid M_d)P(M_d)}
{\sum_{d \in \mathcal{D}_{init}} P(q \mid M_d)P(M_d)}
]

---

## True Relevance Feedback

Procedure:

1. Retrieve ( n ) documents
2. User labels ( m ) relevant documents
3. Assign uniform weight:

[
P(M_d \mid q) = \frac{1}{m}
]

---

## Worked Example

**Query:**

> *“orange apple”*

**Documents:**

* ( d_1 ): orange orange lemon lemon clementine apple
* ( d_2 ): orange and red are beautiful colors
* ( d_3 ): orange and cellcom offer a special deal
* ( d_4 ): the fruits I like most are orange and apple

Assume **Dirichlet smoothing** with ( \mu = 1000 ):

[
P(w \mid M_d)
=============

\frac{tf(w,d) + \mu P(w \mid M_C)}{dl_d + \mu}
]

Example:

[
P(q \mid d_1)
=============

P(\text{orange} \mid d_1)\cdot P(\text{apple} \mid d_1)
= 0.0129
]

---

## Constructing the Relevance Model

Using top-2 documents:

[
P(w \mid R)
\approx
\sum_{d \in \mathcal{D}_{init}}
P(w \mid M_d)P(M_d \mid q)
]

Example:

[
P(\text{apple} \mid R) = 0.139
]

---

## Clipping the Query Model

### Motivation

* Expanded query may contain **many terms**
* Problems:

  * Query drift
  * Computational cost

### Solution

* Keep only top-( v ) terms by probability

[
P(w \mid RM3)
\triangleq
\beta P_{\text{MLE}}(w \mid q)
+
(1-\beta)P(w \mid RM1_{\text{clipped}})
]

---

## Clipping Example (INEX)

* Query: *“nobel prize”*
* Relevant docs: 22
* RM3 parameters:

  * ( \beta = 0.2 )
  * ( v = 100 )

Observation:

* 79/100 top terms are stopwords
* Top terms include:

  > nobel, prize, the, of, in, and, for, on, 2007, medicine, physiology

---

## Additional Thoughts

* Does expansion always move toward the centroid of relevant documents?
* How should ( P(M_d \mid q) ) be estimated?
* KL-ranking induces an **IDF effect**
* Is this effect universal?

(See Raiber & Kurland, 2017)

---

## Does the IDF Effect Always Hold?

Let ( M_d ) be a smoothed document model:

[
M_d = (1-\lambda) M_d^{\text{MLE}} + \lambda M_C^{\text{MLE}}
]

For ( w \in d ):

[
p_s(w \mid M_d)
===============

(1-\lambda)p(w \mid M_d^{\text{MLE}})
+
\lambda p(w \mid M_C^{\text{MLE}})
]

For ( w \notin d ):

[
p_u(w \mid M_d)
===============

\lambda p(w \mid M_C^{\text{MLE}})
]

This yields an IDF-like weighting term:

[
\log\left(
1 +
\frac{1-\lambda}{\lambda}
\frac{p(w \mid M_d^{\text{MLE}})}{p(w \mid M_C^{\text{MLE}})}
\right)
]

---

If you want next:

* **Exam-focused summary**
* **RM1 vs RM3 vs BM25 side-by-side**
* **Concrete PRF parameter guidelines**
* **Parallels to modern LLM-based query expansion**



================================================================================
FILE: Lecture_11_PassageRetrieval.md
================================================================================


Below is a **clean Markdown version** of the slides, with **LaTeX used for math where applicable**.
Figures/diagrams are referenced textually (since Markdown can’t faithfully reconstruct slide graphics).

---

# Text Retrieval and Search Engines

**Lecture 11 – Passage Retrieval**
*Reichman University, Semester 1 – 2026*
(Some slides adapted from Stanford)

---

## Challenges with Document Retrieval

* Documents can be long and topically heterogeneous
* Growing popularity of mobile search applications

---

## The Focused (Passage) Retrieval Task

**Goal:**
Rank *passages* of documents by their relevance to the information need expressed by a query
([Geva et al., 2010; Arvola et al., 2011])

* A **passage** is a short sequence of text in a document
* **Focused relevance judgments**

  * Markups of relevant text in relevant documents

**Example:**

* Document ( d )
* Passages ( g_i \in d )

---

## Passage Annotations

**Query:** *“nobel prize”*

**Relevant document (excerpt):**

> “…Science and engineering awards Organizations based in Sweden Awards Nobel Prize …
> Outstanding contributions in Physics, Chemistry, Literature, Peace, and Physiology or Medicine …”

* **Character offset:** 167
* **Relevant character length:** 538

---

## Applications of Passage Retrieval

* **Passage-based document retrieval**
  [Callan ’94; Wilkinson ’94; Kaszkiel & Zobel ’97, ’01; Liu & Croft ’02; Na et al. ’08; Bendersky & Kurland ’10]

  * Relevant documents may contain much non-relevant information

* **Focused retrieval**
  [Murdock ’06; Buffoni et al. ’10; Fernández et al. ’11; Fernández & Losada ’12; Carmel et al. ’13]

  * Browsing long documents is time consuming

* **Passage retrieval as an intermediate step**

  * Question answering, summarization, LLM context
    [Cardie et al. ’00; Light et al. ’01; Corrada-Emmanuel et al. ’03; Tellex et al. ’03;
    Murdock & Croft ’04; Aktolga et al. ’11; Chen et al. ’17; Ram et al. ’23]

---

## Different Types of Passages

* **Discourse passages**
  [Salton et al. ’93; Callan ’94; Buffoni et al. ’10; Losada ’10; Fernández et al. ’11;
  Carmel et al. ’13; Yang et al. ’16]

* **Semantic passages**
  [Hearst ’93; Hearst & Plaunt ’93; Mittendorf & Schäub le ’94; Denoyer et al. ’01; Jiang & Zhai ’04]

* **Sliding windows**
  [Callan ’94; Kaszkiel & Zobel ’97, ’01; Liu & Croft ’02; Wang & Si ’08;
  Bendersky & Kurland ’10]

---

## Common Passage Ranking Approaches

* Everything used for documents can be used for passages:

  * Query likelihood
  * Relevance models (RM)
  * BM25

### Vocabulary Mismatch Problem

Short relevant passages often mismatch the query vocabulary.

Mitigation strategies:

* Smoothing with the **ambient document**
  [Murdock ’06; Bendersky & Kurland ’08]
* Smoothing with **neighboring passages**
  [Fernández et al. ’11; Carmel et al. ’13]
* **Query expansion**
  [Losada ’10]
* **Semantic similarity**

  * Entities [Ferragina & Scaiella ’12]
  * ESA [Gabrilovich & Markovitch ’07]
  * Neural models
    [Mikolov et al. ’13; Nogueira & Cho ’20; Karpukhin et al. ’20]

---

## Inter-Document Similarity Motivation

Previous work showed inter-document similarities help address vocabulary mismatch:

* **Cluster-based methods**
  [Jardine & van Rijsbergen ’71; Voorhees ’85; Kurland & Lee ’04;
  Liu & Croft ’04, ’08; Raiber & Kurland ’13]

---

## Testing the Cluster Hypothesis with Focused Judgments

*[Sheetrit et al., SIGIR ’18]*

### Research Questions

1. Does the cluster hypothesis hold for **passage retrieval**?
2. Does it hold more strongly for items containing a **high fraction of relevant text**?

---

## Nearest-Neighbor Test (Voorhees ’85)

* Performed for each query over top retrieved items
* Let:

  * ( \beta ): relevance degree threshold
* **Test result:**
  Average relevance degree of a seed’s ( k ) nearest neighbors

### Relevance Judgments

* **Focused**

  * Relevance = fraction of relevant text
  * Seed selection: ( \beta = 0^+ )
* **Binary**

  * Relevance ∈ {0, 1}
  * Seed selection: ( \beta = 1 )

---

## Dataset

| Dataset | #Docs     | #Queries | Avg. Rel. Fraction | Judgments |
| ------- | --------- | -------- | ------------------ | --------- |
| INEX    | 2,666,190 | 120      | 0.40               | Focused   |

---

## Binary Test Results (INEX)

* Passage length: 150
* Metric: Average fraction of relevant neighbors (4-NN)

**Finding:**
The cluster hypothesis holds for passage retrieval.

---

## Focused Test Results (INEX)

* Increasing ( \beta ) increases effect size
* Statistically significant differences at ( \beta = 0.75 )

**Finding:**
The hypothesis holds more strongly for items with a high fraction of relevant text.

---

## The ClustPsg Method

### Notation

* ( L_d ): top-ranked documents
* ( L_g ): top-ranked passages
* ( c ): a cluster of passages

### Method Overview

* Nearest-neighbor clustering
  [Voorhees ’85; Sheetrit et al. ’18]
* Learning-to-rank model

#### Query-dependent features (2)

* Reciprocal rank of passages in ( L_g )
* Reciprocal rank of documents in ( L_d )

#### Cluster priors (4)

* Entropy of term distributions
* Stopword ratio
* Inter-passage similarity
* Number of unique documents in cluster

---

## Experiments

* ClustPsg applied to different initial passage lists ( L_g^{init} )

### Datasets

| Dataset | #Docs     | #Queries | Avg. Rel. Fraction | Judgments | Passage Type                |
| ------- | --------- | -------- | ------------------ | --------- | --------------------------- |
| INEX    | 2,666,190 | 120      | 0.40               | Focused   | Sliding windows (300 terms) |
| AQUAINT | 1,033,461 | 100      | 0.37               | Binary    | Marked sentences            |

---

## Main Results (INEX)

* Metric: **MAiP@100**
* Clusters of size 10

**Findings:**

1. Query-independent priors are effective
2. ClustPsg improves re-ranking across different initial retrieval methods

---

## Optimal Cluster Analysis

* Metric: Average fraction of relevant text in top-10 passages

**Oracle:** Rank clusters by true relevance fraction

* Oracle > ClustPsg > LM baseline

---

## Why Learning to Rank?

* Machine learning for IR
* Text → numeric features
* Combine multiple relevance signals

  * Handcrafted features over ( \langle q, d \rangle )

---

## Learning-to-Rank Framework

*(After Liu ’08)*

* Training data: labeled ( (q, d) ) pairs
* Model:
  [
  f(q, d; \mathbf{w})
  ]
* Optimize loss → produce ranking scores

---

## Pointwise Ranking

* Documents treated independently
* Regression or classification

[
f(\mathbf{w}, \mathbf{V}(d, q)) \rightarrow y
]

---

## Pairwise Ranking

* Compare document pairs:

[
f(\mathbf{w}, \mathbf{V}(d_{ij}, q_i)) \ge
f(\mathbf{w}, \mathbf{V}(d_{ik}, q_i)) + \epsilon
]

* Examples: RankNet, RankSVM

---

## Listwise Ranking

* Optimize ordering of entire list
* Example: ListNet

---

## Useful Libraries

* RankSVM: [http://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html](http://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html)
* RankLib: [https://sourceforge.net/p/lemur/wiki/RankLib/](https://sourceforge.net/p/lemur/wiki/RankLib/)
* TensorFlow Ranking: [https://github.com/tensorflow/ranking](https://github.com/tensorflow/ranking)
* LightGBM: [https://lightgbm.readthedocs.io/](https://lightgbm.readthedocs.io/)

---

## Dense Retrieval

* Encode queries and passages as dense vectors
* Methods:

  * Bi-encoders
  * Cross-encoders
  * LLMs
* Enabled by GPUs/TPUs
* Supports natural, conversational queries

---

## Cross-Encoder with BERT

* Input:

```
[CLS] Query [SEP] Document [SEP]
```

* Use final `[CLS]` embedding
* Fully-connected relevance head
* Typically used for re-ranking top-( K ) (e.g., BM25 candidates)

---

## Fine-Tuning (Nogueira & Cho ’20)

* Retrieve 1K passages via BM25
* Hard negatives from BM25

**Loss:**
[
\mathcal{L}
= - \sum_{j \in J_{\text{pos}}} \log s_j

* \sum_{j \in J_{\text{neg}}} \log (1 - s_j)
  ]

Where:

* ( s_j ): predicted relevance score
* ( J_{\text{pos}}, J_{\text{neg}} ): positive/negative sets

---

## Bi-Encoder

* Encode query and passage independently

* Single vector per input

* Similarity:
  [
  \text{sim}(q, p) = \mathbf{h}_q^\top \mathbf{h}_p
  ]

* Example: Dense Passage Retrieval (DPR)
  [Karpukhin et al. ’20]

---

## Dense Passage Retrieval (DPR)

* 768-dim query and passage embeddings
* N-way cross-entropy loss
* Negatives:

  * BM25 top-100
  * In-batch negatives

---

## Representation Similarity: Downsides

* Single-vector compression loses fine-grained information
* No token-level interaction
* Relevance = single dot product
* We lose:

  * Term interactions (BM25, QL)
  * Cross-attention (BERT)

---

## Beyond Re-Ranking: End-to-End Retrieval

* Cross-encoders require re-ranking pipelines
* End-to-end retrieval improves recall
* Dense models allow:

  * Direct search over all documents
  * Vector indexing

---

## Vector Databases

* Efficient indexing of high-dimensional vectors
* Native vector operations:

  * Cosine similarity
  * Euclidean distance
  * ( k )-NN
* Optimized for Approximate Nearest Neighbor (ANN)
* Support sparse + dense representations

---

If you want, I can also:

* Split this into **exam-ready notes**
* Convert it to **LaTeX** or **Jupyter markdown**
* Add **cross-links** or **IR formulas expansions** (BM25, QL, RM3)



================================================================================
FILE: Lecture_9_LM.md
================================================================================


Below is a **clean Markdown conversion** of **Lecture 9 – The Language Modeling Approach**, with **LaTeX for all math**, organized for study and reference.
(Source: uploaded PDF )

---

# Text Retrieval and Search Engines

## Lecture 9 – The Language Modeling Approach

*Semester 1 – 2026*
(Some slides adapted from *Introduction to Information Retrieval* — Manning, Raghavan, Schütze; and Victor Lavrenko)

---

## Statistical Language Models

A **statistical language model (LM)** defines a probability distribution over sequences of terms.

Examples:

[
P(\text{“Today is Thursday”} \mid M) = 0.001
]
[
P(\text{“Today Thursday is”} \mid M) = 10^{-7}
]
[
P(\text{“Information retrieval is fun”} \mid M) = 0.01
]

Key properties:

* Context-dependent
* Can be viewed as **generative models** of text

---

## Unigram Language Model

A document ( d ) is represented as a **unigram distribution** over the vocabulary ( V ):

* Words are sampled **independently**
* “Bag-of-words” assumption
* Parameters:

[
\sum_{t \in V} P(t \mid M_d) = 1
]

Each document defines its own language model ( M_d ).

---

## Language Models for Retrieval

### Intuition

1. User has an **information need**
2. User imagines relevant documents
3. User generates a query as a sample of terms
4. Rank documents by how likely they are to generate the query

---

## Retrieval Framework

For each document ( d_i ):

1. Estimate a language model ( M_{d_i} )
2. Compute:
   [
   P(q \mid M_{d_i})
   ]
3. Rank documents by this probability

This is known as the **query likelihood** approach.

---

## Unigram Query Generation

Assumptions:

* Query is generated as a multinomial sample
* Independent draws from ( M_d )

Given:

* Vocabulary size ( |V| )
* Document length ( N )

The probability of a document under its model is:

[
P(d \mid M_d)
=============

\frac{N!}{\prod_{t \in V} tf(t \in d)!}
\prod_{t \in V} P(t \mid M_d)^{tf(t \in d)}
]

---

## Maximum Likelihood Estimation (MLE)

For a document ( d ):

[
P_{\text{MLE}}(t \mid M_d)
==========================

\frac{tf(t \in d)}{|d|}
]

Where:

* ( tf(t \in d) ): term frequency
* ( |d| ): document length

---

## Query Likelihood Scoring

Ignoring constants:

[
P(d \mid q)
\propto
P(q \mid M_d)
]

With MLE:

[
P(q \mid M_d)
=============

\prod_{t \in q}
P_{\text{MLE}}(t \mid M_d)
==========================

\prod_{t \in q}
\frac{tf(t \in d)}{|d|}
]

---

## Sparse Data Problem

Problem:

* If ( tf(t \in d) = 0 ), then
  [
  P_{\text{MLE}}(t \mid M_d) = 0
  ]
* A single missing query term makes:
  [
  P(q \mid M_d) = 0
  ]

This is unrealistic for incomplete samples.

---

## General Solution: Smoothing

Use **background probabilities** from the entire collection ( C ):

[
P_{\text{MLE}}(t \mid M_C)
==========================

\frac{tf(t \in C)}{\sum_{w} tf(w \in C)}
]

A non-occurring term is still possible, but rare.

---

## Jelinek–Mercer (JM) Smoothing

[
P_{\text{JM}}(t \mid M_d)
=========================

(1 - \lambda) P_{\text{MLE}}(t \mid M_d)
+
\lambda P_{\text{MLE}}(t \mid M_C)
]

Properties:

* ( \lambda \in [0,1] )
* Fixed across documents
* Interpretation:

  * Small ( \lambda ): conjunctive, short queries
  * Large ( \lambda ): longer queries
* Tuned empirically

---

## Dirichlet Smoothing

Bayesian smoothing with a Dirichlet prior:

[
P_{\text{Dir}}(t \mid M_d)
==========================

\frac{tf(t \in d) + \mu P_{\text{MLE}}(t \mid M_C)}{|d| + \mu}
]

Equivalent form:

[
P_{\text{Dir}}(t \mid M_d)
==========================

(1-\lambda_d) P_{\text{MLE}}(t \mid M_d)
+
\lambda_d P_{\text{MLE}}(t \mid M_C)
]

Where:

[
\lambda_d = \frac{\mu}{|d| + \mu}
]

---

## Worked Example (JM Smoothing)

**Query:**

> *“onion soup onion”*

**Documents:**

* ( D_1 ): onion vegetable soup vegetable mushroom corn
* ( D_2 ): corn onion soup onion mushroom corn
* ( D_3 ): potato pumpkin tofu potato tofu potato

Collection statistics (18 terms, 8 unique):

[
P_{\text{MLE}}(\text{onion} \mid M_C) = \frac{3}{18}
]

Using JM with ( \lambda = 0.2 ):

[
P(\text{onion} \mid D_1)
========================

0.8 \cdot \frac{1}{6}
+
0.2 \cdot \frac{3}{18}
======================

\frac{1}{6}
]

[
P(\text{soup} \mid D_1)
=======================

0.8 \cdot \frac{1}{6}
+
0.2 \cdot \frac{2}{18}
======================

0.156
]

---

## Query Likelihood Scores

[
P(q \mid D_1)
=============

\frac{1}{6} \cdot 0.156 \cdot \frac{1}{6}
= 0.0043
]

[
P(q \mid D_2)
=============

0.3 \cdot 0.156 \cdot 0.3
= 0.014
]

[
P(q \mid D_3)
=============

0.033 \cdot 0.022 \cdot 0.033
\approx 2 \times 10^{-5}
]

**Ranking:**

[
D_2 > D_1 > D_3
]

---

## Language Models vs. Probabilistic Models

Differences:

* LM approach avoids explicit modeling of *relevance*
* Documents and queries are treated symmetrically
* Probabilistic interpretation (vs. geometric similarity)
* Efficient and intuitive

Limitations:

* Very simple language assumptions
* Hard to incorporate:

  * Relevance feedback
  * User preferences
  * Phrases, Boolean queries

---

## Alternative Retrieval Views

Three equivalent perspectives:

1. **Query likelihood**:
   [
   P(q \mid M_d)
   ]

2. **Document likelihood**:
   [
   P(d \mid M_q)
   ]
   (hard because queries are short)

3. **Model comparison**:
   [
   D_{\mathrm{KL}}(M_q \parallel M_d)
   ]

---

## KL-Divergence Ranking

[
D_{\mathrm{KL}}(M_q \parallel M_d)
==================================

\sum_{w}
P(w \mid M_q)
\log
\frac{P(w \mid M_q)}{P(w \mid M_d)}
]

Ranking by **min KL** is equivalent to query likelihood when ( M_q ) is MLE.

---

## KL Example

For query model:

[
P(\text{onion} \mid M_q) = \frac{2}{3}, \quad
P(\text{soup} \mid M_q) = \frac{1}{3}
]

Computed divergences:

[
\text{KL}(M_q \parallel M_{D_2}) = 0.785
]
[
\text{KL}(M_q \parallel M_{D_1}) = 1.177
]
[
\text{KL}(M_q \parallel M_{D_3}) = 2.910
]

**Ranking:**
[
D_2 > D_1 > D_3
]

---

## Relevance Feedback in LM Framework

* Introduce a **query model**
* Update it using feedback documents
* Transition:

  * Expansion-based feedback → Model-based feedback

This leads naturally to **Relevance Models (RM)** (next lecture).

---

## Connection Between Smoothing and IDF

Let:

* ( p_s(w \mid d) ): probability of a **seen** word
* ( p_u(w \mid d) ): probability of an **unseen** word

For query likelihood:

[
\log P(q \mid d)
================

\sum_{w \in q \cap d}
\log
\frac{p_s(w \mid d)}{p_u(w \mid d)}
+
\sum_{w \in q}
\log p_u(w \mid d)
]

With corpus smoothing:

[
p_u(w \mid d)
=============

\alpha_d P_{\text{MLE}}(w \mid M_C)
]

The ratio:

[
\log
\frac{p_s(w \mid d)}{p_u(w \mid d)}
]

acts as an **IDF-like weighting**, document-independent.

---

## Key Takeaways

* Language modeling provides a principled probabilistic foundation for IR
* Smoothing is essential to avoid zero probabilities
* Query likelihood, KL-divergence, and relevance models are tightly connected
* IDF naturally emerges from smoothing assumptions
* Efficient implementations rely on inverted indexes

---

If you want next, I can:

* Produce a **one-page exam cheat sheet**
* Compare **LM vs BM25 vs RM3**
* Show **how this maps to modern neural retrieval**
* Rewrite this as **Jupyter Markdown / LaTeX notes**



================================================================================
FILE: Retrieving and Detecting AI-Generated Fakes - IDC.md
================================================================================


Below is a **clean Markdown conversion** of the presentation
**“The Fake News Arms Race: Retrieving and Detecting AI-Generated Fakes”**, with **LaTeX used where math/technical notation is appropriate** and slide content structured logically.
(Source: uploaded PPTX )

---

# The Fake News Arms Race

## Retrieving and Detecting AI-Generated Fakes

**Or Levi**
VP Data Science, Zefr

---

## Quiz: Real or Fake?

**Which image is the real one?**

> (Interactive slide – visual comparison of real vs. AI-generated images)

---

## Quiz: Real or Fake (Audio & Text)

* **MelNet: A Generative Model for Audio in the Frequency Domain**
  [https://audio-samples.github.io/](https://audio-samples.github.io/)

**Examples of AI-generated text snippets:**

> “Pluck the bright rose without leaves.”
> “The glow deepened in the eyes of the sweet girl.”
> “A cramp is no small danger on a swim.”

---

## Chapter I

## Generative AI Misinformation: Can We No Longer Believe Anything We See?

* AI-generated images used to promote misinformation
* AI-generated videos using **Runway ML**
* AI-generated **deepfakes**

**Example:**
TikTok deepfake account:
[https://www.tiktok.com/@deeptomcruise](https://www.tiktok.com/@deeptomcruise)

---

## Information Wars in the Digital Space

* Misinformation as a geopolitical and societal weapon
* Rapid amplification via social platforms

**Reference:**
[https://www.ynet.co.il/digital/technews/article/b17ychbja](https://www.ynet.co.il/digital/technews/article/b17ychbja)

---

## Not Only “Misinformation”

* **NSFW generative content**

  * Example: *PornPen.ai* generating massive volumes of synthetic pornographic imagery
* **Junk news**

  * Content generated primarily for advertising revenue
* **Brand safety concerns**

  * Large brands increasingly affected

---

## Chapter II

## The Rise of Generative AI

### Key Milestones

* **Transformers**
  *Attention Is All You Need*
* **Human brain:** ~86 billion neurons
* **GANs (2014)**
* **CLIP**
* **Diffusion models**

> AI generators became **very good, very fast**

---

## Chapter III

## Can We Use AI to Catch AI-Generated Images?

---

## Solutions – Watermarking

* **Invisible watermarking**
* Example: **SynthID**

  * Embed signals directly into generated images
  * Detectable post-hoc without modifying visible content

---

## Solutions – Supervised Approaches (Text)

* **OpenAI Text Detection**
* **GPT-Zero**

  * Probability-based estimation of AI-generated text
  * Fragile to paraphrasing and prompt engineering

---

## Solutions – AI Image Detection

### Prior Work & Benchmarks

* **COCOFake**
  *Distinguishing Multimodal DeepFakes from Natural Images*
  Amoroso et al., April 2023
  [https://arxiv.org/abs/2304.00500](https://arxiv.org/abs/2304.00500)

* **GenImage**
  *A Million-Scale Benchmark for Detecting AI-Generated Images*
  Zhu et al., June 2023
  [https://arxiv.org/abs/2306.08571](https://arxiv.org/abs/2306.08571)

---

## Image VerifAI

### Detecting Manipulated Media Using Computer Vision

* Computer vision model trained to detect:

  * Political figures
  * Social-issue related misinformation
* Binary classification:

  * **REAL**
  * **FAKE**

---

## Training Data – COCOFake

* Labeled dataset of manipulated vs. natural images
* Used to train supervised CV models for fake detection

---

## Large-Scale Data Collection

### Sources

* Reddit:

  * `r/midjourney`
  * `r/stablediffusion`
* Social media queries:

  ```
  (#midjourney AND [political figure])
  (#stablediffusion AND [political figure])
  site:tiktok.com [political figure]
  ```

---

## Dataset Overview

* **FAKE images**

  * AI-generated political figures
* **REAL images**

  * Verified authentic photos
* **Scale**

  * Thousands of labeled real/fake images
  * Eventually ~600k images

---

## Efficient Vector Search with Qdrant

* **Qdrant**: fast, scalable vector database
* Supports:

  * Hundreds of millions of content items
  * HNSW (Hierarchical Navigable Small World) indexing
* Enables efficient similarity search

**Docs:**
[https://qdrant.tech/documentation/overview/vector-search/](https://qdrant.tech/documentation/overview/vector-search/)

---

## Multimodal Retrieval

* Image embeddings
* Text embeddings
* Cross-modal similarity search
* Enables:

  * Near-duplicate detection
  * Source discovery
  * Contextual analysis

---

## Retrieval-Augmented Generation (RAG)

* Retrieval used to:

  * Add external evidence
  * Provide context
  * Improve explainability
* Detection + retrieval pipelines work together

---

## Epilogue

> “AI is going to take over the world and kill us.”
> “AI disinformation could destroy us, the internet, and democracy.”
> “We should halt AI development.”

---

## A Counter-Perspective

> **Marc Andreessen:**
> “The development of AI — far from a risk that we should fear — is a moral obligation that we have to ourselves, to our children, and to our future.”

Source:
[https://a16z.com/ai-will-save-the-world/](https://a16z.com/ai-will-save-the-world/)

---

## Image VerifAI

### Specialized for Misinformation on Social Media

---

## Applications of AI Image Detection

* **Content moderation**

  * Protecting users and brands
* **Combatting misinformation**

  * Collaboration with fact-checkers and newsrooms
* **Discovery tools**

  * Surfacing suspicious content for human review

---

## Thank You

**Or Levi**

* LinkedIn: [https://linkedin.com/in/orlevi](https://linkedin.com/in/orlevi)
* Email: [or.levi@zefr.com](mailto:or.levi@zefr.com)

---

If you want, I can next:

* Turn this into **exam-ready notes**
* Add **formal ML formulations** (classification, ROC, PR-AUC)
* Map this to **retrieval + detection architectures** you’re already using
* Convert it into **LaTeX / Jupyter / Confluence format**
