````markdown
# Text Retrieval and Search Engines  
## Text Retrieval Challenge  
**Semester 1 – 2026**

---

## Final Project Part A (70%) – Ranking Competition

### Background

Part A of the final project for the **Text Retrieval and Search Engines** course is a hands-on competition designed to apply the concepts and techniques learned throughout the semester.

In this project, you will be tasked with building the most effective search engine and evaluating its performance against your peers.

Each team will implement **three different information retrieval methods**, where **at least one method must go beyond the material directly taught in class**, requiring you to research and implement an advanced or alternative retrieval technique independently.

Examples include:
- Neural ranking models  
- Learning-to-rank techniques  
- Hybrid approaches  
- Fusion of several models  

The goal is to explore and compare different retrieval strategies to determine which is most effective in retrieving relevant documents from a given dataset.

You will experiment with different parameters, conduct evaluations, and analyze trade-offs between approaches. Ultimately, you will submit **three different retrieval attempts**.

This competition encourages innovation and provides hands-on experience in building end-to-end information retrieval systems, highlighting how retrieval techniques influence relevance and quality of search results.

Your grade will be determined by:
- The quality of your system  
- Creativity  
- Knowledge and correct application of ranking models  

**Have fun!**

---

## Homework Submission Guidelines

1. **Due date:** 22/01/26  
2. Teams of up to **2 students**  
3. Submission instructions appear below  
4. Submission via **Moodle** (by one student only)  
5. **Late submissions will not be considered**

---

## The Challenge

Inside `Final_Project_A/files/` on Moodle:

- **`queriesROBUST.txt`** – 249 queries  
- **`qrels_50_Queries`** – Relevance judgments for the first 50 queries  

### Dataset

The retrieval collection is **ROBUST**, available via Pyserini:

```python
from pyserini.index.lucene import IndexReader
from pyserini.search.lucene import LuceneSearcher

index_reader = IndexReader.from_prebuilt_index('robust04')
lucene_searcher = LuceneSearcher.from_prebuilt_index('robust04')
````

**Note:**
The index was created using **Porter stemming** and **no stopword removal**.

---

## Goal

Achieve the highest retrieval effectiveness measured by **MAP (Mean Average Precision)**.

---

## Task Description

* Total queries: **249**

  * **50 queries** – training (with relevance judgments)
  * **199 queries** – test (used for evaluation)

For each of the **199 test queries**, submit a ranked list of the **top 1,000 documents**, ordered by decreasing relevance.

The relevance judgments for the first 50 queries may be used to:

* Train retrieval methods
* Tune hyperparameters
* Compare ranking models

Evaluation is based **only on the 199 test queries**.

---

## Deliverables

### 1. Retrieval Runs

Submit **3 different result lists (runs)**.

* File names:

  * `run_1.res`
  * `run_2.res`
  * `run_3.res`

### File Format

Standard **6-column TREC format**:

```
630 Q0 ZF08-175-870 1 0.7 run1
630 Q0 ZF08-306-044 2 0.5 run1
630 Q0 ZF09-477-757 3 0.3 run1
630 Q0 ZF08-312-422 4 0.1 run1
630 Q0 ZF08-013-262 5 -0.3 run1
```

**Column definitions:**

1. Topic number
2. Always `Q0`
3. Official document identifier
4. Rank
5. Score (must be in descending / non-increasing order)
6. Run tag (e.g., `run_i`)

---

### Submission Packaging

All runs must be compressed into a single ZIP file named:

```
Final_Project_Part_A_Student_1_email_Student_2_email.zip
```

---

## Presentation Requirement

Each team must present **in person**:

* **Date:** Tuesday, January 27
* **Time:** 17:00 – 20:00

### Presentation Guidelines

* **Duration:**

  * 5 minutes presentation
  * 2 minutes Q&A

* **Content:**

  * Explanation of the three retrieval methods
  * Focus on the method not taught directly in class
  * Innovations and key results
  * Challenges encountered

* **Assessment Criteria:**

  * Clarity and quality of presentation
  * Ability to answer questions
  * Understanding of retrieval concepts

**Note:**
Attendance during the final lecture is mandatory.
Slides do **not** need to be submitted.

---

## Tools and Constraints

1. **Pyserini** toolkit for index interaction
2. Any programming language may be used
3. Algorithms must be **reproducible**
4. Project should be completed using **free Colab resources**

### GPU Usage Tips

* Ensure code runs correctly on **CPU** before using GPU
* Connect to GPU only when ready
* Disconnect GPU runtime during breaks

---

**Good luck!**

```
```
