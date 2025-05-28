# NL2LS: LLM-based Automatic Linking of Knowledge Graphs
### Automating Link Specification Generation from Natural Language using Large Language Models

Can large language models (LLMs) recognize link specifications (LS)? We hypothesize that LS can be considered a low-resource language.

In our previous work, we explored how LS can be translated into natural language (NL).  
In this project, we tackle the reverse problem — translating NL into LS — which we call **NL2LS**.

<p align="center">
  <img src="https://github.com/dice-group/NL2LS/blob/main/Figure.drawio.png" alt="NL2LS Architecture" width="600">
</p>

---

## Overview

NL2LS is a novel framework that automates the generation of link specifications (LS) from English and German natural language inputs using LLMs.

We address this task using:
- **Rule-based methods** (regex-based)
- **Zero-shot learning** via prompting
- **Supervised fine-tuning** of LLMs

---

## Models Used

We experimented with the following model families:
- **T5**: Encoder-Decoder architecture  
- **LLaMA-3**: Decoder-only architecture  
- **LOLA**: Decoder with Mixture-of-Experts (MoE) layers  

> We also evaluated **GPT** and **Mistral** models during early development.  
> However, due to their architectural similarity to LLaMA (i.e., decoder-only), and to avoid redundancy, we decided **not to include them** in the final experiments.

---

## Datasets

We used and extended multiple benchmark datasets:

| Dataset                  | Description                                      |
|--------------------------|--------------------------------------------------|
| LIMES-Silver             | Auto-generated NL–LS pairs                       |
| LIMES-Annotated          | Human-verified LS verbalizations                |
| SILK-Annotated           | Based on SILK link discovery framework          |
| LIMES-Geo-Temporal       | Contains geospatial and temporal LSs            |
| German Counterparts      | Translations of NLs into German                 |

---

## Evaluation & Results

We evaluated all models using:
- **BLEU**
- **METEOR**
- **ChrF++**
- **TER**

### Key Results:
- LOLA and LLaMA (fine-tuned) achieved BLEU scores up to **98.8** on English datasets.
- LOLA showed excellent **generalization**, with **>95 BLEU** on German test sets.

---

### Environment and Dependencies

```
- Ubuntu 10.04.2 LTS
- Python ≥ 3.8
- torch ≥ 1.7.0
```

---

### Instalation

To run NL2LS locally or train/fine-tune models, download NL2LS repository:
```
git@github.com:dice-group/NL2LS.git
```

Recommended: Create a virtual environment
```
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:
```
pip install -r requirements.txt
```
