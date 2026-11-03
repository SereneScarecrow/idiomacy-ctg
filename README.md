# CONTROLLED TEXT GENERATION BASED ON THE LEVEL OF IDIOMACY

> **NOTE: the project code is currently undergoing revision. The main goal of the revision is to make the project easily reproducible. Unfortunately, some features may be unavailable at the moment.**

This repository contains materials for coursework on figurative language generation.

This research proposes an original method for studying idiomaticity and text style using supervised generation. It's development stems from the fact that NLP studies often focuse on singular expressions, when in a coherent text these expressions can occur together in one context. The proposed method involves assessing LLMs' ability to transform literal version of the text into an figurative one with a specified level of idiomaticity (low, medium, or high).

### Research pipeline

First, a parralel Russian‑language corpus of literary and news texts was created. It includes original texts, their literalized versions, which are as devoid of figurative language as possible, as well as manual annotation of six types of figurative units (phraseological, semantic, word‑formational, syntactic, pragmatic, and discursive).

Then, 4 language models were tested: Qwen2.5-3B, Qwen2.5-7B, Qwen2.5-14B. They were prompted to generate texts containing low, medium and high levels of figurativeness. All generations were run through the evaluation pipeline, which contains three parts: LLM-as-a-judge evalutaion, automatic metrics for generation quality and various text qualities.

## Setup

### Installation

```
git clone https://github.com/SereneScarecrow/coursework-idiomacy-ctg.git
cd coursework-idiomacy-ctg
```

### Environment

Set your API-keys in a .env file:
```
GIGACHAT_API_KEY="your-key"
```

## Quick start

## Project structure

```
idiomacy-ctg/
├── README.md
├── requirements.txt
├── .env.example
├── configs/
├── prompts/
├── data/
│   ├── raw/dataset_maxi.csv           # initial data gathered by hand (without metrics) 
│   ├── processed/dataset_maxi.csv     # final version of dataset (metrcis computed via build_dataset.py)
│   └── DATASET.md
├── src/idiomacy/
│   ├── data/          # build_dataset.py (text normalization, metrics, result validation)
│   ├── llm/           # llm-interface module (built to work with both text and pictures)
│   ├── metrics/
│   │   ├── text.py            # unified text processing
│   │   ├── spans.py           # idiom spans finder
│   │   ├── quality/           # bertscore, bleurt, comet, meteor, wmd
│   │   ├── idiom/             # perplexity, abstractness, frequency, coherence, sentiment
│   └── pipeline/      # generate.py, evaluate.py (CLI)
├── results/
├── notebooks/         # 01_dataset, 02_judge_validation, 03_results (читают только results/tables)
└── docs/decisions.md  # decisions for this project
```

## AI usage

- **Literature Search**: The AI-powered literature search assistant provided on the arXiv platform was used during the initial stages of this project to discover relevant research materials and references. All literature sources and citations suggested by the assistant tool were rewieved manualy before inclusion.

- **Prompt Engineering**: DeepSeek-V3 was used to refine prompts for literal and idiomatic text generation.

- **Coding**: Claude (Anthropic) was used as a coding and review assistant. AI-generated code was checked, tested and edited where nessesary by the author before inclusion.

The research design, data collection and annotation, experiments and all final decisions, as well as the conclusions of the study, belong to the author.
