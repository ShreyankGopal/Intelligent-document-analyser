# Document Intelligence System — Technical Overview

## Overview

Our solution implements a **multi-stage pipeline** that combines structural PDF analysis with semantic understanding to extract and rank document sections based on persona-specific requirements. The approach leverages both traditional document processing techniques and modern embedding-based retrieval methods.

---
## How to Build and Run
Upload your pdf files and the .json file in ```/app/input``` folder
- clone the repository
- Build the docker image
```bash
docker build --platform linux/amd64 -t document-analyzer .
```
- Run the docker container
```bash
docker run --rm \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  --network none \
  document-analyzer
```

Your output will be present in ```/app/output/output.json```
## Core Methodology

### 1. PDF Structure Extraction

- Utilizes **PyMuPDF** to extract text along with formatting metadata:
  - Font sizes
  - Styles (bold/italic)
  - Positional coordinates
- Groups consecutive text blocks with identical formatting to detect section boundaries accurately.

---

### 2. Intelligent Text Cleaning

- Applies a **regex-based preprocessing pipeline** to remove bullet points and visual artifacts.
- Implements **Unicode-aware bullet detection**, supporting diverse international formats.
- Ensures **semantic integrity** while normalizing noisy input text.

---

### 3. Feature Engineering Pipeline

Each text segment is represented using a **10-dimensional feature vector** including:

- **Structural Features:**
  - Font size ratios
  - Positional coordinates
  - Vertical spacing (`y-gap`)
  
- **Content Features:**
  - Text length
  - Capitalization ratio
  - Numbering patterns
  
- **Typography Features:**
  - Bold/italic indicators
  - Font uniqueness metrics

> All features are normalized using `MinMaxScaler` to ensure model stability across document formats.

---

### 4. Dual-Stage Heading Classification

A robust hybrid system combining:

- **Heuristic Stage:**
  - Detects headings using:
    - Font size > 85th percentile
    - Formatting cues (bold, capitalized, colon-ending)
    - Numbered headings (e.g., `1.2.`)

- **ML Stage:**
  - Refines predictions using a **pre-trained classifier**
  - Classifies headings into `Title`, `H1`, `H2`, `H3` using the engineered feature set

---
### 6. Chunking

- Every heading identified by the random forest classifier along with the section text is used as a chunk to create the embeddings

---

### 5. Semantic Section Assembly

- Combines headings with subsequent content to construct **hierarchical sections**
- Generates **384-dimensional embeddings** using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Sections are semantically encoded for downstream retrieval tasks

---

### 6. MMR-Based Relevance Ranking

Implements **Maximal Marginal Relevance (MMR)** with λ = 0.72 to balance:

- **Relevance to the query**
- **Diversity across selected sections**

#### MMR Formula:

```math
MMR(S_i) = λ × sim(query, S_i) - (1 - λ) × max(sim(S_i, S_j)), ∀ S_j ∈ Selected
