# Offline RAG System using Endee Vector Database

## Project Overview

This project implements a **Retrieval-Augmented Generation (RAG)** application that works completely offline.
The system reads local PDF documents, converts them into vector embeddings, stores them in the **Endee Vector Database**, and generates answers using a local LLM.

The goal is to build a **private, fast, and low-cost document question-answering system** without using any cloud APIs.

---

## System Design

The application follows a standard RAG pipeline:

1. **Document Ingestion**
   PDF files from `data/docs` are read using `pypdf`.

2. **Chunking & Embedding**
   Text is split into chunks (size 400, overlap 50) and encoded using
   `sentence-transformers (all-MiniLM-L6-v2)` → 384-dim vectors.

3. **Vector Storage (Endee)**
   Embeddings are stored in a local Endee index using cosine similarity and INT8 precision.

4. **Retrieval**
   User query → embedding → Endee Top-K search → relevant chunks returned.

5. **Generation**
   Context + chat memory + question → sent to local LLM using `ctransformers`.

6. **Memory**
   Last 3 interactions stored for better responses.

---

## How Endee is Used

* Connect to local Endee server at `http://localhost:8080/api/v1`
* Create index `rag_index`
* Dimension = 384
* Similarity = cosine
* Precision = INT8
* Store embeddings using `upsert`
* Retrieve context using `query`

Endee provides fast and efficient vector search for the RAG pipeline.

---

## Project Structure

```
project/
│
├── models/
│   └── mistral.gguf
│
├── data/
│   └── docs/
│
├── rag.py
└── README.md
```

---

## Setup Instructions

### 1. Run Endee server

```
docker run -p 8080:8080 endee/server:latest
```

### 2. Install dependencies

```
pip install pypdf sentence-transformers ctransformers endee
```

### 3. Place model

```
models/mistral.gguf
```

### 4. Add PDFs

```
data/docs/
```

### 5. Run

```
python rag.py
```

---

## Usage

```
ingest  → load PDFs
exit    → quit
```

Ask questions after ingestion.

---

## Features

* Offline RAG
* Endee vector database
* Local LLM
* Semantic search
* PDF ingestion
* Chat memory
* No API required

---

## Author

Divya Bansal
Endee Internship Project Submission

---
