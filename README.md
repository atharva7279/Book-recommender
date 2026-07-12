# LLM Semantic Book Recommender

## Overview

LLM Semantic Book Recommender is an AI-powered recommendation system that retrieves books based on the semantic meaning of a user's query instead of exact keyword matching. The application generates vector embeddings for book descriptions, stores them in a vector database, and performs similarity search to recommend the most relevant books. It also classifies books into categories and predicts the emotional tone of each recommendation.

## Features

- Semantic book search using natural language queries
- Vector similarity search with embeddings
- Zero-shot classification for book categorization
- Emotion classification of book descriptions
- Interactive Gradio web interface
- End-to-end recommendation pipeline from preprocessing to retrieval

## Workflow

1. Load and preprocess the book dataset.
2. Generate embeddings for book descriptions.
3. Store embeddings in ChromaDB.
4. Perform semantic similarity search for user queries.
5. Classify retrieved books using zero-shot classification.
6. Predict the emotional tone of recommendations.
7. Display results through the Gradio interface.

## Tech Stack

### Programming Language

- Python

### Libraries and Frameworks

- LangChain
- ChromaDB
- OpenAI Embeddings
- Hugging Face Transformers
- Pandas
- NumPy
- Gradio

### Machine Learning

- Semantic Embeddings
- Vector Similarity Search
- Zero-Shot Classification
- Emotion Classification


```bash
python gradio-dashboard.py
```
