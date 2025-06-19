# Video-based RAG Application
This project is a video-based Retrieval-Augmented Generation (RAG) application designed to analyze lecture videos. It allows users to ask questions related to the video content, leveraging a Large Language Model (LLM)—in this case, Gemini—for intelligent responses.
## Features
- Analyze lecture videos for content understanding.
- Prompt any question about the video and receive context-aware answers.
- Integrates with Gemini LLM and Pinecone for vector storage and retrieval.
## Installation
### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Create a Conda environment

```bash
conda create -n video-rag-env python=3.10
conda activate video-rag-env
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Environment Setup

1. **Create accounts:**
    - [Pinecone](https://www.pinecone.io/)
    - [Google Generative AI (Gemini)](https://ai.google.dev/)

2. **Obtain API keys:**
    - `PINECONE_API_KEY`
    - `GEMINI_API_KEY`

3. **Create a `.env.local` file in the project root:**

```
PINECONE_API_KEY=your-pinecone-api-key
GEMINI_API_KEY=your-gemini-api-key
```

## Contributing

Contributions are welcome! Please ensure you have set up your environment as described above before submitting a pull request.
## Contribution Guidelines

Before contributing, please create a new branch for your changes:

```bash
git checkout -b your-feature-branch
```

If you encounter any conflicts while pushing, do not use force push. Instead, resolve conflicts locally and push again. This helps maintain a clean and collaborative workflow.