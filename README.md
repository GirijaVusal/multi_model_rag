# Start Django app
- uv sync
- python manage.py runserver

# Start React app 
- cd chat-frontend  
- npm install 
- npm start


You can find Demo videos in assets file.

[How to create embedding](./assets/embedding.webm)

[Chat with data](./assets/query.webm)


# System Diagram


        ┌────────────────────────────────────────────┐
        │         Input: PDF URL                     │
        └────────────────────────────────────────────┘
                           │
                           ▼
        ┌────────────────────────────────────────────┐
        │          Download PDF via `requests`       │
        └────────────────────────────────────────────┘
                           │
                           ▼
        ┌────────────────────────────────────────────┐
        │     Process PDF pages using `PyMuPDF`      │
        │  - Extract text, tables, page images       │
        │  - Extract diagrams/images with base64     │
        └────────────────────────────────────────────┘
                           │
            ┌──────────────┴─────────────────┐
            ▼                                ▼
 ┌──────────────────────┐       ┌──────────────────────────────┐
 │ Extract text chunks  │       │     Extract images & tables  │
 │ (Recursive splitter) │       │  - Caption via Groq LLM      │
 └──────────────────────┘       └──────────────────────────────┘
            │                                │
            └──────────────┬─────────────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │ Generate embeddings using Ollama         │
        │  - Model: nomic-embed-text               │
        └──────────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────┐
        │ Store vectors in FAISS index             │
        │ + Metadata in `items.pkl`                │
        └──────────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────┐
        │      User inputs query                   │
        └──────────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────┐
        │ Embed query and search FAISS index       │
        └──────────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────┐
        │ Retrieve top-k matching chunks           │
        └──────────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────┐
        │ Use Ollama LLM to answer (RAG)           │
        │ - Model: llama3.2:3b                     │
        └──────────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────┐
        │            Final Answer                  │
        └──────────────────────────────────────────┘



