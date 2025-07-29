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


                                 ┌──────────────────────────────┐
                         │   Input: PDF URL             │
                         └────────────┬─────────────────┘
                                      │
                                      ▼
                       ┌──────────────────────────────┐
                       │ Download PDF via `requests`  │
                       └────────────┬─────────────────┘
                                      │
                                      ▼
                ┌────────────────────────────────────────────┐
                │         PDFProcessor (PyMuPDF)             │
                └────────────┬─────────────┬─────────────────┘
                             │             │
               ┌─────────────▼───┐     ┌────▼───────────────┐
               │ Extract Text    │     │ Extract Images     │
               │ (Split Chunks)  │     │ & Page Snapshots   │
               └─────────────┬───┘     └────┬───────────────┘
                             │              │
                             ▼              ▼
                ┌────────────────┐   ┌────────────────────────────┐
                │ Text Files     │   │ Base64 + Image Captioning  │
                └────────────────┘   └──────────┬─────────────────┘
                                                ▼
                          ┌────────────────────────────────────┐
                          │     Extract Tables via Tabula      │
                          └────────────────────────────────────┘
                                               │
                      ┌────────────────────────┴────────────────────┐
                      ▼                                             ▼
        ┌────────────────────────────┐            ┌────────────────────────────┐
        │ Text, Tables, Images       │            │ Image Captions via Groq    │
        │ with metadata              │            │ Vision LLM (LLaMA-4 Scout) │
        └────────────┬───────────────┘            └────────────┬───────────────┘
                     ▼                                         ▼
                ┌────────────────────────────────────────────────────┐
                │ Generate Embeddings via Ollama (`nomic-embed-text`)│
                └────────────────────────┬───────────────────────────┘
                                         ▼
                        ┌───────────────────────────────────────────────┐
                        │   Store in FAISS (Vector DB) + items.pkl      │
                        └────────────────────────┬──────────────────────┘
                                                 ▼
                            ┌────────────────────────────────────────┐
                            │         User Query Input               │
                            └────────────────┬───────────────────────┘
                                             ▼
                    ┌────────────────────────────────────────────────┐
                    │ Generate Embedding for Query (Ollama)          │
                    └────────────────────────┬───────────────────────┘
                                             ▼
                            ┌────────────────────────────────────┐
                            │ FAISS Similarity Search (Top-k)    │
                            └────────────────┬───────────────────┘
                                             ▼
                            ┌───────────────────────────────────┐
                            │  Context Construction (Text + Img)│
                            └────────────────┬──────────────────┘
                                             ▼
                    ┌─────────────────────────────────────────────────────┐
                    │  Final Answer via Ollama Chat LLM (`llama3.2:3b`)   │
                    └─────────────────────────────────────────────────────┘
