import os
import base64
import requests
import tabula
import pymupdf  # fitz
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter

# import pytesseract
from PIL import Image
import numpy as np
import faiss
import pickle
from ollama import embeddings as ollama_embeddings


from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import pymupdf  # fitz

import base64
from dotenv import load_dotenv
import os


load_dotenv()
api_key = os.environ["GROQ_API_KEY"]

# Groq LLM instance
vision_model = ChatGroq(
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=api_key,
    temperature=0,
)

# Prompt template for image captioning
image_summary_prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant. Analyze this research diagram or figure from a paper on Transformers.
Give a specific, technical description of the image: what kind of diagram it is, what elements it shows, and what insight it offers.

Only respond with the description.

Image (base64): {image_b64}
"""
)

# Image captioning chain
image_summary_chain = (
    {"image_b64": lambda x: x} | image_summary_prompt | vision_model | StrOutputParser()
)


def download_file(url, directory="data", filename=None):
    if not filename:
        filename = url.split("/")[-1]
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(filepath, "wb") as file:
            file.write(response.content)
        print(f"File downloaded successfully: {filepath}")
        return filepath
    except requests.RequestException as e:
        print(f"Failed to download the file: {e}")
        return None


class PDFProcessor:
    def __init__(self, url, base_dir="data", chunk_size=700, chunk_overlap=200):
        self.url = url
        self.base_dir = base_dir
        self.filename = os.path.basename(url)
        self.filepath = os.path.join(self.base_dir, self.filename)
        self.items = []
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
        )
        self.doc = None

    def download_file(self):
        os.makedirs(self.base_dir, exist_ok=True)
        try:
            response = requests.get(self.url)
            response.raise_for_status()
            with open(self.filepath, "wb") as f:
                f.write(response.content)
            print(f"Downloaded: {self.filepath}")
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to download the file: {e}")

    def create_directories(self):
        for subdir in ["images", "text", "tables", "page_images"]:
            os.makedirs(os.path.join(self.base_dir, subdir), exist_ok=True)

    def process_pdf(self):
        self.download_file()
        self.create_directories()
        self.doc = pymupdf.open(self.filepath)
        num_pages = len(self.doc)

        for page_num in tqdm(range(num_pages), desc="Processing PDF pages"):
            page = self.doc[page_num]
            text = page.get_text()

            self.process_tables(page_num)
            self.process_text_chunks(text, page_num)
            self.process_images(page, page_num)
            self.process_page_images(page, page_num)

        return self.items

    def process_tables(self, page_num):
        try:
            tables = tabula.read_pdf(
                self.filepath, pages=page_num + 1, multiple_tables=True
            )
            if not tables:
                return
            for table_idx, table in enumerate(tables):
                table_text = "\n".join(
                    [" | ".join(map(str, row)) for row in table.values]
                )
                table_file = f"{self.base_dir}/tables/{self.filename}_table_{page_num}_{table_idx}.txt"
                with open(table_file, "w") as f:
                    f.write(table_text)
                self.items.append(
                    {
                        "page": page_num,
                        "type": "table",
                        "text": table_text,
                        "path": table_file,
                    }
                )
        except Exception as e:
            print(f"Error extracting tables from page {page_num}: {e}")

    def process_text_chunks(self, text, page_num):
        chunks = self.text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            text_file = f"{self.base_dir}/text/{self.filename}_text_{page_num}_{i}.txt"
            with open(text_file, "w") as f:
                f.write(chunk)
            self.items.append(
                {"page": page_num, "type": "text", "text": chunk, "path": text_file}
            )

    # def process_images(self, page, page_num):
    # images = page.get_images()
    # for idx, image in enumerate(images):
    #     xref = image[0]
    #     pix = pymupdf.Pixmap(self.doc, xref)
    #     image_path = f"{self.base_dir}/images/{self.filename}_image_{page_num}_{idx}_{xref}.png"
    #     pix.save(image_path)
    #     # try:
    #     #     ocr_text = pytesseract.image_to_string(Image.open(image_path)).strip()
    #     # except Exception as e:
    #     #     print(f"OCR failed for image {image_path}: {e}")
    #     #     ocr_text = ""

    #     with open(image_path, "rb") as f:
    #         encoded_image = base64.b64encode(f.read()).decode("utf8")

    #     self.items.append(
    #         {
    #             "page": page_num,
    #             "type": "image",
    #             "path": image_path,
    #             "image": encoded_image,
    #             # "text": ocr_text if ocr_text else "(No caption extracted)",
    #         }
    #     )

    def process_images(self, page, page_num):
        images = page.get_images()
        for idx, image in enumerate(images):
            xref = image[0]
            pix = pymupdf.Pixmap(self.doc, xref)
            if pix.n > 4:  # CMYK or other
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)

            image_path = f"{self.base_dir}/images/{self.filename}_image_{page_num}_{idx}_{xref}.png"
            pix.save(image_path)

            with open(image_path, "rb") as f:
                encoded_image = base64.b64encode(f.read()).decode("utf8")

            # 🔍 Generate summary using Groq vision LLM
            try:
                summary = image_summary_chain.invoke(encoded_image)
            except Exception as e:
                print(f"[!] Image summary failed for page {page_num}, image {idx}: {e}")
                summary = "(No caption extracted)"

            self.items.append(
                {
                    "page": page_num,
                    "type": "image",
                    "path": image_path,
                    "image": encoded_image,
                    "text": summary,  # ✅ Critical for embedding & RAG later
                }
            )

    def process_page_images(self, page, page_num):
        pix = page.get_pixmap()
        page_path = f"{self.base_dir}/page_images/page_{page_num:03d}.png"
        pix.save(page_path)
        with open(page_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf8")
        self.items.append(
            {"page": page_num, "type": "page", "path": page_path, "image": encoded}
        )


# ========== Generate Embeddings ==========
def generate_ollama_embedding(text: str, model="nomic-embed-text") -> list:
    if not text:
        return None
    response = ollama_embeddings(model=model, prompt=text)
    return response["embedding"]


# ========== Save Embedding DB ==========
def save_embedding_database(
    items, db_name, base_dir="dbs", embedding_model="nomic-embed-text", dim=768
):
    db_path = os.path.join(base_dir, db_name)
    os.makedirs(db_path, exist_ok=True)

    embeddings_list = []
    valid_items = []

    for item in tqdm(items, desc="Generating Ollama embeddings"):
        # Embed only items with text for meaningful embeddings
        if item["type"] in ["text", "table", "image"]:
            text_to_embed = item.get("text")
            if not text_to_embed:
                continue
            embedding = generate_ollama_embedding(text_to_embed, model=embedding_model)
            if embedding is None:
                continue
            item["embedding"] = embedding
            embeddings_list.append(np.array(embedding, dtype=np.float32))
            valid_items.append(item)

    # Save metadata
    with open(os.path.join(db_path, "items.pkl"), "wb") as f:
        pickle.dump(valid_items, f)

    # Save FAISS index
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings_list, dtype=np.float32))
    faiss.write_index(index, os.path.join(db_path, "index.faiss"))

    print(f"[✓] Database saved to '{db_path}'")
    return db_path


# ========== Load FAISS DB ==========
def load_embedding_database(db_path):
    index = faiss.read_index(os.path.join(db_path, "index.faiss"))
    with open(os.path.join(db_path, "items.pkl"), "rb") as f:
        items = pickle.load(f)
    return index, items


# ========== RAG Inference ==========
def query_embedding_database(query, db_path, k=5, model="nomic-embed-text"):
    index, items = load_embedding_database(db_path)

    query_emb = generate_ollama_embedding(query, model=model)
    query_vector = np.array(query_emb, dtype=np.float32).reshape(1, -1)

    distances, indices = index.search(query_vector, k)
    matched_items = [items[i] for i in indices.flatten()]

    # Build context combining text & image captions
    context = "\n---\n".join(
        f"[{item['type'].upper()} - Page {item['page']}]:\n{item['text']}"
        for item in matched_items
        if "text" in item
    )
    print(f"Context:\n{context}\n")

    import ollama

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Use the context to answer the question.",
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
    )

    return {
        "query": query,
        "answer": response["message"]["content"],
        "matches": matched_items,
    }


# ========== Helper to display images (Jupyter) ==========
def show_extracted_images(items, max_images=5):
    """
    Display extracted images from items with base64 encoding (for Jupyter notebooks).

    Parameters:
    - items (list): List of dictionaries with image data (must include 'type'=='image' or 'page' and 'image' key).
    - max_images (int): Maximum number of images to display.
    """
    try:
        from IPython.display import display, Image
    except ImportError:
        print("IPython not found, image display not supported in this environment.")
        return

    count = 0
    for item in items:
        if item.get("type") in ["image", "page"] and "image" in item:
            img_data = base64.b64decode(item["image"])
            display(Image(data=img_data))
            print(f"Page: {item['page']} | Type: {item['type']} | Path: {item['path']}")
            print(f"Caption: {item.get('text', '(No caption)')}\n")
            count += 1
            if count >= max_images:
                break
    if count == 0:
        print("No images found to display.")


def run_embedding_pipeline(
    file_url,
    collection_name="transformer",
    base_dir="data",
    db_base_dir="dbs",
    embedding_model="nomic-embed-text",
    dim=768,
):
    """
    Downloads PDF, processes and extracts content, creates embeddings, and saves to FAISS DB.

    Args:
        file_url (str): Public URL of the PDF file.
        collection_name (str): Folder name for this specific embedding collection.
        base_dir (str): Directory to store extracted files.
        db_base_dir (str): Directory where FAISS index and metadata will be stored.
        embedding_model (str): Ollama embedding model to use.
        dim (int): Dimensionality of embeddings (should match model used).
    """
    # Process and extract PDF content
    processor = PDFProcessor(
        url=file_url, base_dir=os.path.join(base_dir, collection_name)
    )
    items = processor.process_pdf()

    # Save to FAISS
    db_path = save_embedding_database(
        items=items,
        db_name=collection_name,
        base_dir=db_base_dir,
        embedding_model=embedding_model,
        dim=dim,
    )

    return {
        "status": "success",
        "message": f"Embeddings stored in '{db_path}'",
        "db_path": db_path,
    }


def run_rag_query(
    question,
    collection_name="transformer",
    db_base_dir="dbs",
    embedding_model="nomic-embed-text",
    llm_model="llama3.2:3b",
    k=5,
):
    """
    Query the FAISS database for relevant content and generate an answer with Ollama LLM.

    Args:
        question (str): User's natural language question.
        collection_name (str): Name of the embedded collection.
        db_base_dir (str): Path where FAISS index and metadata are stored.
        embedding_model (str): Ollama embedding model.
        llm_model (str): Ollama LLM to use for answering.
        k (int): Number of similar chunks to retrieve.
    """
    db_path = os.path.join(db_base_dir, collection_name)
    index, items = load_embedding_database(db_path)

    query_emb = generate_ollama_embedding(question, model=embedding_model)
    query_vector = np.array(query_emb, dtype=np.float32).reshape(1, -1)

    distances, indices = index.search(query_vector, k)
    matched_items = [items[i] for i in indices.flatten()]

    # Build context
    context = "\n---\n".join(
        f"[{item['type'].upper()} - Page {item['page']}]:\n{item['text']}"
        for item in matched_items
        if "text" in item
    )

    import ollama

    response = ollama.chat(
        model=llm_model,
        messages=[
            {
                "role": "system",
                "content": """You are an expert bot. Use only the retrieved context to answer. 
      Revise the following text to make it more direct and professional by removing phrases like 'the context does not specifically mention', 'the context also mentions', or any similar wording. Focus on clearly stating the services and features. Keep the tone business-appropriate and informative.
      Instructions:
      - Refer char history to be context aware.
      - Avoid using phrases such as "according to the context", "as per the context", or similar contextual qualifiers.
      - Answer using ONLY context provided. 
      - For response which involve steps markdown does.
      - Try to answer each question from provided information rather than just saying ignoring. 
      - If exact answer is not available like for service in one country answer about that service available anywere.
      - If user is doing small talk do talk on small talk for hello say Hi how i can help.""",
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )

    return response["message"]["content"], {
        "query": question,
        "context": context,
        "answer": response["message"]["content"],
        "matches": matched_items,
    }
