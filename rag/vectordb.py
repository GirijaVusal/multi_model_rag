import uuid
from langchain.vectorstores import Chroma
from langchain.storage import LocalFileStore
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.embeddings import OllamaEmbeddings
import base64

embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")

# from langchain

# def get_vdb_retriever(
#     collection_name: str = "multi_modal_rag", id_key="doc_id"
# ) -> MultiVectorRetriever:
#     vectorstore = Chroma(
#         collection_name="multi_modal_rag", embedding_function=OpenAIEmbeddings()
#     )
#     store = InMemoryStore()

#     # The retriever (empty to start)
#     retriever = MultiVectorRetriever(
#         vectorstore=vectorstore,
#         docstore=store,
#         id_key=id_key,
#     )
#     retriever


def get_vdb_retriever(
    collection_name: str = "multi_modal_rag",
    persist_directory: str = "./chroma_db",
    docstore_path: str = "./docstore",
    id_key: str = "doc_id",
) -> MultiVectorRetriever:
    """
    Initializes and returns a persistent MultiVectorRetriever using Chroma for vector storage
    and LocalFileStore for document storage.

    This setup enables storing embeddings and associated documents on the file system,
    ensuring data persists across sessions.

    Args:
        collection_name (str): Name of the Chroma collection to use or create.
        persist_directory (str): Path to the directory where Chroma will store vectors.
        docstore_path (str): Path to the directory where documents will be stored as files.
        id_key (str): The key used to identify documents uniquely.

    Returns:
        MultiVectorRetriever: A retriever capable of storing and retrieving multimodal documents
        using both vector similarity and document ID-based lookups.
    """
    # Persistent Chroma vector store
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_directory,
    )

    # Persistent document store on file system
    store = LocalFileStore(docstore_path)

    # Initialize the retriever with persistent vectorstore and docstore
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    return retriever


def upsert(
    retriever: MultiVectorRetriever,
    texts: list = None,
    text_summaries: list = None,
    tables: list = None,
    table_summaries: list = None,
    images: list = None,
    image_summaries: list = None,
    id_key: str = "doc_id",
) -> bool:
    # Add texts

    if texts and text_summaries:
        doc_ids = [str(uuid.uuid4()) for _ in texts]
        clean_summaries = []
        clean_ids = []
        for i, summary in enumerate(text_summaries):
            if isinstance(summary, str):
                clean_summaries.append(summary)
                clean_ids.append(doc_ids[i])
            else:
                try:
                    clean_summaries.append(str(summary))
                    clean_ids.append(doc_ids[i])
                except Exception as e:
                    print(f"[WARN] Skipping invalid summary at index {i}: {e}")

        summary_texts = [
            Document(page_content=s, metadata={id_key: clean_ids[idx]})
            for idx, s in enumerate(clean_summaries)
        ]

        retriever.vectorstore.add_documents(summary_texts)
        # byte_texts = [t.encode("utf-8") for t in texts[:len(clean_ids)]]
        byte_texts = [str(t).encode("utf-8") for t in texts[: len(clean_ids)]]
        retriever.docstore.mset(list(zip(clean_ids, byte_texts)))
        print("Text upserted")

    # if texts:
    #     doc_ids = [str(uuid.uuid4()) for _ in texts]
    #     summary_texts = [
    #         Document(page_content=summary, metadata={id_key: doc_ids[i]})
    #         for i, summary in enumerate(text_summaries)
    #     ]
    #     retriever.vectorstore.add_documents(summary_texts)
    #     retriever.docstore.mset(list(zip(doc_ids, [t.encode("utf-8") for t in texts])))
    #     print("Text upserted")

    # # Add tables
    # if tables:
    #     table_ids = [str(uuid.uuid4()) for _ in tables]
    #     summary_tables = [
    #         Document(page_content=summary, metadata={id_key: table_ids[i]})
    #         for i, summary in enumerate(table_summaries)
    #     ]
    #     retriever.vectorstore.add_documents(summary_tables)
    #     retriever.docstore.mset(list(zip(table_ids, tables)))
    #     print("Table upserted")

    if tables and table_summaries:
        doc_ids = [str(uuid.uuid4()) for _ in texts]
        clean_summaries = []
        clean_ids = []
        for i, summary in enumerate(text_summaries):
            if isinstance(summary, str):
                clean_summaries.append(summary)
                clean_ids.append(doc_ids[i])
            else:
                try:
                    clean_summaries.append(str(summary))
                    clean_ids.append(doc_ids[i])
                except Exception as e:
                    print(f"[WARN] Skipping invalid summary at index {i}: {e}")

            summary_texts = [
                Document(page_content=s, metadata={id_key: clean_ids[idx]})
                for idx, s in enumerate(clean_summaries)
            ]

        retriever.vectorstore.add_documents(summary_texts)
        byte_texts = [t.encode("utf-8") for t in texts[: len(clean_ids)]]
        retriever.docstore.mset(list(zip(clean_ids, byte_texts)))
        print("Text upserted")

    # #Add image summaries
    if images:
        img_ids = [str(uuid.uuid4()) for _ in images]
        summary_img = [
            Document(page_content=summary, metadata={id_key: img_ids[i]})
            for i, summary in enumerate(image_summaries)
        ]
        retriever.vectorstore.add_documents(summary_img)

        image_bytes = [base64.b64decode(img_str) for img_str in images]
        retriever.docstore.mset(list(zip(img_ids, image_bytes)))

        print("Images Upserted.")

    # if images:
    # img_ids = [str(uuid.uuid4()) for _ in images]
    # summary_img = [
    #     Document(page_content=summary, metadata={id_key: img_ids[i]})
    #     for i, summary in enumerate(image_summaries)
    # ]
    # retriever.vectorstore.add_documents(summary_img)

    # # Convert base64 strings to bytes
    # image_bytes = [base64.b64decode(img_str) for img_str in images]
    # retriever.docstore.mset(list(zip(img_ids, image_bytes)))

    return True


# def upsert(
#     retriever: MultiVectorRetriever,
#     texts: list = None,
#     text_summaries: list = None,
#     tables: list = None,
#     table_summaries: list = None,
#     images: list = None,
#     image_summaries: list = None,
#     id_key: str = "doc_id",
# ) -> bool:
#     # Add texts
#     if texts:
#         doc_ids = [str(uuid.uuid4()) for _ in texts]
#         summary_texts = [
#             Document(page_content=summary, metadata={id_key: doc_ids[i]})
#             for i, summary in enumerate(text_summaries)
#         ]
#         retriever.vectorstore.add_documents(summary_texts)
#         retriever.docstore.mset(list(zip(doc_ids, [t.encode("utf-8") for t in texts])))
#         print("Text upserted")

#     # Add tables
#     if tables:
#         table_ids = [str(uuid.uuid4()) for _ in tables]
#         summary_tables = [
#             Document(page_content=summary, metadata={id_key: table_ids[i]})
#             for i, summary in enumerate(table_summaries)
#         ]
#         retriever.vectorstore.add_documents(summary_tables)
#         retriever.docstore.mset(list(zip(table_ids, tables)))
#         print("Table upserted")

#     # Add image summaries
#     if images:
#         img_ids = [str(uuid.uuid4()) for _ in images]
#         summary_img = [
#             Document(page_content=summary, metadata={id_key: img_ids[i]})
#             for i, summary in enumerate(image_summaries)
#         ]
#         retriever.vectorstore.add_documents(summary_img)
#         retriever.docstore.mset(list(zip(img_ids, images)))
#         print("Images Upserted.")
#     return True
