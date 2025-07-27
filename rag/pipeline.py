from .utils import download_pdf, parse_text_table_images_from_pdf
from .vectordb import get_vdb_retriever, upsert
from .preprocess import generate_table_text_summary, generate_images_summary
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from base64 import b64decode
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import pickle
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    max_tokens=512,
    api_key=os.getenv("GROQ_API_KEY"))

def store_embedding(
    file_url: str,
    collection_name: str = "multi_modal_rag",
    images_background_context: str = " ",
    output_dir: str = "./input_files",
    filename: str = "downloaded.pdf",
    persist_directory: str = "./chroma_db",
    docstore_path: str = "./docstore",
    id_key: str = "doc_id",
):
    file_path = download_pdf(file_url, output_dir, filename)
    if file_path is not None:
        texts, tables, images = parse_text_table_images_from_pdf(file_path)
        retriever = get_vdb_retriever(
            collection_name, persist_directory, docstore_path, id_key
        )
        text_summaries, table_summaries = generate_table_text_summary(texts, tables)
        image_summaries = generate_images_summary(images, images_background_context)

        output_dir = "pdf_pickle_output"
        os.makedirs(output_dir, exist_ok=True)
        output_pickle_path = os.path.join(output_dir, "summary.pkl")
        with open(output_pickle_path, "wb") as f:
            pickle.dump(
                {
                    "table_summaries": table_summaries,
                    "image_summaries": image_summaries,
                },
                f,
            )

        upsert(
            retriever,
            texts,
            text_summaries,
            tables,
            table_summaries,
            images,
            image_summaries,
            id_key,
        )

        return True
    return False

def parse_docs(docs):
    """Split base64-encoded images and texts"""
    b64 = []
    text = []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception as e:
            text.append(doc)
    # print()
    # return {"images": b64, "texts": text}
    return {"texts": text}

def inference(
    query: str,
    collection_name: str,
    persist_directory: str,
    docstore_path: str,
    id_key: str,
):

    retriever = get_vdb_retriever(
        collection_name, persist_directory, docstore_path, id_key
    )
    # Retrieve
    docs = retriever.invoke(query)
    docs = parse_docs(docs)
    return docs






def build_prompt(**kwargs):

    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    context_text = ""
    if len(docs_by_type["texts"]) > 0:
        for text_element in docs_by_type["texts"]:
            context_text += text_element.text

    # construct prompt with context (including images)
    prompt_template = f"""
    Answer the question based only on the following context, which can include text, tables, and the below image.
    Context: {context_text}
    Question: {user_question}
    """

    prompt_content = [{"type": "text", "text": prompt_template}]

    if len(docs_by_type["images"]) > 0:
        for image in docs_by_type["images"]:
            prompt_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            )

    return ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=prompt_content),
        ]
    )


def rag(
    query: str,
    collection_name: str = "multi_modal_rag",
    persist_directory: str = "./chroma_db",
    docstore_path: str = "./docstore",
    id_key: str = "doc_id",
):
    out = inference(
        query,
        collection_name,
        persist_directory,
        docstore_path,
        id_key,)
    
    prompt_template = f"""
    Answer the question based only on the following context, which can include text, tables, and the below image.
    Context: {out}
    User Question: {query}
    """

    out = llm.invoke(prompt_template)
    
    print("=====================================")
    print(out)
       # retriever = get_vdb_retriever(
    #     collection_name, persist_directory, docstore_path, id_key
    # )

    # chain = (
    #     {
    #         "context": retriever | RunnableLambda(parse_docs),
    #         "question": RunnablePassthrough(),
    #     }
    #     | RunnableLambda(build_prompt)
    #     | llm
    #     | StrOutputParser()
    # )

    # chain_with_sources = {
    #     "context": retriever | RunnableLambda(parse_docs),
    #     "question": RunnablePassthrough(),
    # } | RunnablePassthrough().assign(
    #     response=(
    #         RunnableLambda(build_prompt)
    #         | ChatOpenAI(model="gpt-4o-mini")
    #         | StrOutputParser()
    #     )
    # )

    # response = chain.invoke(query)

    # print(response)

    return out.content


if __name__ =="__main__":
    query = "What is pooling layer"
    collection_name = "rcnn"
    persist_directory = "./chroma_db"
    docstore_path = "./docstore"
    id_key = "doc_id"

    # out = inference(
    #     query,
    #     collection_name,
    #     persist_directory,
    #     docstore_path,
    #     id_key,
    # )

    '''
    
    out = rag(
        query,
        collection_name,
        persist_directory,
        docstore_path,
        id_key,
    )
    
    print("=====================Retrieved documents:====================")
    print(len(out))

    print(parse_docs(out))'''
    print("=====================================")
    # print(llm.invoke("how are you doing?"))
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=os.getenv("GROQ_API_KEY"))
    messages = [
    (
        "system",
        "You are a helpful assistant.",
    ),
    ("human", "Can you tell me about pooling layers in CNNs?"),
]
    ai_msg = llm.invoke(messages)
    print(ai_msg)