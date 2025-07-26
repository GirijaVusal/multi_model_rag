from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.runnables import RunnableSequence


load_dotenv()
api_key = os.environ["GROQ_API_KEY"]


def generate_table_text_summary(texts, tables):
    prompt_text = """
    You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text.

    Respond only with the summary, no additionnal comment.
    Do not start your message by saying "Here is a summary" or anything like that.
    Just give the summary as it is.

    Table or text chunk: {element}

    """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Summary chain
    model = ChatGroq(temperature=0.5, model="llama-3.1-8b-instant", api_key=api_key)
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    # Summarize text
    # text_summaries = summarize_chain.batch(texts, {"max_concurrency": 3})

    # Summarize tables
    tables_html = [table for table in tables]
    table_summaries = summarize_chain.batch(tables_html, {"max_concurrency": 3})

    return texts, table_summaries


def generate_images_summary(images, images_background_context: str = None):
    prompt_template = f"""Describe the image in detail.{images_background_context}"""
    image_summaries = []
    vision_model = ChatGroq(
        model_name="meta-llama/llama-4-scout-17b-16e-instruct",
        api_key=api_key,
        temperature=0,
    )

    for image in images:
        try:
            messages = [
                (
                    "user",
                    [
                        {"type": "text", "text": prompt_template},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                        },
                    ],
                )
            ]

            prompt = ChatPromptTemplate.from_messages(messages)
            chain = prompt | vision_model | StrOutputParser()
            summary = chain.invoke({})
            image_summaries.append(summary)

        except Exception as e:
            print(f"[WARN] Skipping image due to error: {e}")
            continue

    print(image_summaries)
    return image_summaries


# def generate_images_summary(images, images_background_context: str = None):

#     prompt_template = f"""Describe the image in detail.{images_background_context}"""
#     messages = [
#         (
#             "user",
#             [
#                 {"type": "text", "text": prompt_template},
#                 {
#                     "type": "image_url",
#                     "image_url": {"url": "data:image/jpeg;base64,{image}"},
#                 },
#             ],
#         )
#     ]

#     prompt = ChatPromptTemplate.from_messages(messages)
# vision_model = ChatGroq(
#     model_name="meta-llama/llama-4-scout-17b-16e-instruct",
#     api_key=api_key,
#     temperature=0,
# )
#     chain = prompt | vision_model | StrOutputParser()

#     image_summaries = chain.batch(images)

#     return image_summaries
