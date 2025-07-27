import os
import requests
# from unstructured.partition.pdf import partition_pdf
from langchain.document_loaders import UnstructuredFileLoader


def download_pdf(
    url: str, output_dir: str = ".", filename: str = "downloaded.pdf"
) -> bool:
    """
    Downloads a PDF from the given URL and saves it in the specified directory with the given filename.

    Parameters:
        url (str): The URL of the PDF file.
        output_dir (str): The directory to save the file. Defaults to current directory.
        filename (str): The name to give the saved PDF file. Defaults to 'downloaded.pdf'.

    Returns:
        filepath: filepath if download was successful, None otherwise.
    """
    try:
        response = requests.get(url)
        if response.status_code == 200 and "application/pdf" in response.headers.get(
            "Content-Type", ""
        ):
            os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
            output_path = os.path.join(output_dir, filename)
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"PDF downloaded successfully: {output_path}")
            return output_path
        else:
            print(
                f"Failed to download PDF. Status code: {response.status_code}, Content-Type: {response.headers.get('Content-Type')}"
            )
            return output_path
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def get_images_base64(chunks):
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64


def parse_text_table_images_from_pdf(file_path: str):
    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        skip_infer_table_types=False,
        strategy="hi_res",
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )

    texts = []
    tables = []
    for chunk in chunks:
        if "Table" in str(type(chunk)):
            tables.append(chunk)

        if "CompositeElement" in str(type((chunk))):
            texts.append(chunk)

    images = get_images_base64(chunks)

    final_texts = []
    final_tables = []
    for text in texts:
        td = text.to_dict()
        clean_text = td["text"]
        final_texts.append(clean_text)
        table = td.get("metadata", {}).get("text_as_html")
        # td["metadata"]["text_as_html"]
        if table:
            table = td["metadata"]["text_as_html"]
            final_tables.append(table)

    return final_texts, final_tables, images


if __name__ == "__main__":
    import pickle

    file_path = "https://arxiv.org/pdf/1506.01497"
    file_path = download_pdf(file_path)
    texts, tables, images = parse_text_table_images_from_pdf(file_path)

    # final_texts = []
    # final_tables = []
    # for text in texts:
    #     td = text.to_dict()
    #     clean_text = td["text"]
    #     final_texts.append(clean_text)
    #     table = td.get("metadata", {}).get("text_as_html")
    #     # td["metadata"]["text_as_html"]
    #     if table:
    #         table = td["metadata"]["text_as_html"]
    #         final_tables.append(table)

    # print(texts)
    # print(tables)
    # print(images)

    output_dir = "pdf_pickle_output"
    os.makedirs(output_dir, exist_ok=True)

    output_pickle_path = os.path.join(output_dir, "pdf_data.pkl")

    with open(output_pickle_path, "wb") as f:
        pickle.dump({"texts": texts, "tables": tables, "images": images}, f)

    print(f"Data saved to pickle file: {output_pickle_path}")
