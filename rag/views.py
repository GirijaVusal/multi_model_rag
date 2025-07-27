from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .pipeline import store_embedding,rag


class ChatBotView(APIView):
    def post(self, request):
        user_message = request.data.get("message")
        collection_name = request.data.get("collection_name")

        if not user_message:
            return Response(
                {"error": "No message provided."}, status=status.HTTP_400_BAD_REQUEST
            )

        if not collection_name:
            return Response(
                {"error": "No collection name provided."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        # call vdb and get chuck
        # prompt

        # TODO: Call your LLM here
        bot_response = self.call_llm(user_message,collection_name)

        return Response({"user": user_message, "bot": bot_response})

    def call_llm(self, message,collection_name):
        # Replace this with your LLM call (e.g., OpenAI API, local model, etc.)
        # For now, let's simulate a response
        return rag(message,collection_name)


class CreateEmbeddingView(APIView):
    def post(self, request):
        file_url = request.data.get("file_url")
        print("=================")
        print(file_url)
        images_background_context = request.data.get("images_background_context")
        collection_name = request.data.get("collection_name")

        # filename: str = "downloaded.pdf", need to parse from file_url
        # output_dir: str = ("./input_files",)
        # persist_directory: str = ("./chroma_db",)
        # docstore_path: str = ("./docstore",)
        # id_key: str = ("doc_id",)

        if not file_url:
            return Response(
                {"error": "No Document provided."}, status=status.HTTP_400_BAD_REQUEST
            )

        if not collection_name:
            return Response(
                {"error": "No collection name provided."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if not images_background_context:
            images_background_context = " "
            # return Response(
            #     {"error": "Please provide why type of images does this document contain."}, status=status.HTTP_400_BAD_REQUEST
            # )

        # TODO: Call your LLM here
        response = self.create_emdedding(
            file_url, collection_name, images_background_context
        )

        return Response({"Response": response}, status=status.HTTP_200_OK)

    def create_emdedding(self, document, collection_name, images_background_context):
        response = store_embedding(document, collection_name, images_background_context)
        # response = True
        if response:
            return f"Vector db is creeted with collection name  {collection_name}. You need to use this when chatting with your document"
        else:

            return f"Some thing went wrong"
