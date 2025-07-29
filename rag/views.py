from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .pipeline import run_embedding_pipeline, run_rag_query


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

        final_answer, full_context = self.call_llm(user_message, collection_name)

        return Response({"user": user_message, "bot": final_answer})

    def call_llm(self, message, collection_name):
        final_answer, full_context = run_rag_query(message, collection_name)
        return final_answer, full_context


class CreateEmbeddingView(APIView):
    def post(self, request):
        file_url = request.data.get("file_url")
        collection_name = request.data.get("collection_name")
        if not file_url:
            return Response(
                {"error": "No Document provided."}, status=status.HTTP_400_BAD_REQUEST
            )

        if not collection_name:
            return Response(
                {"error": "No collection name provided."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        response = self.create_emdedding(file_url, collection_name)

        return Response({"message": response}, status=status.HTTP_200_OK)

    def create_emdedding(self, document, collection_name):
        response = run_embedding_pipeline(document, collection_name)
        # response = True
        if response:
            return f"Vector db is creeted with collection name  {collection_name}. You need to use this when chatting with your document"
        else:

            return f"Some thing went wrong"
