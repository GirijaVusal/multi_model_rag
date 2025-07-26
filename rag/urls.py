from django.urls import path
from .views import ChatBotView, CreateEmbeddingView

urlpatterns = [
    path("chat/ask/", ChatBotView.as_view(), name="chat-bot"),
    path("store_embedding/", CreateEmbeddingView.as_view(), name="chat-bot"),
]
