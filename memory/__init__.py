from typing import Callable, Any, Dict, List, Optional

from pydantic import BaseModel, Extra, Field
from langchain.retrievers import SVMRetriever, KNNRetriever
from langchain.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel
from langchain.embeddings.base import Embeddings

from .episode import Trajectory

class MPNetEmbeddings(BaseModel, Embeddings):
    """Wrapper around sentence_transformers MPNet embedding model."""

    client: Any  #: :meta private:
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    """Model name to use."""
    cache_folder: Optional[str] = None
    """Path to store models."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Key word arguments to pass to the model."""
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Key word arguments to pass when calling the `encode` method of the model."""

    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Could not import sentence_transformers python package. "
                "Please install it with `pip install sentence_transformers`."
            ) from exc

        self.client = SentenceTransformer(
            model_name_or_path=self.model_name,
            cache_folder=self.cache_folder,
            **self.model_kwargs
        )

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using MPNet model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        embeddings = self.client.encode(texts, **self.encode_kwargs)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using MPNet model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        text = text.replace("\n", " ")
        embedding = self.client.encode(text, **self.encode_kwargs)
        return embedding.tolist()

def choose_embedder(key: str) -> Callable:
    if key == 'mpnet':
        return MPNetEmbeddings
    return HuggingFaceEmbeddings

def choose_retriever(key: str) -> Callable:
    if key == 'knn':
        return KNNRetriever
    return SVMRetriever

EMBEDDERS = choose_embedder
RETRIEVERS = choose_retriever