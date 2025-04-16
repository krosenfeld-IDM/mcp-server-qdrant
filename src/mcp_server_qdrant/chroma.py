import logging
from typing import Any, Dict, Optional

import chromadb
from chromadb.config import Settings
from pydantic import BaseModel

from mcp_server_qdrant.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)

Metadata = Dict[str, Any]


class Entry(BaseModel):
    """
    A single entry in the ChromaDB collection.
    """
    content: str
    metadata: Optional[Metadata] = None


class ChromaConnector:
    """
    Encapsulates the connection to a ChromaDB server and all the methods to interact with it.
    :param chroma_url: The URL of the ChromaDB server.
    :param chroma_api_key: The API key to use for the ChromaDB server.
    :param collection_name: The name of the default collection to use. If not provided, each tool will require
                            the collection name to be provided.
    :param embedding_provider: The embedding provider to use.
    :param chroma_local_path: The path to the storage directory for the ChromaDB client, if local mode is used.
    """

    def __init__(
        self,
        chroma_url: Optional[str],
        chroma_api_key: Optional[str],
        collection_name: Optional[str],
        embedding_provider: EmbeddingProvider,
        chroma_local_path: Optional[str] = None,
    ):
        self._chroma_url = chroma_url.rstrip("/") if chroma_url else None
        self._chroma_api_key = chroma_api_key
        self._default_collection_name = collection_name
        self._embedding_provider = embedding_provider

        # Initialize ChromaDB client
        if chroma_url:
            self._client = chromadb.HttpClient(
                host=chroma_url,
                port=8000,  # Default ChromaDB port
                settings=Settings(
                    chroma_api_impl="rest",
                    chroma_server_host=chroma_url,
                    chroma_server_http_port=8000,
                )
            )
        else:
            self._client = chromadb.PersistentClient(path=chroma_local_path or "./chroma_db")

    async def get_collection_names(self) -> list[str]:
        """
        Get the names of all collections in the ChromaDB server.
        :return: A list of collection names.
        """
        return [collection.name for collection in self._client.list_collections()]

    async def store(self, entry: Entry, *, collection_name: Optional[str] = None):
        """
        Store some information in the ChromaDB collection, along with the specified metadata.
        :param entry: The entry to store in the ChromaDB collection.
        :param collection_name: The name of the collection to store the information in, optional. If not provided,
                                the default collection is used.
        """
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None
        collection = await self._ensure_collection_exists(collection_name)

        # Embed the document
        embeddings = await self._embedding_provider.embed_documents([entry.content])

        # Add to ChromaDB
        collection.add(
            embeddings=embeddings,
            documents=[entry.content],
            metadatas=[entry.metadata] if entry.metadata else None,
            ids=[str(len(collection.get()["ids"]))]  # Simple auto-incrementing ID
        )

    async def search(
        self, query: str, *, collection_name: Optional[str] = None, limit: int = 10
    ) -> list[Entry]:
        """
        Find points in the ChromaDB collection. If there are no entries found, an empty list is returned.
        :param query: The query to use for the search.
        :param collection_name: The name of the collection to search in, optional. If not provided,
                                the default collection is used.
        :param limit: The maximum number of entries to return.
        :return: A list of entries found.
        """
        collection_name = collection_name or self._default_collection_name
        try:
            collection = self._client.get_collection(collection_name)
        except ValueError:
            return []

        # Embed the query
        query_vector = await self._embedding_provider.embed_query(query)

        # Search in ChromaDB
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=limit
        )

        entries = []
        for i in range(len(results["documents"][0])):
            entries.append(
                Entry(
                    content=results["documents"][0][i],
                    metadata=results["metadatas"][0][i] if results["metadatas"][0] else None
                )
            )
        return entries

    async def _ensure_collection_exists(self, collection_name: str):
        """
        Ensure that the collection exists, creating it if necessary.
        :param collection_name: The name of the collection to ensure exists.
        """
        try:
            collection = self._client.get_collection(collection_name)
        except ValueError:
            # Create the collection if it doesn't exist
            collection = self._client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
        return collection 