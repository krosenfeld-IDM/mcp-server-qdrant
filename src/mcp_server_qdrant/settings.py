from enum import Enum
from typing import Optional

from pydantic import BaseSettings, Field


class EmbeddingProviderType(str, Enum):
    """
    The type of embedding provider to use.
    """

    FASTEMBED = "fastembed"


class ToolSettings(BaseSettings):
    """
    Configuration for the tools.
    """

    tool_find_description: str = Field(
        default="Find information in the vector database.",
        validation_alias="TOOL_FIND_DESCRIPTION",
    )
    tool_store_description: str = Field(
        default="Store information in the vector database.",
        validation_alias="TOOL_STORE_DESCRIPTION",
    )


class EmbeddingProviderSettings(BaseSettings):
    """
    Configuration for the embedding provider.
    """

    provider_type: EmbeddingProviderType = Field(
        default=EmbeddingProviderType.FASTEMBED,
        validation_alias="EMBEDDING_PROVIDER",
    )
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        validation_alias="EMBEDDING_MODEL",
    )


class ChromaSettings(BaseSettings):
    """
    Configuration for the ChromaDB connector.
    """

    location: Optional[str] = Field(default=None, validation_alias="CHROMA_URL")
    api_key: Optional[str] = Field(default=None, validation_alias="CHROMA_API_KEY")
    collection_name: Optional[str] = Field(
        default=None, validation_alias="COLLECTION_NAME"
    )
    local_path: Optional[str] = Field(
        default=None, validation_alias="CHROMA_LOCAL_PATH"
    )
    search_limit: int = Field(default=10, validation_alias="CHROMA_SEARCH_LIMIT")
    read_only: bool = Field(default=False, validation_alias="CHROMA_READ_ONLY")
