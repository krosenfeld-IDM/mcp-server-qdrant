from mcp_server_qdrant.mcp_server import ChromaMCPServer
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    ChromaSettings,
    ToolSettings,
)

mcp = ChromaMCPServer(
    tool_settings=ToolSettings(),
    chroma_settings=ChromaSettings(),
    embedding_provider_settings=EmbeddingProviderSettings(),
)
