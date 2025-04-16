# MCP Server for ChromaDB

This is a MCP server that uses ChromaDB as a vector database to store and retrieve context.

## Features

- Store and retrieve information using vector similarity search
- Support for metadata with stored information
- Multiple collections support
- FastEmbed-based embeddings
- Configurable search limits
- Read-only mode support

## Installation

```bash
pip install mcp-server-chroma
```

## Usage

### Basic Usage

```python
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
```

### Configuration

The server can be configured using environment variables:

- `CHROMA_URL`: The URL of the ChromaDB server (optional, defaults to local mode)
- `CHROMA_API_KEY`: The API key for the ChromaDB server (optional)
- `COLLECTION_NAME`: The name of the default collection to use (optional)
- `CHROMA_LOCAL_PATH`: The path to the storage directory for the ChromaDB client (optional)
- `CHROMA_SEARCH_LIMIT`: The maximum number of results to return (default: 10)
- `CHROMA_READ_ONLY`: Whether to run in read-only mode (default: false)
- `EMBEDDING_PROVIDER`: The embedding provider to use (default: fastembed)
- `EMBEDDING_MODEL`: The model to use for embeddings (default: sentence-transformers/all-MiniLM-L6-v2)

### Tools

The server provides two main tools:

1. `chroma-store`: Store information in the vector database
   ```python
   await ctx.chroma_store("Some information to store", metadata={"source": "test"})
   ```

2. `chroma-find`: Find information in the vector database
   ```python
   results = await ctx.chroma_find("query to search for")
   ```

## Development

### Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

### Testing

Run the tests with:

```bash
pytest
```

### Linting

Run the linters with:

```bash
ruff check .
mypy .
```

## License

Apache 2.0
