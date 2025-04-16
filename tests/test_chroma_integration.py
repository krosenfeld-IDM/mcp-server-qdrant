import uuid

import pytest

from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider
from mcp_server_qdrant.chroma import Entry, ChromaConnector


@pytest.fixture
async def embedding_provider():
    """Fixture to provide a FastEmbed embedding provider."""
    return FastEmbedProvider(model_name="sentence-transformers/all-MiniLM-L6-v2")


@pytest.fixture
async def chroma_connector(embedding_provider):
    """Fixture to provide a ChromaConnector with in-memory ChromaDB client."""
    # Use a random collection name to avoid conflicts between tests
    collection_name = f"test_collection_{uuid.uuid4().hex}"

    # Create connector with in-memory ChromaDB
    connector = ChromaConnector(
        chroma_url=None,
        chroma_api_key=None,
        collection_name=collection_name,
        embedding_provider=embedding_provider,
        chroma_local_path=":memory:"
    )

    yield connector


@pytest.mark.asyncio
async def test_store_and_search(chroma_connector):
    """Test storing an entry and then searching for it."""
    # Store a test entry
    test_entry = Entry(
        content="The quick brown fox jumps over the lazy dog",
        metadata={"source": "test", "importance": "high"},
    )
    await chroma_connector.store(test_entry)

    # Search for the entry
    results = await chroma_connector.search("fox jumps")

    # Verify results
    assert len(results) == 1
    assert results[0].content == test_entry.content
    assert results[0].metadata == test_entry.metadata


@pytest.mark.asyncio
async def test_search_empty_collection(chroma_connector):
    """Test searching in an empty collection."""
    # Search in an empty collection
    results = await chroma_connector.search("test query")

    # Verify results
    assert len(results) == 0


@pytest.mark.asyncio
async def test_multiple_entries(chroma_connector):
    """Test storing and searching multiple entries."""
    # Store multiple entries
    entries = [
        Entry(
            content="Python is a programming language",
            metadata={"topic": "programming"},
        ),
        Entry(content="The Eiffel Tower is in Paris", metadata={"topic": "landmarks"}),
        Entry(content="Machine learning is a subset of AI", metadata={"topic": "AI"}),
    ]

    for entry in entries:
        await chroma_connector.store(entry)

    # Search for programming-related entries
    programming_results = await chroma_connector.search("Python programming")
    assert len(programming_results) > 0
    assert any("Python" in result.content for result in programming_results)

    # Search for landmark-related entries
    landmark_results = await chroma_connector.search("Eiffel Tower Paris")
    assert len(landmark_results) > 0
    assert any("Eiffel" in result.content for result in landmark_results)

    # Search for AI-related entries
    ai_results = await chroma_connector.search(
        "artificial intelligence machine learning"
    )
    assert len(ai_results) > 0
    assert any("machine learning" in result.content.lower() for result in ai_results)


@pytest.mark.asyncio
async def test_ensure_collection_exists(chroma_connector):
    """Test that the collection is created if it doesn't exist."""
    # The collection shouldn't exist yet
    try:
        chroma_connector._client.get_collection(chroma_connector._default_collection_name)
        collection_exists = True
    except ValueError:
        collection_exists = False
    assert not collection_exists

    # Storing an entry should create the collection
    test_entry = Entry(content="Test content")
    await chroma_connector.store(test_entry)

    # Now the collection should exist
    collection = chroma_connector._client.get_collection(chroma_connector._default_collection_name)
    assert collection is not None


@pytest.mark.asyncio
async def test_metadata_handling(chroma_connector):
    """Test that metadata is properly stored and retrieved."""
    # Store entries with different metadata
    metadata1 = {"source": "book", "author": "Jane Doe", "year": 2023}
    metadata2 = {"source": "article", "tags": ["science", "research"]}

    await chroma_connector.store(
        Entry(content="Content with structured metadata", metadata=metadata1)
    )
    await chroma_connector.store(
        Entry(content="Content with list in metadata", metadata=metadata2)
    )

    # Search and verify metadata is preserved
    results = await chroma_connector.search("metadata")

    assert len(results) == 2

    # Check that both metadata objects are present in the results
    found_metadata1 = False
    found_metadata2 = False

    for result in results:
        if result.metadata.get("source") == "book":
            assert result.metadata.get("author") == "Jane Doe"
            assert result.metadata.get("year") == 2023
            found_metadata1 = True
        elif result.metadata.get("source") == "article":
            assert "science" in result.metadata.get("tags", [])
            assert "research" in result.metadata.get("tags", [])
            found_metadata2 = True

    assert found_metadata1
    assert found_metadata2


@pytest.mark.asyncio
async def test_entry_without_metadata(chroma_connector):
    """Test storing and retrieving entries without metadata."""
    # Store an entry without metadata
    await chroma_connector.store(Entry(content="Entry without metadata"))

    # Search and verify
    results = await chroma_connector.search("without metadata")

    assert len(results) == 1
    assert results[0].content == "Entry without metadata"
    assert results[0].metadata is None


@pytest.mark.asyncio
async def test_custom_collection_store_and_search(chroma_connector):
    """Test storing and searching in a custom collection."""
    # Define a custom collection name
    custom_collection = f"custom_collection_{uuid.uuid4().hex}"

    # Store a test entry in the custom collection
    test_entry = Entry(
        content="This is stored in a custom collection",
        metadata={"custom": True},
    )
    await chroma_connector.store(test_entry, collection_name=custom_collection)

    # Search in the custom collection
    results = await chroma_connector.search(
        "custom collection", collection_name=custom_collection
    )

    # Verify results
    assert len(results) == 1
    assert results[0].content == test_entry.content
    assert results[0].metadata == test_entry.metadata

    # Verify the entry is not in the default collection
    default_results = await chroma_connector.search("custom collection")
    assert len(default_results) == 0


@pytest.mark.asyncio
async def test_nonexistent_collection_search(chroma_connector):
    """Test searching in a collection that doesn't exist."""
    # Search in a collection that doesn't exist
    nonexistent_collection = f"nonexistent_{uuid.uuid4().hex}"
    results = await chroma_connector.search(
        "test query", collection_name=nonexistent_collection
    )

    # Verify results
    assert len(results) == 0 