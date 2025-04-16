import json
import logging
from typing import Any, List

from mcp.server.fastmcp import Context, FastMCP

from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.chroma import Entry, Metadata, ChromaConnector
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    ChromaSettings,
    ToolSettings,
)

logger = logging.getLogger(__name__)


# FastMCP is an alternative interface for declaring the capabilities
# of the server. Its API is based on FastAPI.
class ChromaMCPServer(FastMCP):
    """
    A MCP server for ChromaDB.
    """

    def __init__(
        self,
        tool_settings: ToolSettings,
        chroma_settings: ChromaSettings,
        embedding_provider_settings: EmbeddingProviderSettings,
        name: str = "mcp-server-chroma",
        instructions: str | None = None,
        **settings: Any,
    ):
        self.tool_settings = tool_settings
        self.chroma_settings = chroma_settings
        self.embedding_provider_settings = embedding_provider_settings

        self.embedding_provider = create_embedding_provider(embedding_provider_settings)
        self.chroma_connector = ChromaConnector(
            chroma_settings.location,
            chroma_settings.api_key,
            chroma_settings.collection_name,
            self.embedding_provider,
            chroma_settings.local_path,
        )

        super().__init__(name=name, instructions=instructions, **settings)

        self.setup_tools()

    def format_entry(self, entry: Entry) -> str:
        """
        Feel free to override this method in your subclass to customize the format of the entry.
        """
        entry_metadata = json.dumps(entry.metadata) if entry.metadata else ""
        return f"<entry><content>{entry.content}</content><metadata>{entry_metadata}</metadata></entry>"

    def setup_tools(self):
        """
        Register the tools in the server.
        """

        async def store(
            ctx: Context,
            information: str,
            collection_name: str,
            # The `metadata` parameter is defined as non-optional, but it can be None.
            # If we set it to be optional, some of the MCP clients, like Cursor, cannot
            # handle the optional parameter correctly.
            metadata: Metadata = None,  # type: ignore
        ) -> str:
            """
            Store some information in ChromaDB.
            :param ctx: The context for the request.
            :param information: The information to store.
            :param metadata: JSON metadata to store with the information, optional.
            :param collection_name: The name of the collection to store the information in, optional. If not provided,
                                    the default collection is used.
            :return: A message indicating that the information was stored.
            """
            await ctx.debug(f"Storing information {information} in ChromaDB")

            entry = Entry(content=information, metadata=metadata)

            await self.chroma_connector.store(entry, collection_name=collection_name)
            if collection_name:
                return f"Remembered: {information} in collection {collection_name}"
            return f"Remembered: {information}"

        async def store_with_default_collection(
            ctx: Context,
            information: str,
            metadata: Metadata = None,  # type: ignore
        ) -> str:
            assert self.chroma_settings.collection_name is not None
            return await store(
                ctx, information, self.chroma_settings.collection_name, metadata
            )

        async def find(
            ctx: Context,
            query: str,
            collection_name: str,
        ) -> List[str]:
            """
            Find memories in ChromaDB.
            :param ctx: The context for the request.
            :param query: The query to use for the search.
            :param collection_name: The name of the collection to search in, optional. If not provided,
                                    the default collection is used.
            :return: A list of entries found.
            """
            await ctx.debug(f"Finding results for query {query}")
            if collection_name:
                await ctx.debug(
                    f"Overriding the collection name with {collection_name}"
                )

            entries = await self.chroma_connector.search(
                query,
                collection_name=collection_name,
                limit=self.chroma_settings.search_limit,
            )
            if not entries:
                return [f"No information found for the query '{query}'"]
            content = [
                f"Results for the query '{query}'",
            ]
            for entry in entries:
                content.append(self.format_entry(entry))
            return content

        async def find_with_default_collection(
            ctx: Context,
            query: str,
        ) -> List[str]:
            assert self.chroma_settings.collection_name is not None
            return await find(ctx, query, self.chroma_settings.collection_name)

        # Register the tools depending on the configuration

        if self.chroma_settings.collection_name:
            self.add_tool(
                find_with_default_collection,
                name="chroma-find",
                description=self.tool_settings.tool_find_description,
            )
        else:
            self.add_tool(
                find,
                name="chroma-find",
                description=self.tool_settings.tool_find_description,
            )

        if not self.chroma_settings.read_only:
            # Those methods can modify the database

            if self.chroma_settings.collection_name:
                self.add_tool(
                    store_with_default_collection,
                    name="chroma-store",
                    description=self.tool_settings.tool_store_description,
                )
            else:
                self.add_tool(
                    store,
                    name="chroma-store",
                    description=self.tool_settings.tool_store_description,
                )
