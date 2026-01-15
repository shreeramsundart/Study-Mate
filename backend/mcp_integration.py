#!/usr/bin/env python3
"""
MCP Server for RAG AI Assistant
Run separately: python mcp_integration.py
"""
import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

try:
    from mcp.server import Server
    import mcp.types as types
    import mcp.server.stdio
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("⚠️ MCP package not available. Install with: pip install mcp")

# Import your RAG modules
try:
    from rag.main import process_document, process_query
    from rag.llm import generate_response
    from rag.mongodb import mongodb
    RAG_AVAILABLE = True
except Exception as e:
    print(f"⚠️ RAG system not available: {e}")
    RAG_AVAILABLE = False

class RAGMCPServer:
    def __init__(self):
        self.server = Server("rag-ai-tools")
        self.setup_tools()
    
    def setup_tools(self):
        """Setup all MCP tools"""
        
        @self.server.list_tools()
        async def handle_list_tools():
            tools = [
                types.Tool(
                    name="chat",
                    description="Chat with AI assistant",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "message": {"type": "string", "description": "Your message"},
                            "language": {"type": "string", "description": "Response language", "default": "English"}
                        },
                        "required": ["message"]
                    }
                ),
                types.Tool(
                    name="search_documents",
                    description="Search in uploaded documents",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        },
                        "required": ["query"]
                    }
                ),
                types.Tool(
                    name="summarize",
                    description="Summarize text",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Text to summarize"}
                        },
                        "required": ["text"]
                    }
                )
            ]
            
            if RAG_AVAILABLE:
                tools.append(
                    types.Tool(
                        name="process_document",
                        description="Process a PDF document",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "file_path": {"type": "string", "description": "Path to PDF file"}
                            },
                            "required": ["file_path"]
                        }
                    )
                )
            
            return tools
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> list:
            try:
                if name == "chat":
                    message = arguments.get("message", "")
                    language = arguments.get("language", "English")
                    
                    if RAG_AVAILABLE:
                        # Try to get response from RAG system
                        result = generate_response(message, "", language)
                        return [types.TextContent(type="text", text=result)]
                    else:
                        return [types.TextContent(type="text", text="RAG system not available")]
                
                elif name == "search_documents":
                    query = arguments.get("query", "")
                    if RAG_AVAILABLE:
                        # This would need to be implemented to search MongoDB
                        return [types.TextContent(type="text", text=f"Search for '{query}' in documents (not implemented)")]
                    else:
                        return [types.TextContent(type="text", text="RAG system not available")]
                
                elif name == "summarize":
                    text = arguments.get("text", "")
                    if RAG_AVAILABLE:
                        # Use LLM to summarize
                        prompt = f"Please provide a concise summary of this text:\n\n{text}"
                        result = generate_response(prompt, "", "English")
                        return [types.TextContent(type="text", text=result)]
                    else:
                        return [types.TextContent(type="text", text="AI model not available")]
                
                elif name == "process_document":
                    if RAG_AVAILABLE:
                        file_path = arguments.get("file_path", "")
                        try:
                            with open(file_path, 'rb') as f:
                                result = process_document(f, "mcp_user")
                            return [types.TextContent(type="text", text=str(result))]
                        except Exception as e:
                            return [types.TextContent(type="text", text=f"Error: {str(e)}")]
                    else:
                        return [types.TextContent(type="text", text="RAG system not available")]
                
                else:
                    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]
                
            except Exception as e:
                return [types.TextContent(type="text", text=f"Error: {str(e)}")]
    
    async def run(self):
        """Run the MCP server"""
        if not MCP_AVAILABLE:
            print("❌ MCP not available. Cannot run server.")
            return
        
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )

async def main():
    """Main function to run MCP server"""
    print("=" * 60)
    print("🚀 Starting RAG MCP Server...")
    print("=" * 60)
    
    server = RAGMCPServer()
    await server.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 MCP Server stopped gracefully.")