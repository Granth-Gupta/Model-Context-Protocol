"""
MCP Client Web API
A FastAPI-based web service for interacting with Model Context Protocol servers
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import google.generativeai as genai
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastmcp import Client
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str
    tools_used: List[str] = []
    timestamp: str


class MCPChatbot:
    """Main MCP Chatbot class handling FastAPI server and MCP client connections"""

    def __init__(self) -> None:
        self.gemini_model: Any = None
        self.employee_client: Optional[Client] = None
        self.leave_client: Optional[Client] = None
        self.available_tools: Dict[str, Dict[str, Any]] = {}
        self.is_initialized: bool = False
        self.server_port: Optional[int] = None

        # FastAPI app
        self.app = FastAPI(
            title="MCP Client API",
            description="Web API for MCP Client UI"
        )
        self.setup_fastapi()

    def get_server_config(self) -> List[Dict[str, str]]:
        """Get MCP server configuration from environment variables or use defaults"""
        employee_url = os.getenv("EMPLOYEE_SERVER_URL")
        leave_url = os.getenv("LEAVE_SERVER_URL")

        clients_config = [
            {
                "url": employee_url or "https://mcp-server-employee-273927490120.us-central1.run.app/sse",
                "name": "employee_server",
                "client_attr": "employee_client"
            },
            {
                "url": leave_url or "https://mcp-server-leaving-273927490120.us-central1.run.app/sse",
                "name": "leave_server",
                "client_attr": "leave_client"
            }
        ]

        logger.info("🔧 MCP Server Configuration:")
        for config in clients_config:
            logger.info(f"   - {config['name']}: {config['url']}")

        return clients_config

    async def test_mcp_connection(self, client: Optional[Client], server_name: str) -> bool:
        """Test if MCP server is actually reachable and responding"""
        try:
            if not client:
                logger.warning(f"{server_name}: Client not initialized")
                return False

            # Try to list available tools as a connectivity test
            result = await client.list_tools()
            if result:
                logger.info(f"{server_name}: Connection test successful")
                return True
            else:
                logger.warning(f"{server_name}: Connection test returned empty result")
                return False

        except Exception as e:
            logger.error(f"{server_name}: Connection test failed - {e}")
            return False

    def _extract_tool_info(self, tool: Any) -> Tuple[str, str, Dict[str, Any]]:
        """Extract tool information from tool object or dict"""
        tool_name = tool.name if hasattr(tool, 'name') else tool.get('name', '')
        tool_desc = tool.description if hasattr(tool, 'description') else tool.get('description', '')
        tool_params: Dict[str, Any] = {}

        if hasattr(tool, 'inputSchema'):
            tool_params = tool.inputSchema.get('properties', {})
        elif isinstance(tool, dict) and 'inputSchema' in tool:
            tool_params = tool['inputSchema'].get('properties', {})

        return tool_name, tool_desc, tool_params

    async def _process_server_tools(self, client: Client, server_name: str) -> None:
        """Process tools from a specific server"""
        try:
            tools_response = await client.list_tools()
            tools = tools_response.tools if hasattr(tools_response, 'tools') else tools_response

            for tool in tools:
                tool_name, tool_desc, tool_params = self._extract_tool_info(tool)

                self.available_tools[tool_name] = {
                    'client': client,
                    'server': server_name,
                    'description': tool_desc,
                    'parameters': tool_params
                }
        except Exception as e:
            logger.error(f"Failed to get {server_name} tools: {e}")

    def setup_fastapi(self) -> None:
        """Setup FastAPI routes and middleware"""

        # Health check endpoint
        @self.app.get("/health")
        async def health_check() -> Dict[str, Any]:
            try:
                # Test actual connectivity to MCP servers
                employee_status = await self.test_mcp_connection(self.employee_client, "employee_server")
                leave_status = await self.test_mcp_connection(self.leave_client, "leave_server")

                # Determine overall health
                overall_status = "healthy" if (employee_status and leave_status) else "degraded"

                return {
                    "status": overall_status,
                    "service": "mcp-client",
                    "timestamp": datetime.now().isoformat(),
                    "port": self.server_port or int(os.environ.get('PORT', 8000)),
                    "employees_connected": employee_status,
                    "leave_connected": leave_status,
                    "server_details": {
                        "employee_server": {
                            "status": "connected" if employee_status else "disconnected",
                            "url": os.getenv("EMPLOYEE_SERVER_URL", "not_configured")
                        },
                        "leave_server": {
                            "status": "connected" if leave_status else "disconnected",
                            "url": os.getenv("LEAVE_SERVER_URL", "not_configured")
                        }
                    }
                }
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                return {
                    "status": "unhealthy",
                    "service": "mcp-client",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "employees_connected": False,
                    "leave_connected": False
                }

        # Enable CORS for React frontend
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "http://localhost:5173",
                "http://localhost:3000",
                "http://127.0.0.1:5173",
                "https://*.run.app",
                "https://mcp-ui-273927490120.us-central1.run.app",
                "*"  # For development - remove in production
            ],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Root endpoint
        @self.app.get("/")
        async def root() -> Dict[str, str]:
            return {"message": "MCP Client API is running", "service": "mcp-client"}

        # Chat endpoint
        @self.app.post("/chat", response_model=ChatResponse)
        async def chat_endpoint(request: ChatRequest) -> ChatResponse:
            try:
                if not self.is_initialized:
                    raise HTTPException(status_code=503, detail="MCP client not initialized")

                response, tools_used = await self.chat(request.message)
                return ChatResponse(
                    response=response,
                    tools_used=tools_used,
                    timestamp=datetime.now().isoformat()
                )
            except Exception as e:
                logger.error(f"Chat endpoint error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Status endpoint
        @self.app.get("/status")
        async def status_endpoint() -> Dict[str, Any]:
            """Get system status and available tools"""
            server_configs = self.get_server_config()
            server_status = {}

            for config in server_configs:
                client = getattr(self, config["client_attr"], None)
                server_status[config["name"]] = {
                    "connected": client is not None,
                    "url": config["url"]
                }

            return {
                "initialized": self.is_initialized,
                "available_tools": [
                    {
                        "name": name,
                        "description": info["description"],
                        "server": info["server"]
                    }
                    for name, info in self.available_tools.items()
                ],
                "servers": server_status,
                "gemini_ready": self.gemini_model is not None,
                "total_tools": len(self.available_tools)
            }

        # Tools endpoint
        @self.app.get("/tools")
        async def get_tools() -> Dict[str, Any]:
            """Get available tools list"""
            return {
                "tools": list(self.available_tools.keys()),
                "count": len(self.available_tools),
                "details": [
                    {
                        "name": name,
                        "server": info["server"],
                        "description": info["description"]
                    }
                    for name, info in self.available_tools.items()
                ]
            }

    def setup_gemini(self) -> bool:
        """Initialize Gemini AI model"""
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.warning("GEMINI_API_KEY not found in environment variables")
                return False

            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')

            # Test the connection
            test_response = self.gemini_model.generate_content("Hello")
            if test_response and test_response.text:
                logger.info("✅ Gemini API initialized successfully")
                return True
            else:
                raise ValueError("Gemini API test failed")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini: {e}")
            return False

    async def setup_mcp_clients(self) -> bool:
        """Initialize MCP clients with proper tool processing"""
        try:
            # Get server URLs from environment variables
            employee_url = os.getenv(
                "EMPLOYEE_SERVER_URL",
                "https://mcp-server-employee-273927490120.us-central1.run.app/sse"
            )
            leave_url = os.getenv(
                "LEAVE_SERVER_URL",
                "https://mcp-server-leaving-273927490120.us-central1.run.app/sse"
            )

            logger.info(f"Connecting to Employee Server: {employee_url}")
            logger.info(f"Connecting to Leave Server: {leave_url}")

            # Initialize MCP clients
            self.employee_client = Client(employee_url)
            self.leave_client = Client(leave_url)

            # Enter async context
            await self.employee_client.__aenter__()
            await self.leave_client.__aenter__()

            # Test connections
            employee_connected = await self.test_mcp_connection(self.employee_client, "employee_server")
            leave_connected = await self.test_mcp_connection(self.leave_client, "leave_server")

            logger.info(f"Employee server connection: {'✅' if employee_connected else '❌'}")
            logger.info(f"Leave server connection: {'✅' if leave_connected else '❌'}")

            # Load tools from connected servers
            if employee_connected:
                await self._process_server_tools(self.employee_client, 'employee_server')

            if leave_connected:
                await self._process_server_tools(self.leave_client, 'leave_server')

            logger.info("✅ MCP clients setup completed")
            logger.info(f"📊 Available tools: {list(self.available_tools.keys())}")
            logger.info(f"🔧 Total tools loaded: {len(self.available_tools)}")

            return employee_connected or leave_connected

        except Exception as e:
            logger.error(f"❌ Error initializing MCP clients: {e}")
            return False

    async def call_mcp_tool(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """Call an MCP tool and return the result"""
        if tool_name not in self.available_tools:
            return f"❌ Tool '{tool_name}' not found. Available tools: {list(self.available_tools.keys())}"

        try:
            client = self.available_tools[tool_name]['client']
            result = await client.call_tool(tool_name, parameters)
            return str(result.content[0].text if result.content else "No result returned")
        except Exception as e:
            return f"❌ Error calling tool '{tool_name}': {e}"

    @staticmethod
    def parse_gemini_response(response: str) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """Parse Gemini response to check if it contains a tool call"""
        try:
            if "\"action\":" in response and "\"tool_name\":" in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                if start != -1 and end != 0:
                    json_str = response[start:end]
                    tool_call = json.loads(json_str)

                    if tool_call.get("action") == "use_tool":
                        return True, tool_call.get("tool_name"), tool_call.get("parameters", {})

            return False, None, None
        except json.JSONDecodeError:
            return False, None, None

    def create_system_prompt(self) -> str:
        """Create system prompt with available tools information"""
        tools_info = []
        for tool_name, tool_data in self.available_tools.items():
            params = ", ".join([
                f"{k}: {v.get('type', 'string')}"
                for k, v in tool_data['parameters'].items()
            ])
            tools_info.append(f"- {tool_name}({params}): {tool_data['description']}")

        return f"""You are a helpful chatbot assistant that can help with employee directory and leave management tasks.

Available tools:
{chr(10).join(tools_info)}

When a user asks for something that requires these tools, respond with a tool call in this JSON format:
{{
    "action": "use_tool",
    "tool_name": "tool_name_here",
    "parameters": {{
        "param1": "value1",
        "param2": "value2"
    }}
}}

If the user asks something that doesn't require tools, respond normally with helpful information.

Examples of when to use tools:
- "Show me employee E001's info" → use get_employee_info
- "Find employees in Engineering department" → use search_employees
- "Get team members for manager M001" → use get_team_members
- "How many leave days does E002 have?" → use get_leave_balance  
- "Apply leave for E001 on 2025-08-15" → use apply_leave
"""

    async def chat(self, user_message: str) -> Tuple[str, List[str]]:
        """Process user message and return response"""
        tools_used: List[str] = []

        try:
            if not self.available_tools:
                return "❌ No MCP tools available. Please check server connections.", tools_used

            full_prompt = f"{self.create_system_prompt()}\n\nUser: {user_message}\nAssistant:"

            if self.gemini_model:
                response = self.gemini_model.generate_content(full_prompt)
                gemini_response = response.text

                is_tool_call, tool_name, parameters = self.parse_gemini_response(gemini_response)

                if is_tool_call and tool_name:
                    tool_result = await self.call_mcp_tool(tool_name, parameters)
                    tools_used.append(tool_name)

                    final_prompt = (
                        f"{full_prompt}\n{gemini_response}\n\nTool result: {tool_result}\n\n"
                        "Please provide a natural response based on this information:"
                    )
                    final_response = self.gemini_model.generate_content(final_prompt)
                    return final_response.text, tools_used
                else:
                    return gemini_response, tools_used
            else:
                return f"I'll help you with: {user_message}. Let me use the available tools.", tools_used

        except Exception as e:
            error_msg = f"❌ Error processing message: {e}"
            logger.error(error_msg)
            return error_msg, tools_used

    async def initialize(self) -> bool:
        """Initialize all components"""
        logger.info("🚀 Initializing MCP Chatbot...")

        # Initialize Gemini
        gemini_ready = self.setup_gemini()

        # Initialize MCP clients
        mcp_success = await self.setup_mcp_clients()

        self.is_initialized = mcp_success

        logger.info("✅ MCP Chatbot Web Server Ready!")
        logger.info(f"📊 Gemini API: {'✅ Ready' if gemini_ready else '❌ Unavailable'}")
        logger.info(f"🔧 Available Tools: {len(self.available_tools)}")

        return self.is_initialized

    async def cleanup(self) -> None:
        """Cleanup MCP client connections"""
        try:
            if self.employee_client:
                await self.employee_client.__aexit__(None, None, None)
            if self.leave_client:
                await self.leave_client.__aexit__(None, None, None)
            logger.info("✅ Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Global chatbot instance
chatbot = MCPChatbot()


async def start_web_server() -> None:
    """Start the web server for React UI"""
    # Initialize the chatbot first
    success = await chatbot.initialize()
    if not success:
        logger.error("❌ Failed to initialize chatbot completely")
        return

    # Get port from environment variable (Cloud Run requirement)
    port = int(os.environ.get('PORT', 8000))
    chatbot.server_port = port

    logger.info("🌐 Starting MCP Client Web Server...")
    logger.info(f"🔗 Server starting on: http://0.0.0.0:{port}")
    logger.info(f"🏥 Health Check: http://0.0.0.0:{port}/health")
    logger.info(f"📋 API Documentation: http://0.0.0.0:{port}/docs")

    # Debug: List registered routes
    logger.info("📋 Registered FastAPI routes:")
    for route in chatbot.app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            logger.info(f"  {route.methods} {route.path}")

    config = uvicorn.Config(
        app=chatbot.app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False
    )

    server = uvicorn.Server(config)
    try:
        await server.serve()
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        raise
    finally:
        await chatbot.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(start_web_server())
    except KeyboardInterrupt:
        print("\n👋 Server stopped!")
