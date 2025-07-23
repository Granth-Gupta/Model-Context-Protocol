import asyncio
import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

import google.generativeai as genai
from fastmcp import Client
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import aiohttp

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    tools_used: List[str] = []
    timestamp: str


class MCPChatbot:
    def __init__(self):
        self.gemini_model = None
        self.employee_client = None
        self.leave_client = None
        self.available_tools = {}
        self.is_initialized = False

        # FastAPI app
        self.app = FastAPI(title="MCP Client API", description="Web API for MCP Client UI")
        self.setup_fastapi()

    def get_server_config(self):
        """Get MCP server configuration from environment variables or use defaults"""

        # Get from environment variables first
        employee_url = os.getenv("EMPLOYEE_SERVER_URL")
        leave_url = os.getenv("LEAVE_SERVER_URL")

        # Default configuration
        clients_config = [
            {
                "url": employee_url or "https://mcp-server-employee-hze7pjwixa-uc.a.run.app/sse"
                       or "http://0.0.0.0:8002/sse",
                "name": "employee_server",
                "client_attr": "employee_client"
            },
            {
                "url": leave_url or "https://mcp-server-leaving-hze7pjwixa-uc.a.run.app/sse"
                       or "http://0.0.0.0:8001/sse",
                "name": "leave_server",
                "client_attr": "leave_client"
            }
        ]

        # Log configuration
        logger.info("🔧 MCP Server Configuration:")
        for config in clients_config:
            logger.info(f"   - {config['name']}: {config['url']}")

        return clients_config

    def setup_fastapi(self):
        """Setup FastAPI routes and middleware"""
        # Enable CORS for React frontend - Updated for Cloud Run
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "http://localhost:5173",
                "http://localhost:3000",
                "http://127.0.0.1:5173",
                "https://*.run.app",  # Allow Cloud Run origins
                "*"  # For development - remove in production
            ],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # API Routes
        @self.app.post("/chat", response_model=ChatResponse)
        async def chat_endpoint(request: ChatRequest):
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

        @self.app.get("/status")
        async def status_endpoint():
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

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}

        @self.app.get("/tools")
        async def get_tools():
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

    async def test_server_connection(self, url: str, server_name: str) -> bool:
        """Test if MCP server is reachable"""
        try:
            # Try multiple endpoints to test connectivity
            test_urls = [
                f"{url.replace('/sse', '')}/health",
                f"{url.replace('/sse', '')}/status",
                url.replace('/sse', '')
            ]

            async with aiohttp.ClientSession() as session:
                for test_url in test_urls:
                    try:
                        async with session.get(test_url, timeout=5) as response:
                            if response.status in [200, 404]:  # 404 is also ok, means server is running
                                logger.info(f"✅ {server_name} reachable at {test_url} (status: {response.status})")
                                return True
                    except Exception as e:
                        logger.debug(f"Failed to connect to {test_url}: {e}")
                        continue

                logger.warning(f"⚠️  {server_name} not reachable at any endpoint")
                return False

        except Exception as e:
            logger.warning(f"⚠️  {server_name} connection test failed: {e}")
            return False

    def setup_gemini(self):
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

    async def setup_mcp_clients(self):
        """Initialize MCP clients with proper tool processing (based on main_1.py)"""
        try:
            # Connect to both servers
            # self.employee_client = Client("http://127.0.0.1:8002/sse")
            # self.leave_client = Client("http://127.0.0.1:8001/sse")

            # Get server configuration
            server_configs = self.get_server_config()

            # Connect to servers using environment URLs
            employee_url = os.getenv("EMPLOYEE_SERVER_URL", "https://mcp-server-employee-hze7pjwixa-uc.a.run.app/sse")
            leave_url = os.getenv("LEAVE_SERVER_URL", "https://mcp-server-leaving-hze7pjwixa-uc.a.run.app/sse")

            self.employee_client = Client(employee_url)
            self.leave_client = Client(leave_url)

            # Initialize clients
            await self.employee_client.__aenter__()
            await self.leave_client.__aenter__()

            # Get available tools from both servers
            employee_tools_response = await self.employee_client.list_tools()
            leave_tools_response = await self.leave_client.list_tools()

            # Process employee tools (using main_1.py logic)
            employee_tools = employee_tools_response.tools if hasattr(employee_tools_response,
                                                                      'tools') else employee_tools_response
            for tool in employee_tools:
                tool_name = tool.name if hasattr(tool, 'name') else tool.get('name')
                tool_desc = tool.description if hasattr(tool, 'description') else tool.get('description', '')
                tool_params = {}

                if hasattr(tool, 'inputSchema'):
                    tool_params = tool.inputSchema.get('properties', {})
                elif isinstance(tool, dict) and 'inputSchema' in tool:
                    tool_params = tool['inputSchema'].get('properties', {})

                self.available_tools[tool_name] = {
                    'client': self.employee_client,
                    'server': 'employee_server',
                    'description': tool_desc,
                    'parameters': tool_params
                }

            # Process leave tools (using main_1.py logic)
            leave_tools = leave_tools_response.tools if hasattr(leave_tools_response, 'tools') else leave_tools_response
            for tool in leave_tools:
                tool_name = tool.name if hasattr(tool, 'name') else tool.get('name')
                tool_desc = tool.description if hasattr(tool, 'description') else tool.get('description', '')
                tool_params = {}

                if hasattr(tool, 'inputSchema'):
                    tool_params = tool.inputSchema.get('properties', {})
                elif isinstance(tool, dict) and 'inputSchema' in tool:
                    tool_params = tool['inputSchema'].get('properties', {})

                self.available_tools[tool_name] = {
                    'client': self.leave_client,
                    'server': 'leave_server',
                    'description': tool_desc,
                    'parameters': tool_params
                }

            logger.info(f"✅ MCP clients initialized successfully")
            logger.info(f"📊 Available tools: {list(self.available_tools.keys())}")
            logger.info(f"🔧 Total tools loaded: {len(self.available_tools)}")

        except Exception as e:
            logger.error(f"❌ Error initializing MCP clients: {e}")
            logger.error(f"Error details: {type(e).__name__}")

    async def call_mcp_tool(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """Call an MCP tool and return the result (based on main_1.py)"""
        if tool_name not in self.available_tools:
            return f"❌ Tool '{tool_name}' not found. Available tools: {list(self.available_tools.keys())}"

        try:
            client = self.available_tools[tool_name]['client']
            result = await client.call_tool(tool_name, parameters)
            return str(result.content[0].text if result.content else "No result returned")
        except Exception as e:
            return f"❌ Error calling tool '{tool_name}': {e}"

    def parse_gemini_response(self, response: str) -> tuple[bool, Optional[str], Optional[Dict]]:
        """Parse Gemini response to check if it contains a tool call (from main_1.py)"""
        try:
            # Check if response contains JSON tool call
            if "\"action\":" in response and "\"tool_name\":" in response:
                # Extract JSON from response
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
        """Create system prompt with available tools information (from main_1.py)"""
        tools_info = []
        for tool_name, tool_data in self.available_tools.items():
            params = ", ".join([f"{k}: {v.get('type', 'string')}" for k, v in tool_data['parameters'].items()])
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

    async def chat(self, user_message: str) -> tuple[str, List[str]]:
        """Process user message and return response (based on main_1.py)"""
        tools_used = []

        try:
            if not self.available_tools:
                return "❌ No MCP tools available. Please check server connections.", tools_used

            # Create full prompt with system instructions and user message
            full_prompt = f"{self.create_system_prompt()}\n\nUser: {user_message}\nAssistant:"

            # Get response from Gemini
            if self.gemini_model:
                response = self.gemini_model.generate_content(full_prompt)
                gemini_response = response.text

                # Check if Gemini wants to use a tool
                is_tool_call, tool_name, parameters = self.parse_gemini_response(gemini_response)

                if is_tool_call and tool_name:
                    # Execute the tool
                    tool_result = await self.call_mcp_tool(tool_name, parameters)
                    tools_used.append(tool_name)

                    # Get final response from Gemini with tool result
                    final_prompt = f"{full_prompt}\n{gemini_response}\n\nTool result: {tool_result}\n\nPlease provide a natural response based on this information:"
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

    async def initialize(self):
        """Initialize all components"""
        logger.info("🚀 Initializing MCP Chatbot with fixed tool processing...")

        # Initialize Gemini
        gemini_ready = self.setup_gemini()

        # Initialize MCP clients with proper tool processing
        await self.setup_mcp_clients()

        self.is_initialized = True

        logger.info(f"✅ MCP Chatbot Web Server Ready!")
        logger.info(f"📊 Gemini API: {'✅ Ready' if gemini_ready else '❌ Unavailable'}")
        logger.info(f"🔧 Available Tools: {len(self.available_tools)}")
        logger.info(f"🌐 API Endpoints: http://localhost:8000/docs")

        return self.is_initialized


# Global chatbot instance
chatbot = MCPChatbot()


async def start_web_server():
    """Start the web server for React UI"""
    # Initialize the chatbot
    success = await chatbot.initialize()
    if not success:
        logger.error("❌ Failed to initialize chatbot completely")
        return

    # Get port from environment variable (Cloud Run requirement)
    port = int(os.environ.get('PORT', 8000))

    # Start web server
    logger.info("🌐 Starting MCP Client Web Server...")
    logger.info(f"🔗 React UI should connect to: http://0.0.0.0:{port}")
    logger.info(f"📋 API Documentation: http://0.0.0.0:{port}/docs")
    logger.info(f"🏥 Health Check: http://0.0.0.0:{port}/health")

    config = uvicorn.Config(
        app=chatbot.app,
        host="0.0.0.0",  # Changed from localhost to 0.0.0.0
        port=port,  # Use dynamic port from environment
        log_level="info",
        reload=False
    )

    server = uvicorn.Server(config)
    try:
        await server.serve()
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    finally:
        # Cleanup code remains the same
        try:
            if chatbot.employee_client:
                await chatbot.employee_client.__aexit__(None, None, None)
            if chatbot.leave_client:
                await chatbot.leave_client.__aexit__(None, None, None)
            logger.info("✅ Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(start_web_server())
    except KeyboardInterrupt:
        print("\n👋 Server stopped!")
