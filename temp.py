"""
MCP Client Web API - FastAPI-based service to interact with MCP servers and Gemini LLM.
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
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastmcp import Client

# Load environment variables set via .env
load_dotenv()

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_client_api")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    tools_used: List[str] = []
    timestamp: str


class MCPClientAPI:
    def __init__(self):
        self.gemini_model: Optional[Any] = None
        self.employee_client: Optional[Client] = None
        self.leave_client: Optional[Client] = None
        self.available_tools: Dict[str, Dict[str, Any]] = {}
        self.is_initialized: bool = False
        self.server_port: int = int(os.getenv("PORT", 8000))
        self.start_time = datetime.utcnow()
        self.app = FastAPI(
            title="MCP Client API",
            description="API for MCP client UI"
        )
        self._setup_cors()
        self._setup_routes()

    def _setup_cors(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "https://your-frontend-domain.com",  # Replace with your real frontend URL(s)
                "http://localhost:3000",
                "http://127.0.0.1:3000",
            ],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def get_server_config(self) -> List[Dict[str, Any]]:
        employee_url = os.getenv(
            "EMPLOYEE_SERVER_URL",
            "https://mcp-server-employee-273927490120.us-central1.run.app/sse"
        )
        leave_url = os.getenv(
            "LEAVE_SERVER_URL",
            "https://mcp-server-leaving-273927490120.us-central1.run.app/sse"
        )

        configs = [
            {"url": employee_url, "name": "employee_server", "attr": "employee_client"},
            {"url": leave_url, "name": "leave_server", "attr": "leave_client"},
        ]
        logger.info("MCP server configs:")
        for c in configs:
            logger.info(f" - {c['name']}: {c['url']}")
        return configs

    async def test_mcp_connection(self, client: Client, server_name: str) -> bool:
        if client is None:
            logger.warning(f"No client for {server_name}")
            return False
        try:
            tools = await client.list_tools()
            return bool(tools)
        except Exception as e:
            logger.error(f"Failed to connect to {server_name}: {e}")
            return False

    def _extract_tool_info(self, tool: Any) -> Tuple[str, str, Dict[str, Any]]:
        name = getattr(tool, "name", "") or tool.get("name", "")
        description = getattr(tool, "description", "") or tool.get("description", "")
        parameters = {}
        schema = getattr(tool, "inputSchema", None) or tool.get("inputSchema", None)
        if schema:
            parameters = schema.get("properties", {})
        return name, description, parameters

    async def _process_tools_from_client(self, client: Client, server_name: str) -> None:
        try:
            tools_resp = await client.list_tools()
            tools_list = getattr(tools_resp, "tools", tools_resp)
            for tool in tools_list:
                name, description, params = self._extract_tool_info(tool)
                self.available_tools[name] = {
                    "client": client,
                    "server": server_name,
                    "description": description,
                    "parameters": params,
                }
        except Exception as e:
            logger.error(f"Error loading tools from {server_name}: {e}")

    async def setup_clients_and_tools(self) -> bool:
        configs = self.get_server_config()
        success = False
        for conf in configs:
            try:
                client = Client(conf["url"])
                setattr(self, conf["attr"], client)
                await client.__aenter__()
                connected = await self.test_mcp_connection(client, conf["name"])
                if connected:
                    logger.info(f"Connected to {conf['name']}")
                    await self._process_tools_from_client(client, conf["name"])
                    success = True
                else:
                    logger.warning(f"Could not connect to {conf['name']}")
            except Exception as e:
                logger.error(f"Error connecting to {conf['name']}: {e}")
        return success

    def setup_gemini(self) -> bool:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not set")
            return False
        try:
            genai.configure(api_key=api_key)
            self.gemini_model = genai.TextGenerationModel.from_pretrained("models/text-bison-001")
            logger.info("Gemini model initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            return False

    async def call_tool(self, tool_name: str, params: dict) -> str:
        tool_info = self.available_tools.get(tool_name)
        if not tool_info:
            return f"Tool '{tool_name}' not found."

        try:
            client = tool_info["client"]
            result = await client.call_tool(tool_name, params)
            if hasattr(result, "content"):
                if isinstance(result.content, str):
                    return result.content
                return str(result.content)
            return str(result)
        except Exception as e:
            logger.error(f"Error calling tool '{tool_name}': {e}")
            return f"Error calling tool '{tool_name}': {e}"

    @staticmethod
    def parse_gemini_response(text: str) -> Tuple[bool, Optional[str], Optional[dict]]:
        # Check if text contains a JSON snippet indicating a tool call
        try:
            if '"action":' in text and '"tool_name":' in text:
                start = text.find('{')
                end = text.rfind('}') + 1
                snippet = text[start:end]
                data = json.loads(snippet)
                if data.get("action") == "use_tool":
                    return True, data.get("tool_name"), data.get("parameters")
            return False, None, None
        except Exception:
            return False, None, None

    async def chat(self, user_message: str) -> Tuple[str, List[str]]:
        tools_used: List[str] = []

        if not self.is_initialized:
            return "Service not initialized yet. Please try again later.", tools_used

        tool_descriptions = "\n".join(
            f"- {name}({', '.join(params.keys())}) : {info['description']}"
            for name, info in self.available_tools.items()
        )
        prompt = f"You are a helpful assistant that uses these tools:\n{tool_descriptions}\nUser: {user_message}\nAssistant:"

        try:
            response = self.gemini_model.generate(prompt)
            response_text = response.text if hasattr(response, "text") else str(response)

            is_tool_call, tool_name, params = self.parse_gemini_response(response_text)
            if is_tool_call and tool_name and tool_name in self.available_tools:
                tools_used.append(tool_name)
                tool_result = await self.call_tool(tool_name, params or {})
                # Compose final prompt for summary
                summary_prompt = (
                    f"{prompt}\nTool Response: {tool_result}\nPlease give a summary for the user."
                )
                final_response = self.gemini_model.generate(summary_prompt)
                final_text = final_response.text if hasattr(final_response, "text") else str(final_response)
                return final_text, tools_used
            else:
                return response_text, tools_used

        except Exception as e:
            logger.error(f"Chat processing error: {e}")
            return f"Error during chat processing: {e}", tools_used

    def setup_routes(self):
        @self.app.get("/", tags=["Root"])
        async def root():
            uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds()
            status_str = "healthy" if self.is_initialized else "initializing"
            return {"message": "MCP Client API running", "uptime_seconds": uptime_seconds, "status": status_str}

        @self.app.get("/health", tags=["Health"])
        async def health():
            employee_ok = await self.test_mcp_connection(self.employee_client, "employee_server")
            leave_ok = await self.test_mcp_connection(self.leave_client, "leave_server")
            overall = "healthy" if employee_ok and leave_ok else "degraded"
            code = status.HTTP_200_OK if overall == "healthy" else status.HTTP_503_SERVICE_UNAVAILABLE
            return {"status": overall}, code

        @self.app.get("/tools", tags=["Tools"])
        async def tools():
            if not self.is_initialized:
                raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Service not ready")
            tool_list = []
            for name, info in self.available_tools.items():
                tool_list.append({"name": name, "description": info.get("description", ""), "server": info.get("server", "")})
            return {"tools": tool_list, "count": len(tool_list)}

        @self.app.post("/chat", response_model=ChatResponse, tags=["Chat"])
        async def chat_endpoint(request: ChatRequest):
            if not self.is_initialized:
                raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Service not ready")
            reply, used_tools = await self.chat(request.message)
            return ChatResponse(response=reply, tools_used=used_tools, timestamp=datetime.utcnow().isoformat())

    async def start(self) -> None:
        gemini_ready = self.setup_gemini()
        if not gemini_ready:
            logger.warning("Gemini model not ready.")
        mcp_ready = await self.setup_clients_and_tools()
        self.is_initialized = gemini_ready and mcp_ready
        return self.is_initialized


    async def shutdown(self) -> None:
        # Close MCP client connections gracefully
        for client in (self.employee_client, self.leave_client):
            if client:
                try:
                    await client.__aexit__(None, None, None)
                except Exception as e:
                    logger.error(f"Error closing client: {e}")

    # Placeholder, as routes are set on self.app
    def run(self) -> None:
        pass


mcp_api = MCPClientAPI()


async def main():
    ok = await mcp_api.start()
    if not ok:
        logger.error("Failed to initialize MCP client fully.")
    config = uvicorn.Config(mcp_api.app, host="0.0.0.0", port=mcp_api.server_port, log_level="info")
    server = uvicorn.Server(config)
    try:
        await server.serve()
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        await mcp_api.shutdown()


if __name__ == "__main__":
    import sys
    import asyncio
    import uvicorn

    if sys.version_info >= (3, 7):
        # Run async main method via asyncio.run
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("\n👋 Server stopped!")
    else:
        # For Python versions < 3.7, suggest to use uvicorn CLI to run
        print("Use `uvicorn main:app` to run the server.")

    # Alternatively, for container and production usage,
    # you can omit this and instead run via:
    # uvicorn main:app --host 0.0.0.0 --port <PORT>

