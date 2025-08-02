"""
MCP Client Web API - FastAPI-based service to interact with MCP servers and Gemini LLM.
Adapted for idiomatic FastAPI usage and proper async resource lifecycle.
"""

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

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_client_api")

# =====================
# GLOBAL STATE
# =====================
app = FastAPI(
    title="MCP Client API",
    description="Web API for MCP Client UI"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-frontend-domain.com",  # TODO: Replace with actual frontend domain(s)
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

employee_client: Optional[Client] = None
leave_client: Optional[Client] = None
available_tools: Dict[str, Dict[str, Any]] = {}
gemini_model: Optional[Any] = None
startup_time = datetime.utcnow()
is_initialized = False

# =====================
# Pydantic Schemas
# =====================
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    tools_used: List[str] = []
    timestamp: str

# =====================
# Utility Functions
# =====================
def get_server_config() -> List[Dict[str, Any]]:
    employee_url = os.getenv("EMPLOYEE_SERVER_URL", "https://mcp-server-employee-273927490120.us-central1.run.app/sse")
    leave_url = os.getenv("LEAVE_SERVER_URL", "https://mcp-server-leaving-273927490120.us-central1.run.app/sse")
    return [
        {"url": employee_url, "name": "employee_server", "attr": "employee_client"},
        {"url": leave_url, "name": "leave_server", "attr": "leave_client"},
    ]

async def test_mcp_connection(client: Client, server_name: str) -> bool:
    try:
        if not client:
            logger.warning(f"No MCP client for {server_name}")
            return False
        result = await client.list_tools()
        return bool(result)
    except Exception as e:
        logger.error(f"Connection test failed for {server_name}: {e}")
        return False

def _extract_tool_info(tool: Any) -> Tuple[str, str, Dict[str, Any]]:
    # Extract tool info for prompt injection
    name = getattr(tool, "name", tool.get("name", ""))
    desc = getattr(tool, "description", tool.get("description", ""))
    params = {}
    schema = getattr(tool, "inputSchema", None) or tool.get("inputSchema", None)
    if schema:
        params = schema.get("properties", {})
    return name, desc, params

async def process_tools_from_client(client: Client, server_name: str) -> None:
    global available_tools
    try:
        tools = await client.list_tools()
        tools_list = getattr(tools, "tools", tools)
        for tool in tools_list:
            name, description, parameters = _extract_tool_info(tool)
            available_tools[name] = {
                "client": client,
                "server": server_name,
                "description": description,
                "parameters": parameters,
            }
    except Exception as e:
        logger.error(f"Failed to load tools from {server_name}: {e}")

async def get_tools_for_prompt() -> List[Dict[str, Any]]:
    return [
        {"name": name, "description": info.get("description", "")}
        for name, info in available_tools.items()
    ]

def setup_gemini() -> bool:
    global gemini_model
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("Missing GEMINI_API_KEY")
        return False
    try:
        genai.configure(api_key=api_key)
        gemini_model = genai.TextGenerationModel.from_pretrained("models/text-bison-001")
        logger.info("Gemini model initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Gemini model: {e}")
        return False

async def setup_mcp_clients() -> bool:
    global employee_client, leave_client, available_tools
    configs = get_server_config()
    all_ok = True
    for config in configs:
        try:
            client = Client(config["url"])
            await client.__aenter__()
            connected = await test_mcp_connection(client, config["name"])
            if not connected:
                logger.warning(f"Failed to connect to {config['name']}")
                all_ok = False
            else:
                if config["attr"] == "employee_client":
                    employee_client = client
                else:
                    leave_client = client
                await process_tools_from_client(client, config["name"])
                logger.info(f"Connected to {config['name']}")
        except Exception as e:
            logger.error(f"Error connecting to {config['name']}: {e}")
            all_ok = False
    return all_ok

async def call_tool(tool_name: str, params: dict) -> str:
    if tool_name not in available_tools:
        return f"Tool '{tool_name}' not found."
    client = available_tools[tool_name]["client"]
    try:
        result = await client.call_tool(tool_name, params)
        if hasattr(result, "content"):
            return result.content if isinstance(result.content, str) else str(result.content)
        return str(result)
    except Exception as e:
        logger.error(f"Error calling tool {tool_name}: {e}")
        return f"Error calling tool {tool_name}: {e}"

def parse_gemini_response(response_text: str) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    try:
        if "\"action\":" in response_text and "\"tool_name\":" in response_text:
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            json_str = response_text[start:end]
            parsed = json.loads(json_str)
            if parsed.get("action") == "use_tool":
                return True, parsed.get("tool_name"), parsed.get("parameters", {})
        return False, None, None
    except Exception:
        return False, None, None

async def cleanup():
    global employee_client, leave_client, available_tools, gemini_model
    available_tools = {}
    if employee_client is not None:
        try:
            await employee_client.__aexit__(None, None, None)
        except Exception:
            pass
    employee_client = None
    if leave_client is not None:
        try:
            await leave_client.__aexit__(None, None, None)
        except Exception:
            pass
    leave_client = None
    gemini_model = None

# =====================
# FastAPI Lifecycle and Routes
# =====================

@app.on_event("startup")
async def on_startup():
    global is_initialized
    logger.info("Initializing Gemini and MCP clients...")
    gemini_ok = setup_gemini()
    mcp_ok = await setup_mcp_clients()
    is_initialized = gemini_ok and mcp_ok
    if is_initialized:
        logger.info("Startup: All systems initialized.")
    else:
        logger.error("Startup: Initialization failed; check logs.")

@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Cleaning up resources...")
    await cleanup()

@app.get("/", tags=["Root"])
async def root():
    uptime_seconds = (datetime.utcnow() - startup_time).total_seconds()
    return {
        "message": "MCP Client API is running",
        "uptime_seconds": uptime_seconds,
        "health_status": "healthy" if is_initialized else "initializing"
    }

@app.get("/health", tags=["Health"])
async def health():
    employee_ok = await test_mcp_connection(employee_client, "employee_server")
    leave_ok = await test_mcp_connection(leave_client, "leave_server")
    overall_status = "healthy" if employee_ok and leave_ok else "degraded"
    return {"status": overall_status}

@app.get("/tools", tags=["Tools"])
async def tools():
    if not is_initialized:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Agent not initialized")
    tools_info = []
    for name, info in available_tools.items():
        tools_info.append({
            "name": name,
            "description": info.get("description", ""),
            "server": info.get("server", ""),
        })
    return {"tools": tools_info, "count": len(tools_info)}

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(request: ChatRequest):
    if not is_initialized or gemini_model is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Agent not initialized")
    tool_descriptions = "\n".join(
        f"- {t['name']} : {t['description']}"
        for t in await get_tools_for_prompt()
    )
    user_input = request.message.strip()
    prompt = f"You are an intelligent assistant with access to the following tools:\n{tool_descriptions}\nUser: {user_input}\nAssistant:"

    tools_used: List[str] = []
    try:
        # Call Gemini model (synchronously here)
        response = gemini_model.generate_text(prompt)
        text_response = response.text if hasattr(response, "text") else str(response)

        # Check if response includes a tool call
        is_tool_call, tool_name, params = parse_gemini_response(text_response)
        if is_tool_call and tool_name and tool_name in available_tools:
            tool_result = await call_tool(tool_name, params)
            tools_used.append(tool_name)
            # Second LLM call for summarization
            final_prompt = (f"{prompt}\nTool: {json.dumps({'tool_name': tool_name, 'result': tool_result})}\nPlease provide a summary.")
            final_response = gemini_model.generate_text(final_prompt)
            final_text = final_response.text if hasattr(final_response, "text") else str(final_response)
        else:
            final_text = text_response

        timestamp = datetime.utcnow().isoformat()
        return ChatResponse(response=final_text, tools_used=tools_used, timestamp=timestamp)
    except Exception as e:
        logger.error(f"Chat processing error: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

# ===============
# MAIN LAUNCHER (for local run)
# ===============
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")

