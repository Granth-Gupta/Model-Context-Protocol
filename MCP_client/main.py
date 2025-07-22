import asyncio
import os
import json
from typing import Dict, Any, List, Optional
import google.generativeai as genai
from fastmcp import Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ChatbotClient:
    def __init__(self, gemini_api_key: str):
        """Initialize the chatbot client with Gemini and MCP connections"""
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')

        # MCP clients - will be initialized in async context
        self.employee_client = None
        self.leave_client = None

        # Available tools
        self.available_tools = {}

    async def initialize_mcp_clients(self):
        """Initialize MCP client connections"""
        try:
            # Connect to both Employee Directory and Leave Manager MCP Servers
            self.employee_client = Client("http://127.0.0.1:8002/sse")
            self.leave_client = Client("http://127.0.0.1:8001/sse")

            # Initialize clients
            await self.employee_client.__aenter__()
            await self.leave_client.__aenter__()

            # Get available tools from both servers
            employee_tools_response = await self.employee_client.list_tools()
            leave_tools_response = await self.leave_client.list_tools()

            # Process employee tools
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
                    'description': tool_desc,
                    'parameters': tool_params
                }

            # Process leave tools
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
                    'description': tool_desc,
                    'parameters': tool_params
                }

            print("✅ MCP clients initialized successfully")
            print(f"Available tools: {list(self.available_tools.keys())}")

        except Exception as e:
            print(f"❌ Error initializing MCP clients: {e}")
            print(f"Error details: {type(e).__name__}")

    def create_system_prompt(self) -> str:
        """Create system prompt with available tools information"""
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

    def parse_gemini_response(self, response: str) -> tuple[bool, Optional[str], Optional[Dict]]:
        """Parse Gemini response to check if it contains a tool call"""
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

    async def chat(self, user_message: str) -> str:
        """Process user message and return response"""
        try:
            # Create full prompt with system instructions and user message
            full_prompt = f"{self.create_system_prompt()}\n\nUser: {user_message}\nAssistant:"

            # Get response from Gemini
            response = self.gemini_model.generate_content(full_prompt)
            gemini_response = response.text

            # Check if Gemini wants to use a tool
            is_tool_call, tool_name, parameters = self.parse_gemini_response(gemini_response)

            if is_tool_call and tool_name:
                # Execute the tool
                tool_result = await self.call_mcp_tool(tool_name, parameters)

                # Get final response from Gemini with tool result
                final_prompt = f"{full_prompt}\n{gemini_response}\n\nTool result: {tool_result}\n\nPlease provide a natural response based on this information:"
                final_response = self.gemini_model.generate_content(final_prompt)
                return final_response.text
            else:
                return gemini_response

        except Exception as e:
            return f"❌ Error processing message: {e}"

    async def cleanup(self):
        """Clean up MCP client connections"""
        try:
            if hasattr(self, 'employee_client') and self.employee_client:
                await self.employee_client.__aexit__(None, None, None)
                print("✅ Employee client disconnected")

            if hasattr(self, 'leave_client') and self.leave_client:
                await self.leave_client.__aexit__(None, None, None)
                print("✅ Leave client disconnected")

        except Exception as e:
            print(f"❌ Error during client cleanup: {e}")


async def main():
    # Set your Gemini API key
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # Check if API key is available
    if not GEMINI_API_KEY:
        print("❌ Error: GEMINI_API_KEY not found in environment variables")
        print("Please create a .env file with: GEMINI_API_KEY=your_api_key_here")
        return

    # Initialize chatbot
    chatbot = ChatbotClient(GEMINI_API_KEY)

    try:
        # Initialize MCP connections
        await chatbot.initialize_mcp_clients()

        print("\n🤖 Chatbot Ready! Type 'quit' to exit.")
        print("You can ask about employee information or leave management.")
        print("\nExample queries:")
        print("- What's employee E001's information?")
        print("- Find employees in Engineering department")
        print("- Show me team members for manager M001")
        print("- Search for Marketing Manager positions")
        print("- How many leave days does E002 have?")
        print("- Apply leave for E001 on 2025-08-15")
        print("-" * 50)

        # Chat loop
        while True:
            user_input = input("\n👤 You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break

            if not user_input:
                continue

            # Get response
            response = await chatbot.chat(user_input)
            print(f"\n🤖 Assistant: {response}")

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Clean up connections
        await chatbot.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
