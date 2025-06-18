from typing import Optional
from contextlib import AsyncExitStack
import traceback
from utils.logger import logger
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from datetime import datetime
import json
import os

from openai import OpenAI

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.client = OpenAI(
            base_url="http://localhost:12434/engines/v1",  # Docker runner endpoint
            api_key="docker"
        )
        self.tools = []
        self.messages = []
        self.logger = logger

    async def call_tool(self, tool_name: str, tool_args: dict):
        try:
            result = await self.session.call_tool(tool_name, tool_args)
            return result
        except Exception as e:
            self.logger.error(f"Failed to call tool: {str(e)}")
            raise Exception(f"Failed to call tool: {str(e)}")

    async def connect_to_server(self, server_script_path: str):
        try:
            is_python = server_script_path.endswith(".py")
            is_js = server_script_path.endswith(".js")
            if not (is_python or is_js):
                raise ValueError("Server script must be a .py or .js file")

            self.logger.info(f"Connecting to server with: {server_script_path}")
            command = "python" if is_python else "node"
            server_params = StdioServerParameters(
                command=command, args=[server_script_path], env=None
            )

            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

            await self.session.initialize()
            mcp_tools = await self.get_mcp_tools()
            self.tools = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
                for tool in mcp_tools
            ]
            self.logger.info(f"Connected to server. Tools: {[t['name'] for t in self.tools]}")
            return True
        except Exception as e:
            self.logger.error(f"Server connection failed: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise

    async def get_mcp_tools(self):
        try:
            self.logger.info("Getting MCP tools from server...")
            response = await self.session.list_tools()
            return response.tools
        except Exception as e:
            self.logger.error(f"Tool listing failed: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise

    async def call_llm(self) -> dict:
        try:
            response = self.client.chat.completions.create(
                model="ai/llama3.2:1B-Q4_0",  # Change if your model ID differs
                messages=self.messages,
                tools=self.tools,
                max_tokens=1000,
            )
            return response
        except Exception as e:
            self.logger.error(f"LLM call failed: {str(e)}")
            raise

    async def process_query(self, query: str):
        try:
            self.logger.info(f"Processing query: {query[:100]}...")
            user_message = {"role": "user", "content": query}
            self.messages.append(user_message)
            await self.log_conversation(self.messages)
            messages = [user_message]

            while True:
                self.logger.debug("Calling local LLM")
                response = await self.call_llm()

                choice = response.choices[0].message
                content = choice.get("content", "")
                tool_calls = choice.get("tool_calls", [])

                assistant_message = {"role": "assistant", "content": content}
                self.messages.append(assistant_message)
                await self.log_conversation(self.messages)
                messages.append(assistant_message)

                if not tool_calls:
                    break  # Final response

                for tool_call in tool_calls:
                    tool_name = tool_call.get("function", {}).get("name")
                    tool_args = tool_call.get("function", {}).get("arguments")
                    tool_use_id = tool_call.get("id")

                    self.logger.info(f"Tool request: {tool_name} with args: {tool_args}")
                    try:
                        result = await self.call_tool(tool_name, json.loads(tool_args))
                        tool_result_message = {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_use_id,
                                    "content": result.content,
                                }
                            ],
                        }
                        self.messages.append(tool_result_message)
                        await self.log_conversation(self.messages)
                        messages.append(tool_result_message)
                    except Exception as e:
                        self.logger.error(f"Tool failed: {str(e)}")
                        raise

            return messages

        except Exception as e:
            self.logger.error(f"Query processing error: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise

    async def log_conversation(self, conversation: list):
        os.makedirs("conversations", exist_ok=True)
        serializable_conversation = []

        for message in conversation:
            try:
                serializable = {"role": message["role"], "content": []}
                if isinstance(message["content"], str):
                    serializable["content"] = message["content"]
                elif isinstance(message["content"], list):
                    for c in message["content"]:
                        if hasattr(c, 'to_dict'):
                            serializable["content"].append(c.to_dict())
                        elif hasattr(c, 'dict'):
                            serializable["content"].append(c.dict())
                        elif hasattr(c, 'model_dump'):
                            serializable["content"].append(c.model_dump())
                        else:
                            serializable["content"].append(c)
                serializable_conversation.append(serializable)
            except Exception as e:
                self.logger.error(f"Log message error: {str(e)}")
                self.logger.debug(f"Message: {message}")
                raise

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join("conversations", f"conversation_{timestamp}.json")
        try:
            with open(path, "w") as f:
                json.dump(serializable_conversation, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"File write error: {str(e)}")
            self.logger.debug(f"Data: {serializable_conversation}")
            raise

    async def cleanup(self):
        try:
            self.logger.info("Cleaning up MCP client")
            await self.exit_stack.aclose()
        except Exception as e:
            self.logger.error(f"Cleanup failed: {str(e)}")
