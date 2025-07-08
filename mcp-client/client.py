import asyncio
import os
import json
from typing import Optional
from contextlib import AsyncExitStack
from openai import OpenAI  
from dotenv import load_dotenv
import base64
import re
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
 
# 加载 .env 文件，确保 API Key 受到保护
load_dotenv()

class MCPClient:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("BASE_URL")
        self.model = os.getenv("MODEL")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set in.env file")
        self.client = OpenAI(api_key=self.openai_api_key,base_url=self.base_url)
        self.session: Optional[ClientSession] = None
        
    async def connect_to_server(self,server_script_path:str):
        """connect to server and list tools"""
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not is_python and not is_js:
            raise ValueError("server_script_path should end with .py or.js")
        
        command = 'python' if is_python else 'node'
        server_params = StdioServerParameters(
            command = command,
            args = [server_script_path], 
            env = None
        )

        #set up MCP server and connect to it
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio,self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio,self.write))
        
        await self.session.initialize()

        #list tools in MCP server
        response = await self.session.list_tools()
        tools = response.tools
        print("Tools available on server: ",[tool.name for tool in tools])

    async def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")


    async def extract_video_path(self, query: str) -> str:
        # 更宽松的正则表达式，匹配Windows路径格式
        pattern = r'([a-zA-Z]:[\\\/][^<>:"|?*\n\r]*\.(?:mp4|avi|mov|mkv|flv))'
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # 如果第一个模式没匹配到，尝试更简单的模式
        simple_pattern = r'([a-zA-Z]:\\.*?\.(?:mp4|avi|mov|mkv|flv))'
        match = re.search(simple_pattern, query, re.IGNORECASE)
        if match:
            return match.group(1)
        
        return None

    async def extract_text_question(self, query: str) -> str:
        """从查询中提取纯文本问题，去除视频路径"""
        pattern = r'([a-zA-Z]:[\\\/][^<>:"|?*\n\r]*\.(?:mp4|avi|mov|mkv|flv))'
        text_question = re.sub(pattern, '', query, flags=re.IGNORECASE).strip()
        
        # 如果去除路径后文本为空，使用默认问题
        text_question = re.sub(pattern, '', query).strip()
        
        # 如果去除路径后文本为空，使用默认问题
        if not text_question:
            return "这段视频的内容是什么？"
        
        return text_question


    async def process_query(self,query:str) -> str:
        messages = [{
            'role':'user',
            'content':query
                     }]
        response = await self.session.list_tools()
        available_tools = [{
            "type":"function",
            "function":{
                "name":tool.name,
                "description":tool.description,
                "input_schema": tool.inputSchema
            }
        }for tool in response.tools]
        print(f"tool description: {available_tools}")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools = available_tools,
        )

        content = response.choices[0]
        if content.finish_reason == "tool_calls":
            tool_call = content.message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            if tool_name == "inference_video":
                #path validation ,video encode ,text question
                text_question = await self.extract_text_question(query)
                video_path = await self.extract_video_path(query)
                base_video = await self.encode_image(video_path)

                #video inference
                video_result = await self.session.call_tool("inference_video", {"path": video_path})
                video_params = json.loads(video_result.content[0].text) if video_result.content else {}
                video_info = f"视频信息：FPS={video_params.get('fps')}, 时长={video_params.get('duration')}秒, 总帧数={video_params.get('frame_count')}"


                messages.append({
                    "role":"user",
                    "content": [
                        {
                            "type": "video_url", 
                            "video_url": {"url": f"data:video/mp4;base64,{base_video}"}
                        },
                        {"type": "text", "text": f"{text_question}\n{video_info}"}
                    ]
                })
            
            if tool_name == "process_video_binarization":
                text_question = await self.extract_text_question(query)
                video_path = await self.extract_video_path(query)
                threshold = tool_args.get("threshold", 127)
                base_video = await self.encode_image(video_path)


                binary_result = await self.session.call_tool("process_video_binarization", {"path": video_path, "threshold": threshold})
                binary_params = json.loads(binary_result.content[0].text) if binary_result.content else {}
                binary_info = f"视频信息：FPS={binary_params.get('fps')}, 时长={binary_params.get('duration')}秒,总帧数={binary_params.get('frame_count')},分辨率={binary_params.get('resolution')}"

                messages.append({
                    "role":"user",
                    "content": [
                        {
                            "type": "video_url", 
                            "video_url": {"url": f"data:video/mp4;base64,{base_video}"}
                        },
                        {"type": "text", "text": f"{text_question}\n{binary_info}"}
                    ]
                })

            result = await self.session.call_tool(tool_name,tool_args)

            print(f"\n\n[Calling tool {tool_name} with args {tool_args}]\n\n")

            messages.append(content.message.model_dump())
            messages.append({
                "role":"tool",
                "content":result.content[0].text,
                "tool_call_id":tool_call.id,
            })

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            return response.choices[0].message.content

        return content.message.content


    async def chat_loop(self):
        """运行交互式聊天循环"""
        print("\n🤖 MCP 客户端已启动！若需要图像视频传入请输入'1',输入 'quit' 退出")
 
        while True:
            try:
                query = input("\n你: ").strip()
                if query.lower() == 'quit':
                    break
      
                response = await self.process_query(query)  # 发送用户输入到 OpenAI API
                print(f"\n🤖 OpenAI: {response}")
 
            except Exception as e:
                print(f"\n⚠️ 发生错误: {str(e)}")
 
    async def cleanup(self):
        """清理资源"""
        await self.exit_stack.aclose()
 
async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)
 
    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()
 
if __name__ == "__main__":
    import sys
    asyncio.run(main())

