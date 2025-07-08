import asyncio
from mcp import ClientSession
from contextlib import AsyncExitStack
from openai import OpenAI
import os
from dotenv import load_dotenv

class MCPClient(AsyncExitStack):
    def __init__(self):
        self.session = None
        self.exit_stack = AsyncExitStack()
        load_dotenv()  # 确保加载 .env 文件中的变量
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("BASE_URL")
        self.model = os.getenv("MODEL")
        if not self.openai_api_key:
            raise ValueError("未找到OpenAI API Key")
        self.client = OpenAI(api_key=self.openai_api_key, base_url=self.base_url)

    async def process_query(self, query: str) -> str:
        """调用API 处理用户查询"""
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query},  # 使用传入的 query 而不是硬编码的字符串
            ],
            # Qwen3模型通过enable_thinking参数控制思考过程（开源版默认True，商业版默认False）
            # 使用Qwen3开源版模型时，若未启用流式输出，请将下行取消注释，否则会报错
            # extra_body={"enable_thinking": False},
        )
        print(completion.model_dump_json())
        return completion.choices[0].message.content

    async def chat_loop(self):
        """运行交互式聊天循环"""
        print("\n🤖 MCP 客户端已启动！输入 'quit' 退出")

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
    client = MCPClient()
    try:
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
