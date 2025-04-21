import os
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

def gpt(content):
    """
    调用 DeepSeek API 获取聊天响应，并将输入和输出记录到日志文件。

    参数:
        content (str): 用户输入的提示内容。

    返回:
        str: 模型返回的响应内容。
    """
    # 记录开始时间
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 调用 API 获取响应
    client = OpenAI(api_key=os.getenv('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": content},
        ],
        stream=False
    )
    result = response.choices[0].message.content

    # 记录结束时间
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 构造日志数据
    log_entry = {
        "start_time": start_time,
        "end_time": end_time,
        "input_message": content,
        "output_response": result
    }

    # 确保日志目录存在
    log_dir = os.path.join("chat_log")
    os.makedirs(log_dir, exist_ok=True)

    # 根据当前日期生成日志文件名
    current_date = datetime.now().strftime("%Y%m%d")
    log_file = os.path.join(log_dir, f"prompt_log_{current_date}.log")

    # 如果文件不存在，自动创建
    if not os.path.exists(log_file):
        with open(log_file, "w", encoding="utf-8") as file:
            file.write("")  # 创建空文件

    # 写入日志文件
    with open(log_file, "a", encoding="utf-8") as file:
        file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    return result