import os
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Tuple
load_dotenv()
from jinja2 import Template

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
    log_dir = os.path.join("project", "chat_log")
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

def multi_chat(
    user_input: str,
    messages: List[Dict[str, str]] = None,
    max_history: int = 5
) -> Tuple[str, List[Dict[str, str]]]:
    """
    基于 gpt(content) 的多轮对话封装
    
    Args:
        user_input: 当前用户输入
        messages: 历史消息列表，格式 [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]
        max_history: 最大保留对话轮次（按用户输入计数）

    Returns:
        tuple: (当前回复内容, 更新后的对话历史)
    """
    # 初始化对话历史
    messages = messages.copy() if messages else []
    
    try:
        # 添加当前用户输入
        messages.append({"role": "user", "content": user_input})
        
        # 构建对话上下文
        context = "\n".join([f"[{msg['role']}] {msg['content']}" for msg in messages])
        
        # 调用单轮对话函数
        assistant_reply = gpt(context)  # 调用 gpt(content: str) -> str
        # assistant_reply = "ttttttt"

        # 添加助手回复到历史
        messages.append({"role": "assistant", "content": assistant_reply})
        
        # 控制历史长度（保留最近 N 轮完整对话）
        if max_history > 0:
            keep_items = max_history * 2  # 每轮包含 user+assistant
            messages = messages[-keep_items:] if len(messages) > keep_items else messages
            
        return assistant_reply, messages
    
    except Exception as e:
        # 这里捕获 gpt() 函数可能抛出的异常
        error_msg = f"对话失败: {str(e)}"
        return error_msg, messages


class PromptTemplate:
    ''' 一个类，用于表示带变量的提示模板
    属性：
        template (str): 带变量的模板字符串
        input_variables (list): 模板中的变量名称列表
    '''
    def __init__(self, template, input_variables):
        self.template = Template(template)
        self.input_variables = input_variables
    
    def format(self, **kwargs):
        return self.template.render(**kwargs)
    
def create_chain(prompt_template):
    """
    使用给定的提示模板创建一个基于 gpt 的链。

    参数：
        prompt_template (str): 提示模板字符串。

    Returns:
        function: 一个函数，接受输入变量并返回 gpt 的响应。
    """
    # 创建 PromptTemplate 实例
    prompt = PromptTemplate(template=prompt_template, input_variables=[])

    def chain(**kwargs):
        """
        使用模板生成提示并调用 gpt 获取响应。

        参数：
            kwargs: 模板变量的键值对。

        返回：
            str: gpt 的响应内容。
        """
        # 格式化模板
        formatted_prompt = prompt.format(**kwargs)
        # 调用 gpt 方法
        response = gpt(formatted_prompt)
        return response

    return chain

# 显示模型输出的函数
def display_output(output):
    """以格式化方式显示模型的输出。"""
    print("模型输出:")
    print("-" * 40)
    print(output)
    print("-" * 40)
    print()

def dd(text):
    print(text)
    exit()