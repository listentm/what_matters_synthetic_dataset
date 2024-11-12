from tenacity import retry, wait_random_exponential, stop_after_attempt  # 导入tenacity库中的重试机制
from openai import OpenAI, AzureOpenAI  # 导入OpenAI和AzureOpenAI库
from anthropic import Anthropic  # 注释掉的anthropic库导入
import traceback  # 导入traceback库，用于异常处理
from zhipuai import ZhipuAI  # 导入ZhipuAI库

def singleton(cls):  # 定义单例模式装饰器
    instances = {}  # 存储实例的字典

    def get_instance(*args, **kwargs):  # 获取实例的内部函数
        if cls not in instances:  # 如果实例不存在
            instances[cls] = cls(*args, **kwargs)  # 创建实例并存储
        return instances[cls]  # 返回实例

    return get_instance  # 返回内部函数

@singleton  # 使用单例模式装饰器
class ModelAPI:  # 定义ModelAPI类
    def __init__(self, config, model_type='glm', temperature=0.8):  # 初始化方法
        self.config = config  # 配置参数
        self.model_type = config['generation_settings']['model_type'].lower()  # 模型类型
        self.temperature = config['generation_settings']['temperature']  # 温度参数
        if self.model_type not in ['gpt', 'claude', 'llama3', 'glm']:  # 检查模型类型是否支持
            raise ValueError(f"Unsupported model type: {model_type}")  # 抛出异常

    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(10))  # 使用重试机制装饰器
    def get_res(self, text, model=None, message=None, azure=True, json_format=False):  # 获取响应的方法
        temperature = self.temperature  # 获取温度参数
        try:
            if self.model_type == 'gpt':  # 如果模型类型是gpt
                return self.api_send_gpt4(text, model, message, azure, json_format, temperature)  # 调用GPT4 API
            elif self.model_type == 'claude':  # 如果模型类型是claude
                self.api_key = self.config["api_settings"]["claude_api_key"]  # 获取API密钥
                return self.api_send_claude(text, model, message, json_format, temperature)  # 调用Claude API
            elif self.model_type == 'llama3':  # 如果模型类型是llama3
                self.api_key = self.config["api_settings"]["llama3_api_key"]  # 获取API密钥
                return self.api_send_llama3(text, model, message, json_format, temperature)  # 调用Llama3 API
            elif self.model_type == 'glm':  # 如果模型类型是glm
                self.api_key = self.config["api_settings"]["glm_api_key"] # 获取API密钥
                return self.api_send_glm4(text, model, message, json_format, temperature)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")  # 抛出异常
        except Exception as e:  # 捕获异常
            print(traceback.format_exc())  # 打印异常信息

    def api_send_llama3(self, string, model=None, message=None, json_format=False, temperature=0.8):  # 调用Llama3 API的方法
        client = OpenAI(api_key=self.api_key, base_url="https://api.deepinfra.com/v1/openai")  # 创建OpenAI客户端
        top_p = 1 if temperature <= 1e-5 else 0.9  # 设置top_p参数
        temperature = 0.01 if temperature <= 1e-5 else temperature  # 设置温度参数
        model = 'meta-llama/Meta-Llama-3-70B-Instruct'  # 设置模型名称
        print(f"Sending API request...{model},temperature:{temperature}")  # 打印请求信息
        chat_completion = client.chat.completions.create(  # 创建聊天完成请求
            model='meta-llama/Meta-Llama-3-70B-Instruct',
            messages=[{"role": "user", "content": string}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=4096,
        )
        if not chat_completion.choices[0].message.content:  # 检查响应内容是否为空
            raise ValueError("The response from the API is NULL or an empty string!")  # 抛出异常
        return chat_completion.choices[0].message.content  # 返回响应内容

    def api_send_claude(self, string, model="claude-3-opus-20240229", message=None, json_format=False, temperature=0.8):  # 调用Claude API的方法
        client = Anthropic(api_key=self.api_key)  # 创建Anthropic客户端
        model = "claude-3-opus-20240229"  # 设置模型名称
        print(f"Sending API request...{model}")  # 打印请求信息
        message = client.messages.create(  # 创建消息请求
            temperature=temperature,
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": string,
                }
            ],
            model="claude-3-opus-20240229",
        )
        if not message.choices[0].message.content:  # 检查响应内容是否为空
            raise ValueError("The response from the API is NULL or an empty string!")  # 抛出异常
        full_response = message.content[0].text  # 获取完整响应内容
        return full_response  # 返回响应内容

    def api_send_glm4(self, string, model = "glm-4-air", message=None, json_format=False, temperature=0.8):
        client = ZhipuAI(api_key=self.api_key)  # 创建ZhipuAI客户端
        model="glm-4-air"
        print(f"Sending API request...{model}")  # 打印请求信息
        response = client.chat.completions.create(
            model="glm-4-air",
            temperature=0.1,
            messages=[
                {
                    "role": "user" , 
                    "content": string 
                }
            ],
        )
        if not response.choices[0].message.content:
            raise ValueError("The response from the API is NULL or an empty string!")  # 抛出异常
        return response.choices[0].message.content  # 返回响应内容
        
    def api_send_gpt4(self, string, model, message=None, azure=True, json_format=False, temperature=0.8):  # 调用GPT4 API的方法
        azure = self.config["api_settings"]["use_azure"]  # 获取是否使用Azure的配置
        if message is None:  # 如果消息为空
            message = [{"role": "user", "content": string}]  # 设置默认消息
        response_format = {"type": "json_object"} if json_format else None  # 设置响应格式
        if azure:  # 如果使用Azure
            azure_endpoint = self.config["api_settings"]["azure_base_url"]  # 获取Azure端点
            api_key = self.config['api_settings']['azure_api_key']  # 获取API密钥
            api_version = self.config["api_settings"]["azure_version"]  # 获取API版本
            model = self.config["api_settings"]["azure_model"]  # 获取模型名称
            print(f"Sending API request...{model},temperature:{temperature}")  # 打印请求信息
            client = AzureOpenAI(  # 创建AzureOpenAI客户端
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version=api_version,
            )
            chat_completion = client.chat.completions.create(  # 创建聊天完成请求
                model=model,
                messages=message,
                temperature=temperature,
                response_format=response_format,
            )
        else:  # 如果不使用Azure
            model = self.config["api_settings"]["openai_chat_model"]  # 获取OpenAI聊天模型
            base_url = self.config["api_settings"].get("base_url")  # 获取基础URL
            api_key = self.config['api_settings']['openai_api_key']  # 获取API密钥

            # Correct the client initialization
            if base_url:  # 如果基础URL存在
                client = OpenAI(api_key=api_key, base_url=base_url)  # 创建OpenAI客户端
            else:
                client = OpenAI(api_key=api_key)  # 创建OpenAI客户端
            model = "gpt-4-0125-preview"  # 设置模型名称
            print(f"Sending API request...{model},temperature:{temperature}")  # 打印请求信息
            chat_completion = client.chat.completions.create(  # 创建聊天完成请求
                model=model,
                messages=message,
                temperature=temperature,
                response_format=response_format,
            )
        dict(chat_completion.usage)  # {'completion_tokens': 28, 'prompt_tokens': 12, 'total_tokens': 40}  # 打印使用情况
        if not chat_completion.choices[0].message.content:  # 检查响应内容是否为空
            raise ValueError("The response from the API is NULL or an empty string!")  # 抛出异常
        full_response = chat_completion.choices[0].message.content  # 获取完整响应内容
        return full_response  # 返回响应内容
