import os.path  # 导入os.path模块，用于处理文件和目录路径
import json  # 导入json模块，用于处理JSON数据
import random  # 导入random模块，用于生成随机数
import tiktoken  # 导入tiktoken模块，用于处理文本编码
import copy  # 导入copy模块，用于深拷贝对象
from tqdm import tqdm  # 从tqdm模块导入tqdm，用于显示进度条
from pprint import pprint  # 从pprint模块导入pprint，用于美化打印输出
from .utils import attribute, embedding, data_format, RAG_eval, math_eval, self_reflection  # 从当前包的utils模块导入多个工具函数和类
from unigen.utils.IO import print, input  # 从unigen.utils.IO模块导入print和input函数
from unigen.utils.LLM_model import ModelAPI  # 从unigen.utils.LLM_model模块导入ModelAPI类
from unigen.utils.embedding import EmbeddingProcessor  # 从unigen.utils.embedding模块导入EmbeddingProcessor类
from unigen.utils.file_process import save_json, load_json, check_and_rename_file  # 从unigen.utils.file_process模块导入多个文件处理函数
from joblib import Parallel, delayed  # 从joblib模块导入Parallel和delayed，用于并行计算
from threading import Thread  # 从threading模块导入Thread类，用于创建线程
from queue import Queue  # 从queue模块导入Queue类，用于创建队列
import warnings  # 导入warnings模块，用于处理警告信息
import traceback  # 导入traceback模块，用于跟踪异常
from dataclasses import dataclass, field  # 从dataclasses模块导入dataclass和field，用于创建数据类
from concurrent.futures import ThreadPoolExecutor, as_completed  # 从concurrent.futures模块导入ThreadPoolExecutor和as_completed，用于线程池管理
from datetime import datetime  # 从datetime模块导入datetime类，用于处理日期和时间
from tenacity import retry, wait_random_exponential, stop_after_attempt  # 从tenacity模块导入retry, wait_random_exponential, stop_after_attempt，用于重试机制
from .utils.prompt import prompt_template  # 从当前包的utils.prompt模块导入prompt_template

warnings.filterwarnings("ignore")  # 忽略所有警告信息

class UniGen:
    def __init__(self, config, **kwargs):
        self.config = config  # 配置参数
        self.efficiency_configuration = config['efficiency_configuration']  # 效率配置

        generation_config = config['generation_settings']  # 生成设置
        self.batch_size = generation_config['batch_size']  # 批处理大小
        self.random_example = generation_config['random_example']  # 随机示例
        self.few_shot_num = generation_config['few_shot_num']  # few-shot示例数量
        self.generation_number = generation_config['generation_number']  # 生成数量
        self.max_worker = generation_config['max_worker']  # 最大工作线程数
        self.temperature = generation_config['temperature']  # 温度参数
        
        self.dataset_config = config['generation_hint']  # 数据集配置
        self.dataset_name = self.dataset_config["dataset_name"]  # 数据集名称
        self.dataset_description = self.dataset_config['dataset_description']  # 数据集描述
        self.constraints = self.dataset_config["dataset_constraint"]  # 数据集约束
        
        self.with_label = self.dataset_config['with_label']  # 是否包含标签
        self.with_attribute = self.dataset_config['with_attribute']  # 是否包含属性
        self.add_attribute = self.dataset_config['add_attribute']  # 是否添加属性
        self.attribute_key = self.dataset_config['attribute_key'] if self.with_attribute or self.add_attribute else None  # 属性键
        self.extra_info_keys = self.dataset_config.get('extra_info_keys', [])  # 额外信息键
        
        self.Embedder = EmbeddingProcessor(self.config)  # 嵌入处理器
        self.LLM_model = ModelAPI(self.config)  # 模型API

    def _get_dataset_keys(self):
        keys = ['text', 'label'] + self.extra_info_keys  # 获取数据集键
        if self.attribute_key:
            keys.append(self.attribute_key)  # 如果有属性键，添加到键列表中
        return keys

    def example_selection(self, random_=False):
        data = self.Embedder.preprocess_original_dataset()  # 预处理原始数据集
        keys = self._get_dataset_keys()  # 获取数据集键
        filtered_data = [{k: item[k] for k in keys if k in item} for item in data]  # 过滤数据
        
        if random_:
            random.shuffle(data)  # 随机打乱数据
            examples = data[:self.few_shot_num]  # 选择few-shot示例
            filtered_examples = [{k: item[k] for k in keys if k in item} for item in examples]  # 过滤示例
        else:
            examples = self.Embedder.cluster_embeddings(data, num_clusters=self.few_shot_num)  # 聚类嵌入
            filtered_examples = [{k: item[k] for k in keys if k in item} for item in examples]  # 过滤示例
            
        random.shuffle(filtered_examples)  # 随机打乱过滤后的示例

        return filtered_examples

    def few_shot_description(self, examples):
        random.shuffle(examples)  # 随机打乱示例
        json_output = json.dumps(examples, indent=4)  # 将示例转换为JSON格式
        return json_output

    def add_constraints(self, constraints):
        constraints_text = prompt_template["constraints_prefix"]  # 约束前缀
        for i, constraint in enumerate(constraints, 1):
            constraints_text += f"{i}. {constraint}\n"  # 添加约束
        constraints_text += prompt_template["constraints_suffix"]  # 约束后缀
        return constraints_text

    def learn_from_human_feedback(self, examples):
        for example in examples:
            self._collect_user_feedback(example)  # 收集用户反馈
        feedback_string = ""
        for index, item in enumerate(examples):
            if 'label' in item:
                feedback_string += "Example: " + item['text'] + "\n" + "Label: " + item['label'] + '\n' + "Human Feedback: " + item['feedback'] + '\n\n'
            else:
                feedback_string += "Example: " + item['text'] + "\n" + "Human Feedback: " + item['feedback'] + '\n\n'
        return feedback_string

    def _collect_user_feedback(self, example):
        if 1:
            feedback = 'good'  # 默认反馈为“good”
            example['feedback'] = feedback
            return
        print("---------------------Please input your feedback-------------------------", "GREEN")
        if 'label' in example:
            print(f"Example: Text: {example['text']},    Label: {example['label']}")
            print("-------------------------------------------------------------------------", "GREEN")
            feedback = input("Please provide your feedback: ")
        else:
            print(f"Example: Text: {example['text']}")
            print("-------------------------------------------------------------------------", "green")
            feedback = input("Please provide your feedback: ", "red")
        example['feedback'] = feedback

    def count_tokens(self, text):
        enc = tiktoken.encoding_for_model(self.model)  # 获取模型的编码
        num_tokens = len(enc.encode(text))  # 计算文本的token数量
        return num_tokens

    def run(self,):
        assert self.generation_number % self.batch_size == 0, "generation_number must be divisible by batch_size"  # 断言生成数量必须是批处理大小的整数倍
        self.Embedder = embedding.EmbeddingProcessor(config=self.config)  # 嵌入处理器
        self.Embedder.preprocess_original_dataset()  # 预处理原始数据集
                
        save_path = self.dataset_config['save_path']  # 保存路径
        data_path = os.path.join(save_path, f"{self.dataset_name}_generated.json")  # 数据路径
        generated_data_file_path = check_and_rename_file(data_path)  # 检查并重命名文件
        print(generated_data_file_path)
        
        @retry(wait=wait_random_exponential(min=5, max=20), stop=stop_after_attempt(3))  # 重试机制
        def batch_generation(batch_id, queue):
            try:
                batch_data = []
                if self.few_shot_num > 0:
                    examples = self.example_selection(self.random_example)  # 选择示例
                    few_shot_des = self.few_shot_description(examples)  # few-shot描述
                else:
                    constraint_des = ""
                if self.constraints != []:
                    constraint_des = self.add_constraints(self.constraints)  # 添加约束
                else:
                    constraint_des = ""
                description_prompt = prompt_template["description_prompt"].format(
                    description_for_dataset=self.dataset_description,
                )
                initial_prompt = prompt_template["initial_prompt"].format(batch_size=self.batch_size,
                                                                           dataset_constraint=constraint_des,
                                                                           few_shot_examples=few_shot_des)
                epoch_prompt = description_prompt + initial_prompt

                if self.add_attribute and not self.with_attribute:
                    examples = attribute.get_attribute(examples, dataset_description=self.dataset_description)  # 获取属性
                    self.with_attribute = True
                if self.with_attribute:
                    epoch_prompt += attribute.add_attributes(examples=examples, attribute_key=self.attribute_key, attr=None)  # 添加属性
                epoch_prompt += data_format.create_data_entries(num_elements=self.batch_size, with_label=self.with_label, attribute_key=self.attribute_key)  # 创建数据条目
                
                res_data = data_format.get_res_data(epoch_prompt)  # 获取响应数据
                epoch_data_item = data_format.extract_data_item(res_data)  # 提取数据条目

                if self.efficiency_configuration["self_reflection"]:
                    reflection_res = self_reflection.reflection(epoch_data_item, self.dataset_description,
                                                                few_shot_des, constraint_des)  # 自我反思
                    for index, reflect_res in enumerate(reflection_res):
                        if reflect_res['text']:
                            batch_data.append(reflect_res)
                        else:
                            batch_data.append(epoch_data_item[index])
                else:
                    batch_data += epoch_data_item
                if self.efficiency_configuration["math_eval"]:
                    for item in batch_data:
                        batch_data[batch_data.index(item)] = math_eval.math_eval(item)  # 数学评估

                if self.efficiency_configuration["truthfulness_eval"]:
                    if batch_data:
                        for item in batch_data:
                            truthfulness_eval_res = RAG_eval.wiki_check(item)  # 真实性评估
                            if truthfulness_eval_res:
                                batch_data[batch_data.index(item)] = truthfulness_eval_res
                            else:
                                batch_data[batch_data.index(item)] = item
                queue.put(batch_data)
                return batch_data
            except Exception as e:
                print(traceback.format_exc())
                return None

        total_batches = int(self.generation_number / self.batch_size)  # 总批次数
        print(f'total_batches:{total_batches}', color='GREEN', )

        def save_dataset(generated_dataset):
            """
            Save the dataset to a JSON file.
            """
            try:
                current_time = datetime.now()  # 当前时间
                human_readable_time = current_time.strftime("%Y-%m-%d %H:%M:%S")  # 可读时间格式
                config = copy.deepcopy(self.config)  # 深拷贝配置
                del config['api_settings']  # 删除API设置
                config['data_entry_num'] = len(generated_dataset)  # 数据条目数量
                filtered_dataset = list(filter(lambda item: item['isgood'], generated_dataset))  # 过滤数据集
                config['filtered_data_entry_num'] = len(filtered_dataset)  # 过滤后的数据条目数量
                genset = {
                    'update_time': human_readable_time,
                    "config": config,
                    "dataset": generated_dataset
                }
                    
                base_dir = os.path.dirname(generated_data_file_path)  # 基础目录
                if not os.path.exists(base_dir):
                    os.makedirs(base_dir)  # 创建目录
                    
                save_json(genset, generated_data_file_path)  # 保存JSON文件
                print(f"Data save path:{generated_data_file_path}\n\n")
                print("Dataset saved successfully.", color='BLUE', )
            except Exception as e:
                print(f"Failed to save dataset: {e}")

        all_data = []

        def save_data_to_file(queue):
            while True:
                data = queue.get() 
                if data == "DONE":
                    break
                all_data.extend(data)
                save_dataset(all_data)
                queue.task_done()

        data_queue = Queue()
        save_thread = Thread(target=save_data_to_file, args=(data_queue,))
        save_thread.start()
        with ThreadPoolExecutor(max_workers=self.max_worker) as executor:
            futures = [executor.submit(batch_generation, batch_id=i, queue=data_queue) for i in range(total_batches)]
            for _ in tqdm(as_completed(futures), total=total_batches, desc="Processing Batches"):
                pass
        data_queue.put("DONE")
        save_thread.join()
        
        
def unigen_generation(config):
    generator = UniGen(config)
    generator.run()


def eval_generation(config):
    dataset_name = config['dataset_name']
    generation_number = config['generation_number']
    data_file_path = config['data_file_path']
    generated_data_file_path = config['generated_file']

    data_file_path = f"test_dataset/{dataset_name}/{dataset_name}.json"
    generator = UniGen(config)
    generator.run(data_file_path, generated_data_file_path)
    
def eval_generation(config):
    dataset_name = config['dataset_name']
    generation_number = config['generation_number']
    data_file_path = config['data_file_path']
    generated_data_file_path = config['generated_file']

    data_file_path = f"test_dataset/{dataset_name}/{dataset_name}.json"
    generator = UniGen(config)
    generator.run(data_file_path, generated_data_file_path)
