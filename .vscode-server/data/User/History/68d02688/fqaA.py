import os
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import random
import concurrent.futures
from openai import OpenAI, AzureOpenAI
#from .file_process import save_json, load_json

class EmbeddingProcessor:
    def __init__(self, config):
        self.config = config  # 初始化EmbeddingProcessor类，保存配置

    def preprocess_original_dataset(self):
        original_dataset_path = self.config['generation_hint']['original_dataset_path']  # 获取原始数据集路径
        base_path = os.path.splitext(original_dataset_path)[0]  # 获取文件基础路径（不含扩展名）
        embedding_path = f"{base_path}_dataset_embedding.json"  # 构建嵌入文件路径
        if not os.path.exists(embedding_path):  # 如果嵌入文件不存在
            data = load_json(original_dataset_path)  # 加载原始数据集
            embeddings = self.generate_dataset_embedding(data)  # 生成数据集嵌入
            save_json(embeddings, embedding_path)  # 保存嵌入到文件
        else:
            embeddings = load_json(embedding_path)  # 加载已有的嵌入文件
        #return embeddings  # 返回嵌入数据
        print(base_path)

    def get_embedding(string, config):
        settings = config["api_settings"]  # 获取API设置
        azure = settings['use_azure']  # 检查是否使用Azure
        if azure:
            client = AzureOpenAI(
                azure_endpoint=settings["api_base"],
                api_key=settings['azure_api_key'],
                api_version=settings["azure_version"],
            )  # 初始化AzureOpenAI客户端
            response = client.embeddings.create(
                model=settings["embedding_model"],
                input=string
            )  # 使用Azure API生成嵌入
        else:
            client = OpenAI(api_key=settings['openai_api_key'])  # 初始化OpenAI客户端
            response = client.embeddings.create(
                model=settings['embedding_model'],
                input=string
            )  # 使用OpenAI API生成嵌入
        return response.data[0].embedding  # 返回嵌入结果

    def get_single_item_embedding(self, item):
        item["embedding"] = EmbeddingProcessor.get_embedding(item["text"], self.config)  # 获取单个项目的嵌入
        return item  # 返回带有嵌入的项目

    def generate_dataset_embedding(self, data):
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            embeddings = list(filter(None, executor.map(self.get_single_item_embedding, data)))  # 并行生成数据集嵌入
        return embeddings  # 返回嵌入列表

    def select_embeddings_with_auto_min_similarity(self, embeddings, n, start_similarity=0.0, decrement=0.05):
        embeddings_array = np.array([el['embedding'] for el in embeddings])  # 将嵌入转换为数组
        similarity_matrix = cosine_similarity(embeddings_array)  # 计算余弦相似度矩阵
        min_similarity = start_similarity  # 初始化最小相似度

        while min_similarity < 1:
            selected_indices = [np.random.choice(range(len(embeddings)))]  # 随机选择一个初始嵌入
            available_indices = set(range(len(embeddings))) - set(selected_indices)  # 获取可用索引集合

            while len(selected_indices) < n and available_indices:
                min_similarities = similarity_matrix[selected_indices, :][:, list(available_indices)].min(axis=0)
                candidates = [i for i, sim in zip(available_indices, min_similarities) if sim <= min_similarity]
                if not candidates:
                    break  # 如果没有候选项，退出循环
                new_selected = np.random.choice(candidates)  # 随机选择一个新的嵌入
                selected_indices.append(new_selected)  # 添加到已选择列表
                available_indices.remove(new_selected)  # 从可用索引中移除

            if len(selected_indices) == n:
                return selected_indices  # 如果选择的嵌入数量达到要求，返回索引列表
            min_similarity += decrement  # 增加最小相似度

        raise ValueError("Unable to find enough embeddings with any minimum similarity threshold.")  # 抛出异常

    def cluster_embeddings(self, embeddings, num_clusters, method='cosine_similarity'):
        embeddings_array = np.array([el['embedding'] for el in embeddings])  # 将嵌入转换为数组
        assert method in ['kmeans', 'agglomerative', 'cosine_similarity']  # 确保方法合法
        clustering_model = None
        if method == 'kmeans':
            clustering_model = KMeans(n_clusters=num_clusters)  # 使用KMeans聚类
        elif method == 'agglomerative':
            clustering_model = AgglomerativeClustering(n_clusters=num_clusters)  # 使用层次聚类
        elif method == 'cosine_similarity':
            selected_indices = self.select_embeddings_with_auto_min_similarity(embeddings, num_clusters)  # 使用余弦相似度选择嵌入
            return [embeddings[idx] for idx in selected_indices]  # 返回选择的嵌入

        labels = clustering_model.fit_predict(embeddings_array)  # 进行聚类并获取标签
        cluster_embeddings = {cluster_label: [] for cluster_label in np.unique(labels)}  # 初始化聚类结果字典

        for i, label in enumerate(labels):
            cluster_embeddings[label].append(embeddings[i])  # 将嵌入添加到对应的聚类中

        random_embeddings = {label: random.choice(cluster) for label, cluster in cluster_embeddings.items()}  # 随机选择每个聚类中的一个嵌入
        selected_indices = [embeddings.index(random_embeddings[label]) for label in random_embeddings]  # 获取选择的嵌入索引

        return [embeddings[idx]['text'] for idx in selected_indices]  # 返回选择的嵌入文本

   
config_file = r"UniGen/examples/generation_config_example.yaml"
temp_item = EmbeddingProcessor(config_file)
temp_item.preprocess_original_dataset()
