import requests
from lxml import etree
from bs4 import BeautifulSoup
import json
import time

# 读取爬取的论文HTML链接文件
with open(r'E:\benchmark_paper\cvpr2023_paper_links.json', 'r') as file:
    paper_links = json.load(file)

# 定义一个函数用于爬取论文标题和摘要，并处理重试机制
def get_paper_info(url, retries=3):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # 尝试多次重试请求
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # 如果返回错误状态码，抛出异常
            break  # 请求成功，跳出重试循环
        except requests.exceptions.RequestException as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            if attempt < retries - 1:
                time.sleep(2)  # 等待2秒后重试
            else:
                print(f"Failed to retrieve page: {url} after {retries} attempts")
                return None
    
    # 使用 BeautifulSoup 解析页面
    soup = BeautifulSoup(response.content, 'lxml')
    
    # 使用 lxml 的 etree 来解析 HTML
    dom = etree.HTML(str(soup))

    # 根据给定的 XPath 提取标题和摘要
    title = dom.xpath('//*[@id="papertitle"]/text()')
    abstract = dom.xpath('//*[@id="abstract"]/text()')

    # 如果标题或摘要为空，返回默认值
    title = title[0].strip() if title else "No Title Found"
    abstract = abstract[0].strip() if abstract else "No Abstract Found"

    return {"title": title, "abstract": abstract}

# 创建一个空字典，用于存储论文信息
papers_info = {}

# 遍历每个链接，爬取论文的标题和摘要，并实时保存到文件中
for link in paper_links:
    paper_info = get_paper_info(link)
    if paper_info:
        papers_info[link] = paper_info  # 将链接作为键，标题和摘要作为值
        print(paper_info)

        # 每爬取到一个论文信息后立即保存到 JSON 文件
        with open(r'E:\benchmark_paper\cvpr2023_papers_info.json', 'w') as json_file:
            json.dump(papers_info, json_file, indent=4)

print(f"Successfully saved paper information for {len(papers_info)} papers to 'cvpr2023_papers_info_partial.json'")