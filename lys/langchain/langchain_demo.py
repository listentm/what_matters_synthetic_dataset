from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
import pdb
import openai
api_key = "sk-dUXHtWYlEgQOhnNC82opf5HhVrfQjHUzcTDRnOpwdDI01Jev"
api_base="https://api.keya.pw/v1"
# 定义链式任务
llm = ChatOpenAI(temperature=0.7, model_name="gpt-4",openai_api_key=api_key,openai_api_base=api_base)

# 定义 Prompt 模板
prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are a reasoning assistant. Answer the following question step by step.

Question: {question}
Answer step-by-step:
""",
)
pdb.set_trace()
# 创建任务链
chain = LLMChain(llm=llm, prompt=prompt)

# 执行任务
question = "If a train travels at 60 km/h for 2 hours, how far does it travel?"
answer = chain.run(question)

print("Chain of Thought Answer:")
print(answer)
