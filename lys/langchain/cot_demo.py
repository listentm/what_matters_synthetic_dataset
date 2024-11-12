import openai

# 设置 API 密钥
openai.api_key = "sk-dUXHtWYlEgQOhnNC82opf5HhVrfQjHUzcTDRnOpwdDI01Jev"
openai.api_base="https://api.keya.pw/v1"
# 定义问题和提示模板
def chain_of_thought(prompt):
    cot_prompt = f"""
You are a helpful assistant skilled in reasoning. Answer the question step by step to ensure logical correctness.

Question: {prompt}
Answer step-by-step:
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",  # 使用 GPT-4 模型
        messages=[{"role": "system", "content": "You are a reasoning assistant."},
                  {"role": "user", "content": cot_prompt}],
        max_tokens=100,
        temperature=0.7,
    )
    return response['choices'][0]['message']['content']

# 示例问题
question = "Xiao Ming was born in 1990, and Xiao Hong was born in 1995. What is the age difference between them?"
answer = chain_of_thought(question)

print("Chain of Thought Answer:")
print(answer)
