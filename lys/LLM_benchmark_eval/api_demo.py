from openai import OpenAI
import openai
import os

os.environ["OPENAI_API_KEY"] = "sk-dUXHtWYlEgQOhnNC82opf5HhVrfQjHUzcTDRnOpwdDI01Jev"
# os.environ['OPENAI_API_BASE_URL']
# os.environ["http_proxy"] = 'http://127.0.0.1:7890'
# os.environ["https_proxy"] = 'http://127.0.0.1:7890'

client = openai.OpenAI(
  base_url="https://api.keya.pw/v1",
)
completion = client.chat.completions.create(
  model="gpt-4",
  messages=[
    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
  ]
)

print(completion.choices[0].message.content)