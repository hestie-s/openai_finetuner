import os
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_type = "azure"
openai.api_base = openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = os.getenv("OPEN_API_VERSION")
openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.ChatCompletion.create(
  engine="gpt-4",
  messages = [{"role":"system","content":"You are an AI assistant that helps people find information."},{"role":"user","content":"hello"}],
  temperature=0.7,
  max_tokens=800,
  top_p=0.95,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None)

print(response)