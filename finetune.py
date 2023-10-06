import os
import openai
from dotenv import load_dotenv

load_dotenv()


openai.api_type = "azure"
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = os.getenv("OPEN_API_VERSION")
openai.api_key = os.getenv("OPENAI_API_KEY")



file = openai.File.create(
  file=open("hello.jsonl", "rb"),
  purpose='fine-tune',
  user_provided_filename="hello.jsonl"
)

print(f"Uploaded: {file}")

fine_tuned_model = openai.FineTuningJob.create(
  training_file=file["id"], 
  model="test-gpt-35-turbo"
  )

print(f"fined tune model returned: {fine_tuned_model}")

completion = openai.ChatCompletion.create(
  model=fine_tuned_model["id"],
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion.choices[0].message)