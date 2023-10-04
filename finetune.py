import os
import openai
from dotenv import load_dotenv

load_dotenv()


openai.api_type = "azure"
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = os.getenv("OPEN_API_VERSION")
openai.api_key = os.getenv("OPENAI_API_KEY")



file = openai.File.create(
  file=open("train.jsonl", "rb"),
  purpose='fine-tune',
  user_provided_filename="train.jsonl"
)


fine_tuned_model = openai.FineTuningJob.create(training_file=file["id"], model="gpt-3.5-turbo")


completion = openai.ChatCompletion.create(
  model=fine_tuned_model["id"],
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion.choices[0].message)