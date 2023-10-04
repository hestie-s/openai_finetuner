import os
import openai
from dotenv import load_dotenv

load_dotenv()


openai.api_type = "azure"
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = os.getenv("OPEN_API_VERSION")
openai.api_key = os.getenv("OPENAI_API_KEY")



files = openai.File.list()

print(files)

for file in files.data:
  print("delete", file)
  openai.File.delete(file["id"]);