
#Note: The openai-python library support for Azure OpenAI is in preview.
import os
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_type = "azure"
openai.api_base = openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = os.getenv("OPEN_API_VERSION")
openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  engine="test-gpt-35-turbo",
  prompt="Write a product launch email for new AI-powered headphones that are priced at $79.99 and available at Best Buy, Target and Amazon.com. The target audience is tech-savvy music lovers and the tone is friendly and exciting.\n\n1. What should be the subject line of the email?  \n2. What should be the body of the email?",
  temperature=1,
  max_tokens=350,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None)

print(response)