import pandas as pd
from sklearn.model_selection import train_test_split
import json
from dotenv import load_dotenv
import os
import openai

load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
MODEL = "gpt-3.5-turbo"

dataset_path = "./question_answer.xlsx"
df = pd.read_excel(dataset_path)

train, test = train_test_split(df, test_size=0.3)

def format_item(system_content, user_content, assistant_content):
  return {
      "messages": 
        [
          {"role": "system", "content": f"{system_content}"}, 
          {"role": "user", "content": f"{user_content}"}, 
          {"role": "assistant", "content": f"{assistant_content}"}
        ]
}

train = train.apply(lambda row: format_item("you are a blockchain expert", row[0], row[1]), axis=1)
test = test.apply(lambda row: format_item("you are a blockchain expert", row[0], row[1]), axis=1)

# Convert the formatted data to list before saving to jsonl
train = train.tolist()
test = test.tolist()

with open('train.jsonl', 'w') as f:
    for entry in train:
        json.dump(entry, f)
        f.write('\n')

with open('test.jsonl', 'w') as f:
    for entry in test:
        json.dump(entry, f)
        f.write('\n')

def upload_file_to_openai(file_path):
    with open(file_path, "rb") as f:
        return openai.File.create(file=f.read(), purpose="fine-tune")

def list_uploaded_files():
    return openai.File.list()


def fine_tune_model(train_file_id, test_file_id, model="text-davinci-002"):
    return openai.FineTuning.create(
        model=model,
        dataset=train_file_id,
        validation_dataset=test_file_id,
        n_epochs=10,
        learning_rate_multiplier=0.1
    )


def list_fine_tuning_models():
    return openai.FineTuning.list()

def check_fine_tune_events(fine_tuned_model_id):
    return openai.FineTuning.retrieve(fine_tuned_model_id)


def query_fine_tuned_model(prompt, fine_tuned_model):
    return openai.Completion.create(
        model=fine_tuned_model["id"],
        messages =[{"role": "system", "content": "You are a blockchain expert."},{"role": "user", "content": prompt}],
        max_tokens=60
    )

# Upload the train and test files
train_file = upload_file_to_openai('train.jsonl')
test_file = upload_file_to_openai('test.jsonl')

print(f"Train file upload results: {train_file}")
print(f"Test file upload results: {test_file}")
print(f"Here is the list of all uploaded files: {list_uploaded_files()}")

# Fine tune the model
print(f"Starting finetunning process...")
fine_tuned_model = fine_tune_model(train_file["id"], test_file["id"])

# Check fine tune events
fine_tune_events = check_fine_tune_events(fine_tuned_model["id"])
print(fine_tune_events)

prompt = "Here some questions about the topic"

response = query_fine_tuned_model(prompt=prompt, fine_tuned_model=fine_tuned_model["id"])

print(response['choices'][0]['message'])

