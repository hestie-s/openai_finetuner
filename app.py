import pandas as pd
from sklearn.model_selection import train_test_split
import json
from dotenv import load_dotenv
import os
import openai

load_dotenv()

openai.api_type = "azure"
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = os.getenv("OPEN_API_VERSION")
openai.api_key = os.getenv("OPENAI_API_KEY")

MODEL = "test-gpt-35-turbo"

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
        file_content = f.read()
        return openai.File.create(file=file_content, purpose="fine-tune",user_provided_filename=file_path)

def list_uploaded_files():
    return openai.File.list()


def fine_tune_model(train_file_id, test_file_id, model=MODEL):
    return openai.FineTuningJob.create(
        model=model,
        training_file=train_file_id,
        validation_file=test_file_id,
        # n_epochs=10,
        # learning_rate_multiplier=0.1
    )


def list_fine_tuning_models():
    return openai.FineTuningJob.list()

def check_fine_tune_events(fine_tuned_model_id):
    return openai.FineTuningJob.retrieve(fine_tuned_model_id)


def query_fine_tuned_model(prompt, fine_tuned_model):
    return openai.Completion.create(
        model=fine_tuned_model["id"],
        messages =[{"role": "system", "content": "You are a blockchain expert."},{"role": "user", "content": prompt}],
        max_tokens=60
    )

train_file = upload_file_to_openai('train.jsonl')
test_file = upload_file_to_openai('test.jsonl')

print(f"Train file upload results: {train_file}")
print(f"Test file upload results: {test_file}")
print(f"Here is the list of all uploaded files: {list_uploaded_files()}")

print(f"Starting finetunning process...")
fine_tuned_model = fine_tune_model(train_file["id"], test_file["id"])


fine_tune_events = check_fine_tune_events(fine_tuned_model["id"])
print(f"Fine-tunning events: {fine_tune_events}")

prompt = "Here some questions about the topic"

response = query_fine_tuned_model(prompt=prompt, fine_tuned_model=fine_tuned_model)

print(response['choices'][0]['message'])

