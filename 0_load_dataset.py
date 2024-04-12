from datasets import load_dataset

dataset = load_dataset("lang-uk/recruitment-dataset-job-descriptions-english")

print(len(dataset['train'][3]))
