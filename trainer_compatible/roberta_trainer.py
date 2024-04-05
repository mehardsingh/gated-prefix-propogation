from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

from distilbert_prefix import DistilBertForSequenceClassification_Prefix
from bert_prefix import BertForSequenceClassification_Prefix
from roberta_prefix import RobertaForSequenceClassification_Prefix

imdb = load_dataset("imdb")
model_name = "FacebookAI/roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512-2)

tokenized_imdb = imdb.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

# config = AutoConfig.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
config.num_labels = 2
config.prefix_len = 2

# model = DistilBertForSequenceClassification_Prefix(config=config)
model = RobertaForSequenceClassification_Prefix(config=config)

for name, param in model.named_parameters():
    if not name in ["roberta.encoder.prefix", "roberta.pooler.dense.weight", "roberta.pooler.dense.bias", "classifier.dense.weight", "classifier.dense.bias", "classifier.out_proj.weight", "classifier.out_proj.bias"]:
       param.requires_grad = False
    #print(name)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters: {total_params}")

training_args = TrainingArguments(
    output_dir="./my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    use_cpu = True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

