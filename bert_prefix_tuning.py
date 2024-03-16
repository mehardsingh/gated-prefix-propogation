import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F

class PrefixTunedBERT(nn.Module):
    def __init__(self, bert):
        super(PrefixTunedBERT, self).__init__()
        self.backbone = bert
        
    def forward(self, input_ids, attention_mask):
        curr_hidden_state = self.backbone.distilbert.embeddings(input_ids)

        for i in range(len(self.backbone.distilbert.transformer.layer)):
            print(f"Layer {i}")
            layer = self.backbone.distilbert.transformer.layer[i]
            outputs = layer(curr_hidden_state, attention_mask)
            curr_hidden_state = outputs[0]
        
        pooled_output = curr_hidden_state[:,0,:]
        
        pooled_output = self.backbone.pre_classifier(pooled_output)
        pooled_output = self.backbone.dropout(F.relu(pooled_output))
        logits = self.backbone.classifier(pooled_output)

        return logits

model_name = "distilbert/distilbert-base-uncased"
num_labels = 5

bert = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
model = PrefixTunedBERT(bert=bert)

tokenizer = AutoTokenizer.from_pretrained(model_name)
text = ["The movie is bad", "This was the greatest movie I have ever seen"]

tokenizer_out = tokenizer(text,  padding=True, truncation=True, return_tensors='pt')
input_ids = tokenizer_out["input_ids"]
attention_mask = tokenizer_out["attention_mask"]

print("input_ids", input_ids.shape)
print("attention_mask", attention_mask.shape)

first = tokenizer.decode(input_ids[0])
print(first)
second = tokenizer.decode(input_ids[1])
print(second)

output = model(input_ids, attention_mask)
print(output.shape)


# [CLS] The movie is bad
