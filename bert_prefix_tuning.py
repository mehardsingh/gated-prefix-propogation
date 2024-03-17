import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F

class PrefixTunedBERT(nn.Module):
    def __init__(self, bert, prefix_len):
        super(PrefixTunedBERT, self).__init__()
        self.backbone = bert
        self.prefix_len = prefix_len

        self.num_layers = len(self.backbone.distilbert.transformer.layer)
        self.hidden_size = self.backbone.config.hidden_size
        self.prefix = torch.nn.Parameter(torch.randn(
            self.num_layers,
            self.prefix_len,
            self.hidden_size,
        ))
        self.prefix.requires_grad = True
        self.max_seq_len = self.backbone.config.max_position_embeddings

    def add_curr_prefix(self, layer_idx, curr_hidden_state, attention_mask, contains_prefix=False):
        device = curr_hidden_state.device
        batch_size = curr_hidden_state.shape[0]
        curr_prefix = self.prefix[layer_idx].unsqueeze(0).repeat(batch_size, 1, 1)
        normal_token_start_idx = 1 if not contains_prefix else 1 + self.prefix_len

        curr_hidden_state = torch.cat((
            curr_hidden_state[:,0,:].unsqueeze(1),
            curr_prefix,
            curr_hidden_state[:,normal_token_start_idx:,:]
        ), dim=1)

        attention_mask = torch.cat((
            attention_mask[:,0].unsqueeze(1),
            torch.ones(batch_size, self.prefix_len).to(device),
            attention_mask[:,normal_token_start_idx:]
        ), dim=1)

        return curr_hidden_state, attention_mask
        
    def forward(self, input_ids, attention_mask):
        curr_hidden_state = self.backbone.distilbert.embeddings(input_ids)
        for i in range(self.num_layers):
            layer = self.backbone.distilbert.transformer.layer[i]
            curr_hidden_state, attention_mask = self.add_curr_prefix(
                i, curr_hidden_state, attention_mask, i!=0
            )
            outputs = layer(curr_hidden_state, attention_mask)
            curr_hidden_state = outputs[0]
        
        pooled_output = curr_hidden_state[:,0,:]
        pooled_output = self.backbone.pre_classifier(pooled_output)
        pooled_output = self.backbone.dropout(F.relu(pooled_output))
        logits = self.backbone.classifier(pooled_output)

        return logits

model_name = "distilbert/distilbert-base-uncased"
num_labels = 5
prefix_len = 5

bert = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
model = PrefixTunedBERT(bert=bert, prefix_len=prefix_len)

tokenizer = AutoTokenizer.from_pretrained(model_name)
text = ["The movie is bad", "This was the greatest movie I have ever seen"]

tokenizer_out = tokenizer(text,  max_length=512-prefix_len, padding=True, truncation=True, return_tensors='pt')
input_ids = tokenizer_out["input_ids"]
attention_mask = tokenizer_out["attention_mask"]

print("input_ids", input_ids.shape)
print("attention_mask", attention_mask.shape)
print(attention_mask)

first = tokenizer.decode(input_ids[0])
print(first)
second = tokenizer.decode(input_ids[1])
print(second)

output = model(input_ids, attention_mask)
print(output.shape)
