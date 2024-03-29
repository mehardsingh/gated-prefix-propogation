import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AutoModelForSequenceClassification, AutoTokenizer, BertPreTrainedModel, AutoModel, AutoConfig
import torch.nn.functional as F

class PrefixTunedBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        print(self.bert)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.num_layers = len(self.bert.encoder.layer)
        self.hidden_size = self.config.hidden_size
        self.prefix_len = self.config.prefix_len
        self.prefix = torch.nn.Parameter(torch.randn(
            self.num_layers,
            self.prefix_len,
            self.hidden_size,
        ))
        self.prefix.requires_grad = True
        self.max_seq_len = self.config.max_position_embeddings

        # Initialize weights and apply final processing
        self.post_init()

    def add_curr_prefix(self, layer_idx, curr_hidden_state, attention_mask, contains_prefix=False):
        device = curr_hidden_state.device
        batch_size = curr_hidden_state.shape[0]
        curr_prefix = self.prefix[layer_idx].unsqueeze(0).repeat(batch_size, 1, 1)
        normal_token_start_idx = 1 if not contains_prefix else 1 + self.prefix_len # [CLS] [... prefix ...] [... normal ...] [... PAD ...]

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
        input_shape = input_ids.size()

        curr_hidden_state = self.bert.embeddings(input_ids)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        for i in range(len(self.bert.encoder.layer)):
            layer = self.bert.encoder.layer[i]
            curr_hidden_state, attention_mask = self.add_curr_prefix(
                i, curr_hidden_state, attention_mask, i!=0
            )
            extended_attention_mask = self.get_extended_attention_mask(attention_mask, (curr_hidden_state.shape[0], curr_hidden_state.shape[1]))
            outputs = layer(curr_hidden_state, extended_attention_mask)
            curr_hidden_state = outputs[0]
        
        pooled_output = curr_hidden_state[:,0,:]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits

model_name = "google-bert/bert-base-uncased"
num_labels = 5
prefix_len = 5

config = AutoConfig.from_pretrained(model_name)
config.num_labels = num_labels
config.prefix_len = prefix_len

model = PrefixTunedBert(config=config)

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
