import torch
import torch.nn as nn
from transformers import AutoConfig, DistilBertModel, BertModel, BertTokenizer, AutoModelForSequenceClassification, AutoTokenizer, DistilBertPreTrainedModel
import torch.nn.functional as F

class PrefixTunedDistilbert(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        self.num_layers = len(self.distilbert.transformer.layer)
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

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings
        """
        return self.distilbert.get_position_embeddings()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end. If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the
                size will add correct vectors at the end following the position encoding algorithm, whereas reducing
                the size will remove vectors from the end.
        """
        self.distilbert.resize_position_embeddings(new_num_position_embeddings)

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
        curr_hidden_state = self.distilbert.embeddings(input_ids)
        for i in range(len(self.distilbert.transformer.layer)):
            layer = self.distilbert.transformer.layer[i]
            curr_hidden_state, attention_mask = self.add_curr_prefix(
                i, curr_hidden_state, attention_mask, i!=0
            )
            outputs = layer(curr_hidden_state, attention_mask)
            curr_hidden_state = outputs[0]
        
        pooled_output = curr_hidden_state[:,0,:]
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = self.dropout(F.relu(pooled_output))
        logits = self.classifier(pooled_output)

        return logits

model_name = "distilbert/distilbert-base-uncased"
num_labels = 5
prefix_len = 5

config = AutoConfig.from_pretrained(model_name)
config.num_labels = num_labels
config.prefix_len = prefix_len

model = PrefixTunedDistilbert(config)

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
