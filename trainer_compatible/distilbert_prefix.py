from transformers.models.distilbert.modeling_distilbert import DistilBertModel, DistilBertForSequenceClassification, Transformer
import torch
from typing import Dict, List, Optional, Set, Tuple, Union
from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer

class Transformer_Prefix(Transformer):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.prefix = torch.nn.Parameter(torch.randn(
            config.n_layers,
            config.prefix_len,
            config.hidden_size
        ))

    def add_curr_prefix(self, layer_idx, curr_hidden_state, attention_mask):
        device = curr_hidden_state.device
        batch_size = curr_hidden_state.shape[0]
        curr_prefix = self.prefix[layer_idx].unsqueeze(0).repeat(batch_size, 1, 1)

        curr_hidden_state = torch.cat((
            curr_prefix,
            curr_hidden_state[:,self.config.prefix_len:,:]
        ), dim=1)

        attention_mask = torch.cat((
            torch.ones(batch_size, self.config.prefix_len).to(device),
            attention_mask[:,self.config.prefix_len:]
        ), dim=1)

        return curr_hidden_state, attention_mask
    
    def add_random_prefix(self, hidden_state, attention_mask):
        hidden_state = torch.cat((
            torch.rand(
                hidden_state.shape[0], 
                self.config.prefix_len, 
                hidden_state.shape[2]
            ).to(hidden_state.device),
            hidden_state
        ), dim=1)

        attention_mask = torch.cat((
            torch.ones(hidden_state.shape[0], self.config.prefix_len).to(hidden_state.device),
            attention_mask
        ), dim=1)

        return hidden_state, attention_mask

        return hidden_state

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: Optional[bool] = None,
    ) -> Union[BaseModelOutput, Tuple[torch.Tensor, ...]]:  # docstyle-ignore
        """
        Parameters:
            x: torch.tensor(bs, seq_length, dim) Input sequence embedded.
            attn_mask: torch.tensor(bs, seq_length) Attention mask on the sequence.

        Returns:
            hidden_state: torch.tensor(bs, seq_length, dim) Sequence of hidden states in the last (top)
            layer all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
                Tuple of length n_layers with the hidden states from each layer.
                Optional: only if output_hidden_states=True
            all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
                Tuple of length n_layers with the attention weights from each layer
                Optional: only if output_attentions=True
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        print(attn_mask.shape)

        hidden_state = x
        hidden_state, attn_mask = self.add_random_prefix(hidden_state, attn_mask)

        print(hidden_state.shape, attn_mask.shape)

        for i, layer_module in enumerate(self.layer):
            hidden_state, attn_mask = self.add_curr_prefix(i, hidden_state, attn_mask)
            print(i, hidden_state.shape)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_state,
                    attn_mask,
                    head_mask[i],
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_state,
                    attn_mask,
                    head_mask[i],
                    output_attentions,
                )

            hidden_state = layer_outputs[-1]

            if output_attentions:
                if len(layer_outputs) != 2:
                    raise ValueError(f"The length of the layer_outputs should be 2, but it is {len(layer_outputs)}")

                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
            else:
                if len(layer_outputs) != 1:
                    raise ValueError(f"The length of the layer_outputs should be 1, but it is {len(layer_outputs)}")

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(v for v in [hidden_state, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_state, hidden_states=all_hidden_states, attentions=all_attentions
        )

class DistilBertModel_Prefix(DistilBertModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = Transformer_Prefix(config)

class DistilBertForSequenceClassification_Prefix(DistilBertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.distilbert = DistilBertModel_Prefix(config)

model_name = "distilbert-base-uncased"
config = AutoConfig.from_pretrained("distilbert-base-uncased")
config.num_labels = 3
config.prefix_len = 2

model = DistilBertForSequenceClassification_Prefix(config=config)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenizer_out = tokenizer(["My name is Vishwa", "Shoulder ice pack thing"], max_length=512-config.prefix_len, padding=True, truncation=True, return_tensors='pt')
input_ids = tokenizer_out["input_ids"]
attention_mask = tokenizer_out["attention_mask"]

# input = torch.rand(8, 4, 768)

output = model(input_ids=input_ids, attention_mask=attention_mask)

print(output.logits.shape)