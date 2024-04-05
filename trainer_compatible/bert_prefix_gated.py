from transformers.models.bert.modeling_bert import BertModel, BertForSequenceClassification, BertEncoder
import torch
from typing import Dict, List, Optional, Set, Tuple, Union
from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    BaseModelOutputWithPastAndCrossAttentions
)
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer

class BertEncoder_Prefix_Gated(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.prefix = torch.nn.Parameter(torch.randn(
            config.num_hidden_layers,
            config.prefix_len,
            config.hidden_size
        ))

        self.gate_mlps = torch.nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.gate_mlps.append(
                torch.nn.Sequential(
                    torch.nn.Linear(config.hidden_size, config.hidden_size//2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(config.hidden_size//2, config.hidden_size//4),
                    torch.nn.ReLU(),
                    torch.nn.Linear(config.hidden_size//4, 1),
                    torch.nn.Sigmoid()
                )
            )

    def add_curr_prefix(self, layer_idx, curr_hidden_state):
        device = curr_hidden_state.device
        batch_size = curr_hidden_state.shape[0]

        existing_prefix = curr_hidden_state[:,:self.config.prefix_len,:] # b x L x h
        next_prefix = self.prefix[layer_idx].unsqueeze(0).repeat(batch_size, 1, 1) # b x L x H
        concat_prefix = torch.cat([existing_prefix, next_prefix], dim=1) # b x 2L x H
        pooled_prefix = torch.mean(concat_prefix, dim=1) # b x h
        gate_mlp = self.gate_mlps[layer_idx]
        gate = gate_mlp(pooled_prefix).unsqueeze(-1) # b x 1

        new_prefix = (next_prefix * gate) + (existing_prefix * (1 - gate))

        curr_hidden_state = torch.cat((
            new_prefix,
            curr_hidden_state[:,self.config.prefix_len:,:]
        ), dim=1)

        return curr_hidden_state
    
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
            torch.ones(attention_mask.shape[0], attention_mask.shape[1], attention_mask.shape[2], self.config.prefix_len).to(hidden_state.device),
            attention_mask
        ), dim=-1)

        return hidden_state, attention_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                # logger.warning_once(
                #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                # )
                use_cache = False


        hidden_states, attention_mask = self.add_random_prefix(hidden_states, attention_mask)

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            hidden_states = self.add_curr_prefix(i, hidden_states)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class BertModel_Prefix_Gated(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = BertEncoder_Prefix_Gated(config)

class BertForSequenceClassification_Prefix_Gated(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel_Prefix_Gated(config)

# model_name = "google-bert/bert-base-uncased"
# config = AutoConfig.from_pretrained(model_name)
# config.num_labels = 3
# config.prefix_len = 2

# model = BertForSequenceClassification_Prefix_Gated(config=config)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer_out = tokenizer(["My name is Vishwa", "Shoulder ice pack thing"], max_length=512-config.prefix_len, padding=True, truncation=True, return_tensors='pt')
# input_ids = tokenizer_out["input_ids"]
# attention_mask = tokenizer_out["attention_mask"]

# # input = torch.rand(8, 4, 768)

# output = model(input_ids=input_ids, attention_mask=attention_mask)

# print(output.logits.shape)
        
# =============================
# N layers
# N MLPs mapping h -> 1
# input (b x L x h)
# avgpool (b x h) --> (bx1) = gate
# new prefix = p_i * gate + p_i-1 * (1 - gate)