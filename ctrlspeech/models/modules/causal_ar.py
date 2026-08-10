import torch
import torch.nn as nn

from ..backbone.qwen3 import Qwen3Config, Qwen3ModelForMultiModal, Qwen3ForCausalLM


class CausalAR(nn.Module):
    def __init__(
        self,
        qwen_config_path,
        pretrained_LM_path,
        load_pretrained_weights,
        load_phoneme=False,
        weighted_layers=False,
    ):
        super().__init__()
        config = Qwen3Config.from_pretrained(qwen_config_path)
        if load_phoneme:
            config.vocab_size = 100
        
        self.model = Qwen3ModelForMultiModal(config)
        if load_pretrained_weights:
            print(f"Loading pre-trained weights from '{pretrained_LM_path}'...")
            pretrained_model = Qwen3ForCausalLM.from_pretrained(pretrained_LM_path)
            state_dict = pretrained_model.model.state_dict()
            if load_phoneme:
                state_dict.pop("embed_tokens.weight")

            loading_info = self.model.load_state_dict(state_dict, strict=False)
            print(loading_info)
            print("✅ Pre-trained weights loaded successfully into custom model.")            
        
        self.weighted_layers = weighted_layers
        if weighted_layers:
            num_layers = config.num_hidden_layers
             # self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
            self.acoustic_layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
            self.semantic_layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
    
    def forward(
        self,
        input_embeds: torch.Tensor,
        modality_type_ids: torch.Tensor,
        padding_mask=None,
    ) -> torch.Tensor:
        outputs = self.model(
            inputs_embeds=input_embeds, 
            modality_type_ids=modality_type_ids,
            attention_mask=padding_mask, 
            output_hidden_states=True,
        )
        
        if self.weighted_layers:
            all_hidden_states = outputs.hidden_states  #(, , ,) 28层 层数，每层#(B, L, D)
            # normalized_weights = torch.softmax(self.layer_weights, dim=0)
            normalized_acoustic_weights = torch.softmax(self.acoustic_layer_weights, dim=0)
            normalized_semantic_weights = torch.softmax(self.semantic_layer_weights, dim=0)
            # weighted_sum = sum(weight * hidden_state for weight, hidden_state in zip(normalized_weights, all_hidden_states))
            acoustic_weighted_sum = sum(weight * hidden_state for weight, hidden_state in zip(normalized_acoustic_weights, all_hidden_states)) #(B, L, D)
            semantic_weighted_sum = sum(weight * hidden_state for weight, hidden_state in zip(normalized_semantic_weights, all_hidden_states))
            # return weighted_sum
            return semantic_weighted_sum, acoustic_weighted_sum
        else:
            last_hidden_state = outputs.last_hidden_state
            return last_hidden_state 

    def inference(
        self,
        input_embeds: torch.Tensor,
        modality_type_ids: torch.Tensor,
        padding_mask = None,
        past_key_values = None,
        use_cache = None,
    ) -> torch.Tensor:

        outputs = self.model(
            inputs_embeds = input_embeds,
            modality_type_ids = modality_type_ids,
            attention_mask = padding_mask,
            output_hidden_states=True,
            past_key_values = past_key_values,
            use_cache = use_cache,
        )

        if self.weighted_layers:
            all_hidden_states = outputs.hidden_states
            # normalized_weights = torch.softmax(self.layer_weights, dim=0)
            normalized_acoustic_weights = torch.softmax(self.acoustic_layer_weights, dim=0)
            normalized_semantic_weights = torch.softmax(self.semantic_layer_weights, dim=0)
            # weighted_sum = sum(weight * hidden_state for weight, hidden_state in zip(normalized_weights, all_hidden_states))
            acoustic_weighted_sum = sum(weight * hidden_state for weight, hidden_state in zip(normalized_acoustic_weights, all_hidden_states))
            semantic_weighted_sum = sum(weight * hidden_state for weight, hidden_state in zip(normalized_semantic_weights, all_hidden_states))
            # return weighted_sum
            past_key_values = outputs.past_key_values
            return semantic_weighted_sum, acoustic_weighted_sum, past_key_values
        else:
            last_hidden_state = outputs.last_hidden_state
            past_key_values = outputs.past_key_values
            return last_hidden_state, past_key_values