from contextlib import nullcontext

import torch
from transformers import Gemma3ForConditionalGeneration

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class EncodeGemma3Text(
    PipelineModule,
    RandomAccessPipelineModule,
):
    """Encode text with Gemma3 and stack ALL hidden states.

    LTX-2.3 uses Gemma3-12B as its text encoder. Unlike standard single-layer
    encoding, it stacks every hidden state (all layers + embedding) and flattens
    the layer dimension into the channel dimension:

        (seq_len, hidden_size, num_layers) → (seq_len, hidden_size * num_layers)

    This matches ``Ltx2Model.encode_text()`` and ai-toolkit's ``get_prompt_embeds()``.
    The resulting tensor is what ``LTX2TextConnectors`` expects as input.
    """

    def __init__(
            self,
            tokens_in_name: str,
            tokens_attention_mask_in_name: str | None,
            hidden_state_out_name: str,
            text_encoder: Gemma3ForConditionalGeneration,
            autocast_contexts: list[torch.autocast | None] = None,
            dtype: torch.dtype | None = None,
    ):
        super(EncodeGemma3Text, self).__init__()
        self.tokens_in_name = tokens_in_name
        self.tokens_attention_mask_in_name = tokens_attention_mask_in_name
        self.hidden_state_out_name = hidden_state_out_name
        self.text_encoder = text_encoder

        self.autocast_contexts = [nullcontext()] if autocast_contexts is None else autocast_contexts
        self.dtype = dtype

    def length(self) -> int:
        return self._get_previous_length(self.tokens_in_name)

    def get_inputs(self) -> list[str]:
        return [self.tokens_in_name, self.tokens_attention_mask_in_name]

    def get_outputs(self) -> list[str]:
        return [self.hidden_state_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        tokens = self._get_previous_item(variation, self.tokens_in_name, index)
        tokens = tokens.unsqueeze(0)  # (1, seq_len)

        if self.tokens_attention_mask_in_name is not None:
            tokens_attention_mask = self._get_previous_item(
                variation, self.tokens_attention_mask_in_name, index,
            )
            tokens_attention_mask = tokens_attention_mask.unsqueeze(0)  # (1, seq_len)
        else:
            tokens_attention_mask = None

        with self._all_contexts(self.autocast_contexts):
            text_encoder_output = self.text_encoder(
                input_ids=tokens,
                attention_mask=tokens_attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )

        # Stack all hidden states: tuple of (1, seq_len, hidden_size) tensors
        # → (1, seq_len, hidden_size, num_layers) → (1, seq_len, hidden_size * num_layers)
        hidden_states = torch.stack(text_encoder_output.hidden_states, dim=-1)
        embeddings = hidden_states.flatten(2, 3)  # (1, seq_len, hidden_size * num_layers)
        embeddings = embeddings.squeeze(0)  # (seq_len, hidden_size * num_layers)

        if self.dtype is not None:
            embeddings = embeddings.to(dtype=self.dtype)

        return {
            self.hidden_state_out_name: embeddings,
        }
