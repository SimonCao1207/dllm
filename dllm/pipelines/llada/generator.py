"""
reference: https://github.com/ML-GSAI/LLaDA/blob/main/generate.py
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from dllm.utils.generation_utils import get_num_transfer_tokens
from dllm.core.generation.generator import (
    GeneratorOutput,
    GeneratorConfig,
    BaseGenerator,
)


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_transfer_index(
    logits: torch.Tensor,
    x: torch.Tensor,
    mask_index: torch.Tensor,
    num_transfer_tokens: torch.Tensor,
    step_idx: int,
    remasking: str,
    temperature: float,
    restrict_ranges: list[tuple[int, int]] | None = None,
) -> torch.Tensor:
    """
    Compute which masked positions to reveal based on confidence scores.

    Args:
        logits: Model output logits [B, T, vocab_size]
        x: Current sequence [B, T]
        mask_index: Boolean mask of currently masked positions [B, T]
        num_transfer_tokens: Number of tokens to transfer per sample [B, num_steps]
        step_idx: Current step index
        remasking: Strategy for computing confidence ("low_confidence" or "random")
        temperature: Temperature for Gumbel noise
        restrict_ranges: Optional list of (start, end) ranges per sample to restrict selection

    Returns:
        transfer_index: Boolean tensor indicating which positions to update [B, T]
    """
    B, T = x.shape

    # Argmax decoding with optional Gumbel-Max noise
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # [B, T]

    # Compute confidence scores
    if remasking == "low_confidence":
        p = F.softmax(logits, dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == "random":
        x0_p = torch.rand((B, T), device=x.device)
    else:
        raise NotImplementedError(remasking)

    # Restrict selection to valid ranges if provided
    if restrict_ranges is not None:
        for j, (start, end) in enumerate(restrict_ranges):
            if start > 0:
                x0_p[j, :start] = -np.inf
            if end < T:
                x0_p[j, end:] = -np.inf

    # Only consider currently masked positions
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    # Select top-k positions per sample
    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    for j in range(B):
        k = int(num_transfer_tokens[j, step_idx].item())
        if k > 0:
            _, select_index = torch.topk(confidence[j], k=k)
            transfer_index[j, select_index] = True

    return transfer_index


@dataclass
class LLaDAGeneratorConfig(GeneratorConfig):
    max_new_tokens: int = 128
    max_length: int = (
        None  # There's no explicit length_limit except for the tokenizer/model context
    )
    block_length: int = 128
    steps: int = 128
    temperature: float = 0.0
    remasking: str = "low_confidence"
    stochastic_transfer: bool = False
    cfg_scale: float = 0.0
    cfg_keep_tokens: list[int] | None = None


@dataclass
class LLaDAGenerator(BaseGenerator):
    @torch.no_grad()
    def generate(
        self,
        inputs: list[torch.Tensor | list],
        config: LLaDAGeneratorConfig | None = None,
        **kwargs,
    ) -> GeneratorOutput | torch.Tensor:
        if config is None:
            config = LLaDAGeneratorConfig()

        # ----- pull args from config, allow kwargs to override -----
        steps = kwargs.get("steps", config.steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        block_length = kwargs.get("block_length", config.block_length)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        stochastic_transfer = kwargs.get(
            "stochastic_transfer", config.stochastic_transfer
        )
        return_dict_in_generate = kwargs.get(
            "return_dict_in_generate", config.return_dict_in_generate
        )

        assert 1 <= block_length
        assert 1 <= steps
        mask_id = self.tokenizer.mask_token_id
        eos_id = self.tokenizer.eos_token_id

        # ----- Shape bookkeeping: per-sample prompt lengths and final canvas width -----
        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]
        prompt_lens = [p.shape[0] for p in inputs]

        if max_new_tokens:
            max_length = max_new_tokens + max(prompt_lens)
        else:
            max_new_tokens = max_length - max(prompt_lens)

        B = len(inputs)
        T = max_length

        # ----- Initialize canvas with EOS, copy inputs, and append mask tail -----
        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, p in enumerate(inputs):
            x[i, : prompt_lens[i]] = p  # keep original prompt tokens
            x[i, prompt_lens[i] : prompt_lens[i] + max_new_tokens] = (
                mask_id  # append `max_new_tokens` masks to be generated
            )
        attention_mask = (x != eos_id).long() if B > 1 else None

        # Tokens that were *given* at the start (non-mask, non-EOS).
        # These will be masked in the unconditional forward pass for CFG.
        # Tokens from `cfg_keep_tokens` should *not* be treated as "given" for CFG
        unmasked_index = (x != mask_id) & (x != eos_id)
        if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
            keep_mask = torch.isin(
                x, torch.as_tensor(cfg_keep_tokens, device=self.model.device)
            )
            unmasked_index = unmasked_index & ~keep_mask

        # ----- Block scheduling over the appended mask tail -----
        num_blocks = math.ceil(max_new_tokens / block_length)
        steps = math.ceil(steps / num_blocks)  # per-block step budget
        histories = [x.clone()] if return_dict_in_generate else None

        for b in range(num_blocks):
            # Build a per-sample mask *within this block* (aligned to each prompt's tail)
            block_mask_index = torch.zeros(
                (B, block_length), dtype=torch.bool, device=x.device
            )

            for j in range(B):
                start = prompt_lens[j] + b * block_length
                end = min(start + block_length, prompt_lens[j] + max_new_tokens, T)
                if start < end:
                    width = end - start
                    block_mask_index[j, :width] = (
                        x[j, start:end] == mask_id
                    )  # which positions in this block are still masked

            # Decide how many tokens to reveal per step in this block
            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )

            # Some steps may be skipped if there are no transfers
            effective_steps = num_transfer_tokens.size(1)

            # ----- Iterative reveal inside the current block -----
            for i in range(effective_steps):
                mask_index = x == mask_id  # current global mask map

                # Optional CFG: second forward where original prompt tokens are masked out
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(
                        x_, attention_mask=attention_mask
                    ).logits  # Use attention mask here
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(
                        x, attention_mask=attention_mask
                    ).logits  # Use attention mask here

                # Compute transfer indices using helper function
                restrict_ranges = [
                    (0, prompt_lens[j] + (b + 1) * block_length) for j in range(B)
                ]
                transfer_index = get_transfer_index(
                    logits=logits,
                    x=x,
                    mask_index=mask_index,
                    num_transfer_tokens=num_transfer_tokens,
                    step_idx=i,
                    remasking=remasking,
                    temperature=temperature,
                    restrict_ranges=restrict_ranges,
                )

                # Commit chosen predictions into the canvas
                x[transfer_index] = torch.argmax(
                    add_gumbel_noise(logits, temperature=temperature), dim=-1
                )[transfer_index]
                if histories is not None:
                    histories.append(x.clone())

        # ----- Output format -----
        if not return_dict_in_generate:
            return x
        else:
            return GeneratorOutput(sequences=x, histories=histories)

    @torch.no_grad()
    def generate_with_prefix_cache(
        self,
        inputs: list[torch.Tensor | list],
        config: LLaDAGeneratorConfig | None = None,
        **kwargs,
    ) -> GeneratorOutput | torch.Tensor:
        # ----- pull args from config, allow kwargs to override -----
        steps = kwargs.get("steps", config.steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        block_length = kwargs.get("block_length", config.block_length)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        stochastic_transfer = kwargs.get(
            "stochastic_transfer", config.stochastic_transfer
        )
        return_dict_in_generate = kwargs.get(
            "return_dict_in_generate", config.return_dict_in_generate
        )

        assert 1 <= block_length
        assert 1 <= steps
        mask_id = self.tokenizer.mask_token_id
        eos_id = self.tokenizer.eos_token_id

        # ----- Shape bookkeeping: per-sample prompt lengths and final canvas width -----
        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]
        prompt_lens = [p.shape[0] for p in inputs]

        if max_new_tokens:
            max_length = max_new_tokens + max(prompt_lens)
        else:
            max_new_tokens = max_length - max(prompt_lens)

        B = len(inputs)
        T = max_length

        # ----- Initialize canvas with EOS, copy inputs, and append mask tail -----
        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, p in enumerate(inputs):
            x[i, : prompt_lens[i]] = p  # keep original prompt tokens
            x[i, prompt_lens[i] : prompt_lens[i] + max_new_tokens] = (
                mask_id  # append `max_new_tokens` masks to be generated
            )
        attention_mask = (x != eos_id).long() if B > 1 else None

        # Tokens that were *given* at the start (non-mask, non-EOS).
        # These will be masked in the unconditional forward pass for CFG.
        # Tokens from `cfg_keep_tokens` should *not* be treated as "given" for CFG
        unmasked_index = (x != mask_id) & (x != eos_id)
        if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
            keep_mask = torch.isin(
                x, torch.as_tensor(cfg_keep_tokens, device=self.model.device)
            )
            unmasked_index = unmasked_index & ~keep_mask

        # ----- Block scheduling over the appended mask tail -----
        num_blocks = math.ceil(max_new_tokens / block_length)
        steps = math.ceil(steps / num_blocks)  # per-block step budget
        histories = [x.clone()] if return_dict_in_generate else None

        for b in range(num_blocks):
            # Build a per-sample mask *within this block* (aligned to each prompt's tail)
            block_mask_index = torch.zeros(
                (B, block_length), dtype=torch.bool, device=x.device
            )
            for j in range(B):
                start = prompt_lens[j] + b * block_length
                end = min(start + block_length, prompt_lens[j] + max_new_tokens, T)
                if start < end:
                    width = end - start
                    block_mask_index[j, :width] = (
                        x[j, start:end] == mask_id
                    )  # which positions in this block are still masked

            # Decide how many tokens to reveal per step in this block
            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )

            effective_steps = num_transfer_tokens.size(1)

            # ----- First forward pass: full sequence to get KV cache -----
            # Compute prefix cache boundary (max prompt length for this block)
            max_prompt_len = max(prompt_lens)
            current_block_start = max_prompt_len + b * block_length

            # Initial forward pass with cache
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[unmasked_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                output = self.model(x_, attention_mask=attention_mask, use_cache=True)
                logits = output.logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                past_key_values = output.past_key_values
            else:
                output = self.model(x, attention_mask=attention_mask, use_cache=True)
                logits = output.logits
                past_key_values = output.past_key_values

            # First step: process full sequence
            mask_index = x == mask_id

            restrict_ranges = [
                (0, prompt_lens[j] + (b + 1) * block_length) for j in range(B)
            ]
            transfer_index = get_transfer_index(
                logits=logits,
                x=x,
                mask_index=mask_index,
                num_transfer_tokens=num_transfer_tokens,
                step_idx=0,
                remasking=remasking,
                temperature=temperature,
                restrict_ranges=restrict_ranges,
            )

            x[transfer_index] = torch.argmax(
                add_gumbel_noise(logits, temperature=temperature), dim=-1
            )[transfer_index]
            if histories is not None:
                histories.append(x.clone())

            # Truncate past_key_values to prefix (before current block)
            new_past_key_values = []
            for layer_past in past_key_values:
                new_layer_past = ()
                for kv in layer_past:
                    # Keep only up to current_block_start
                    new_layer_past += (kv[:, :, :current_block_start],)
                new_past_key_values.append(new_layer_past)
            past_key_values = tuple(new_past_key_values)

            # ----- Subsequent steps: only process current block with cache -----
            for i in range(1, effective_steps):
                mask_index = x == mask_id

                # Only forward the current block region
                block_x = x[:, current_block_start:]

                if cfg_scale > 0.0:
                    un_block_x = block_x.clone()
                    un_block_x_mask = unmasked_index[:, current_block_start:]
                    un_block_x[un_block_x_mask] = mask_id
                    block_x_ = torch.cat([block_x, un_block_x], dim=0)

                    # Duplicate past_key_values for CFG
                    cfg_past_key_values = []
                    for layer_past in past_key_values:
                        cfg_layer_past = ()
                        for kv in layer_past:
                            cfg_layer_past += (torch.cat([kv, kv], dim=0),)
                        cfg_past_key_values.append(cfg_layer_past)

                    output = self.model(
                        block_x_,
                        past_key_values=tuple(cfg_past_key_values),
                        use_cache=True
                    )
                    logits = output.logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    output = self.model(
                        block_x,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                    logits = output.logits

                # Map confidence computation to full sequence coordinates
                # Restrict to current block width
                block_restrict_ranges = []
                for j in range(B):
                    block_end = prompt_lens[j] + (b + 1) * block_length - current_block_start
                    block_restrict_ranges.append((0, min(block_end, logits.shape[1])))

                # Create full-sized tensors for helper function
                full_logits = torch.full(
                    (B, T, logits.shape[-1]), -np.inf, device=logits.device, dtype=logits.dtype
                )
                full_logits[:, current_block_start:current_block_start + logits.shape[1]] = logits

                transfer_index_full = get_transfer_index(
                    logits=full_logits,
                    x=x,
                    mask_index=mask_index,
                    num_transfer_tokens=num_transfer_tokens,
                    step_idx=i,
                    remasking=remasking,
                    temperature=temperature,
                    restrict_ranges=[(current_block_start, current_block_start + r[1]) for r in block_restrict_ranges],
                )

                x[transfer_index_full] = torch.argmax(
                    add_gumbel_noise(full_logits, temperature=temperature), dim=-1
                )[transfer_index_full]
                if histories is not None:
                    histories.append(x.clone())

        # ----- Output format -----
        if not return_dict_in_generate:
            return x
        else:
            return GeneratorOutput(sequences=x, histories=histories)

    @torch.no_grad()
    def infill(
        self, inputs: list[torch.Tensor | list], config, **kwargs
    ) -> GeneratorOutput | torch.Tensor:
        """
        Fill in-place the <|mdm_mask|> tokens contained in `inputs`.
        The whole (padded) sequence is split into block windows of length
        `block_length`; within each window we progressively "unmask" positions
        according to the scheduler and chosen remasking strategy.

        Notes:
        - Right padding uses EOS.
        - CFG masks out *originally known* (non-mask, non-EOS) tokens in the
        unconditional branch, identical to `generate`.
        - Only masked positions are ever updated; non-mask tokens are left intact.
        """
        # ----- pull args from config, allow kwargs to override -----
        steps = kwargs.get("steps", config.steps)
        block_length = kwargs.get("block_length", config.block_length)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        stochastic_transfer = kwargs.get(
            "stochastic_transfer", config.stochastic_transfer
        )
        return_dict_in_generate = kwargs.get(
            "return_dict_in_generate", config.return_dict_in_generate
        )

        mask_id = self.tokenizer.mask_token_id
        eos_id = self.tokenizer.eos_token_id

        # ----- Build canvas: right-pad with EOS to the max length in the batch -----
        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]

        B = len(inputs)
        seq_lens = [t.shape[0] for t in inputs]
        T = max(seq_lens)

        # Default to a single block spanning the whole sequence
        if block_length is None:
            block_length = T

        assert 1 <= block_length
        assert 1 <= steps

        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, t in enumerate(inputs):
            x[i, : seq_lens[i]] = t
        attention_mask = (x != eos_id).long() if B > 1 else None

        # Tokens that were *given* at the start (non-mask, non-EOS).
        # These will be masked in the unconditional forward pass for CFG.
        # Tokens from `cfg_keep_tokens` should *not* be treated as "given" for CFG
        unmasked_index = (x != mask_id) & (x != eos_id)
        if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
            keep_mask = torch.isin(
                x, torch.as_tensor(cfg_keep_tokens, device=self.model.device)
            )
            unmasked_index = unmasked_index & ~keep_mask

        # ----- Blockwise schedule over the *entire* (padded) sequence -----
        num_blocks = math.ceil(T / block_length)
        steps_per_block = math.ceil(steps / num_blocks)
        histories = [x.clone()] if return_dict_in_generate else None

        # Create attention mask where eos_token_id is masked (set to 0)
        attention_mask = (x != eos_id).long()

        for b in range(num_blocks):
            start = b * block_length
            stop = min(start + block_length, T)

            # Per-sample view of which positions in this block are masks
            block_mask_index = torch.zeros(
                (B, block_length), dtype=torch.bool, device=self.model.device
            )
            widths = []
            for j in range(B):
                # Width limited by sample's true length and sequence end
                width = max(0, min(seq_lens[j], stop) - start)
                widths.append(width)
                if width > 0:
                    block_mask_index[j, :width] = x[j, start : start + width] == mask_id

            # Decide how many tokens to reveal at each step in this block
            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps_per_block,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )

            # Some blocks may have no masks => effective_steps == 0
            effective_steps = num_transfer_tokens.size(1)

            for s in range(effective_steps):
                mask_index_full = x == mask_id

                # ----- Forward pass (+ optional CFG) -----
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(
                        x_, attention_mask=attention_mask
                    ).logits  # Use attention mask here
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(
                        x, attention_mask=attention_mask
                    ).logits  # Use attention mask here

                # Greedy with optional Gumbel-Max noise
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)  # [B, T]

                # Confidence used for choosing which masks to commit this step
                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(
                        -1
                    )  # [B, T]
                elif remasking == "random":
                    x0_p = torch.rand((B, T), device=self.model.device)
                else:
                    raise NotImplementedError(remasking)

                # Restrict selection to the *current* block only
                for j in range(B):
                    end_j = start + widths[j]
                    # Outside current block => impossible to select
                    x0_p[j, :start] = -np.inf
                    x0_p[j, end_j:] = -np.inf

                # Only consider currently-masked positions as candidates
                x0 = torch.where(mask_index_full, x0, x)
                confidence = torch.where(mask_index_full, x0_p, -np.inf)

                # Pick exactly num_transfer_tokens[j, s] positions per sample
                transfer_index = torch.zeros_like(x, dtype=torch.bool)
                for j in range(B):
                    k = int(num_transfer_tokens[j, s].item())
                    if k > 0:
                        _, select_idx = torch.topk(confidence[j], k=k)
                        transfer_index[j, select_idx] = True

                # Commit selected predictions into the canvas
                x[transfer_index] = x0[transfer_index]
                if histories is not None:
                    histories.append(x.clone())

        # ----- Output format -----
        if not return_dict_in_generate:
            return x
        else:
            return GeneratorOutput(sequences=x, histories=histories)
