#!/usr/bin/env python3
"""Base pruning interface for CodeBreaker safety experiments."""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import re
from typing import Any, Callable, Iterator, Iterable

import torch
import torch.nn.functional as F


class BaseSafetyPruner(ABC):
    def __init__(self, model: torch.nn.Module, device: str | torch.device | None = None):
        self.model = model
        self.device = torch.device(device) if device is not None else self._infer_device()

        self._activation_hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._activation_sums: dict[tuple[str, int], torch.Tensor] = {}
        self._activation_counts: dict[tuple[str, int], int] = {}

        self._hidden_state_hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._last_hidden_states: dict[int, torch.Tensor] = {}

        self._grad_component_hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._grad_component_sums: dict[tuple[str, int], torch.Tensor] = {}
        self._grad_component_counts: dict[tuple[str, int], int] = {}

        self._grad_buffer_clean: dict[str, torch.Tensor] | None = None
        self._grad_buffer_trigger: dict[str, torch.Tensor] | None = None

    def _infer_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _move_to_device(self, batch: Any) -> Any:
        if torch.is_tensor(batch):
            return batch.to(self.device)
        if isinstance(batch, Mapping):
            return {key: self._move_to_device(value) for key, value in batch.items()}
        if isinstance(batch, tuple):
            return tuple(self._move_to_device(value) for value in batch)
        if isinstance(batch, list):
            return [self._move_to_device(value) for value in batch]
        return batch

    def _resolve_decoder_layers(self) -> list[torch.nn.Module]:
        """Best-effort resolve of decoder layers for LLaMA-like HF models."""
        # NOTE: PEFT wrappers often nest the underlying HF model under attributes like
        # `base_model` and/or `model`. We try a few common paths.
        candidates: list[torch.nn.Module] = []
        candidates.append(self.model)

        m = getattr(self.model, "model", None)
        if isinstance(m, torch.nn.Module):
            candidates.append(m)

        bm = getattr(self.model, "base_model", None)
        if isinstance(bm, torch.nn.Module):
            candidates.append(bm)
            bm_m = getattr(bm, "model", None)
            if isinstance(bm_m, torch.nn.Module):
                candidates.append(bm_m)

        for cand in candidates:
            if hasattr(cand, "model") and hasattr(getattr(cand, "model"), "layers"):
                return list(cand.model.layers)
            if hasattr(cand, "layers"):
                return list(cand.layers)

        raise ValueError(
            "Unable to resolve decoder layers; expected model.model.layers or model.layers (including PEFT wrappers)"
        )

    def _remove_hidden_state_hooks(self) -> None:
        while self._hidden_state_hooks:
            handle = self._hidden_state_hooks.pop()
            try:
                handle.remove()
            except Exception:
                pass

    def _remove_grad_component_hooks(self) -> None:
        while self._grad_component_hooks:
            handle = self._grad_component_hooks.pop()
            try:
                handle.remove()
            except Exception:
                pass

    def _flatten_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim < 2:
            raise ValueError("hidden_states must have at least 2 dimensions")
        return hidden_states.reshape(hidden_states.size(0), -1)

    def _extract_loss(self, batch: Mapping[str, Any], loss_fn: Any | None) -> torch.Tensor:
        outputs = self.model(**batch)
        loss = getattr(outputs, "loss", None)
        if loss is not None:
            return loss

        if loss_fn is None:
            raise ValueError("loss_fn is required when the model output does not expose a loss field")

        logits = getattr(outputs, "logits", None)
        if logits is None:
            if isinstance(outputs, Mapping):
                logits = outputs.get("logits")
            elif isinstance(outputs, Sequence):
                logits = outputs[0]

        labels = batch.get("labels")
        if logits is None or labels is None:
            raise ValueError("Unable to compute loss: missing logits or labels")
        return loss_fn(logits, labels)

    def _collect_named_gradients(self) -> dict[str, torch.Tensor]:
        gradients: dict[str, torch.Tensor] = {}
        for name, parameter in self.model.named_parameters():
            if parameter.grad is not None:
                gradients[name] = parameter.grad.detach().clone()
        return gradients

    def prepare_gradient_buffer(
        self,
        *,
        only_trainable: bool = True,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> dict[str, Any]:
        """Allocate GPU buffers to hold both clean and trigger gradients simultaneously.

        Returns metadata including total bytes allocated.
        """

        target_device = torch.device(device) if device is not None else self.device
        clean: dict[str, torch.Tensor] = {}
        trigger: dict[str, torch.Tensor] = {}
        total_bytes = 0

        for name, parameter in self.model.named_parameters():
            if only_trainable and not parameter.requires_grad:
                continue

            buffer_dtype = dtype if dtype is not None else parameter.dtype
            clean_tensor = torch.zeros(parameter.shape, device=target_device, dtype=buffer_dtype)
            trigger_tensor = torch.zeros(parameter.shape, device=target_device, dtype=buffer_dtype)
            clean[name] = clean_tensor
            trigger[name] = trigger_tensor
            total_bytes += clean_tensor.numel() * clean_tensor.element_size()
            total_bytes += trigger_tensor.numel() * trigger_tensor.element_size()

        self._grad_buffer_clean = clean
        self._grad_buffer_trigger = trigger
        return {
            "device": str(target_device),
            "dtype": str(dtype) if dtype is not None else "param_dtype",
            "param_count": len(clean),
            "bytes": total_bytes,
        }

    @staticmethod
    def _gradient_cosine(
        clean_gradients: Mapping[str, torch.Tensor],
        trigger_gradients: Mapping[str, torch.Tensor],
        eps: float = 1e-12,
    ) -> torch.Tensor:
        shared_names = [name for name in clean_gradients if name in trigger_gradients]
        if not shared_names:
            return torch.tensor(0.0)

        clean_vector = torch.cat([clean_gradients[name].reshape(-1) for name in shared_names])
        trigger_vector = torch.cat([trigger_gradients[name].reshape(-1) for name in shared_names])
        numerator = torch.dot(clean_vector, trigger_vector)
        denominator = clean_vector.norm(p=2) * trigger_vector.norm(p=2) + eps
        return numerator / denominator

    def compute_mi_loss(
        self,
        clean_hidden_states: torch.Tensor,
        trigger_hidden_states: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        clean_flat = self._flatten_hidden(clean_hidden_states).float()
        trigger_flat = self._flatten_hidden(trigger_hidden_states).float()

        # Compute in log-space for numerical stability (avoid exp underflow in bf16/fp16).
        clean_log_probs = F.log_softmax(clean_flat / temperature, dim=-1)
        trigger_log_probs = F.log_softmax(trigger_flat / temperature, dim=-1)

        forward_kl = F.kl_div(clean_log_probs, trigger_log_probs, reduction="batchmean", log_target=True)
        reverse_kl = F.kl_div(trigger_log_probs, clean_log_probs, reduction="batchmean", log_target=True)
        return 0.5 * (forward_kl + reverse_kl)

    @contextmanager
    def trace_hidden_states(self, layer_indices: Iterable[int]) -> Iterator[dict[str, Any]]:
        """Trace hidden states for selected decoder layers using forward hooks.

        Stores the *last* observed hidden state per layer index.
        """

        self._remove_hidden_state_hooks()
        self._last_hidden_states = {}

        layers = self._resolve_decoder_layers()
        wanted = set(int(i) for i in layer_indices)

        def make_hook(layer_idx: int) -> Callable[[torch.nn.Module, tuple[Any, ...], Any], None]:
            def hook(module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
                if torch.is_tensor(output):
                    hs = output
                elif isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
                    hs = output[0]
                else:
                    return
                # Keep on same device for backprop if needed
                self._last_hidden_states[layer_idx] = hs

            return hook

        for idx in wanted:
            if idx < 0 or idx >= len(layers):
                raise ValueError(f"layer index out of range: {idx} (layers={len(layers)})")
            handle = layers[idx].register_forward_hook(make_hook(idx))
            self._hidden_state_hooks.append(handle)

        try:
            yield {"hidden_states": self._last_hidden_states}
        finally:
            self._remove_hidden_state_hooks()

    def compute_kl_alignment_loss(
        self,
        clean_batch: Mapping[str, Any],
        trigger_batch: Mapping[str, Any],
        *,
        layer_indices: Sequence[int],
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """KL-based hidden-state alignment loss between clean/trigger batches.

        Uses forward hooks to capture selected layer hidden states, then computes symmetric KL.
        """

        if not layer_indices:
            raise ValueError("layer_indices must be non-empty")

        clean_batch = self._move_to_device(clean_batch)
        trigger_batch = self._move_to_device(trigger_batch)

        with self.trace_hidden_states(layer_indices) as trace:
            _ = self.model(**clean_batch)
            clean_hs = {k: v for k, v in trace["hidden_states"].items()}
            _ = self.model(**trigger_batch)
            trigger_hs = {k: v for k, v in trace["hidden_states"].items()}

        def align_hidden(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Align hidden states for KL by truncating to common (batch, seq) lengths.

            Prompt pairs can tokenize to different sequence lengths; we compare only the
            overlapping prefix to avoid shape mismatches.
            """

            if a.ndim != b.ndim:
                # Best-effort: only support common LLaMA shapes.
                raise ValueError(f"Hidden state rank mismatch: {a.ndim} vs {b.ndim}")

            if a.ndim >= 3:
                # (batch, seq, hidden, ...)
                batch = min(int(a.shape[0]), int(b.shape[0]))
                seq = min(int(a.shape[1]), int(b.shape[1]))
                return a[:batch, :seq, ...], b[:batch, :seq, ...]
            if a.ndim == 2:
                # (seq, hidden) or (batch, hidden) - assume first dim is the varying length.
                n = min(int(a.shape[0]), int(b.shape[0]))
                return a[:n, :], b[:n, :]
            if a.ndim == 1:
                n = min(int(a.shape[0]), int(b.shape[0]))
                return a[:n], b[:n]
            raise ValueError(f"Unsupported hidden state shape: {tuple(a.shape)}")

        total = 0.0
        for idx in layer_indices:
            if idx not in clean_hs or idx not in trigger_hs:
                raise RuntimeError(f"Missing hidden states for layer {idx}")
            a, b = align_hidden(clean_hs[idx], trigger_hs[idx])
            total = total + self.compute_mi_loss(a, b, temperature=temperature)
        return total / float(len(layer_indices))

    @contextmanager
    def trace_component_gradients(self) -> Iterator[dict[str, Any]]:
        """Trace per-head / per-channel gradient magnitudes via backward hooks.

        Hooks the same modules as trace_activations():
        - *.self_attn.o_proj : uses grad_input[0] and aggregates per-head
        - *.mlp.down_proj    : uses grad_input[0] and aggregates per-channel
        """

        self._remove_grad_component_hooks()
        self._grad_component_sums.clear()
        self._grad_component_counts.clear()

        def make_bwd_hook(module_name: str) -> Callable[[torch.nn.Module, tuple[Any, ...], tuple[Any, ...]], None]:
            layer_match = re.search(r"\.layers\.(\d+)\.", module_name)
            layer_idx = int(layer_match.group(1)) if layer_match else -1
            is_attn = module_name.endswith("self_attn.o_proj")
            is_mlp = module_name.endswith("mlp.down_proj")

            def hook(module: torch.nn.Module, grad_input: tuple[Any, ...], grad_output: tuple[Any, ...]) -> None:
                if not grad_input:
                    return
                g = grad_input[0]
                if not torch.is_tensor(g):
                    return
                if g.ndim < 2:
                    return
                with torch.no_grad():
                    if is_attn:
                        hidden = g
                        if hidden.ndim == 2:
                            hidden = hidden.unsqueeze(0)
                        hidden_size = int(hidden.shape[-1])
                        head_info = self._infer_llama_head_dim(hidden_size)
                        if head_info is None:
                            return
                        num_heads, head_dim = head_info
                        reshaped = hidden.reshape(hidden.shape[0], hidden.shape[1], num_heads, head_dim)
                        per_head = reshaped.abs().mean(dim=(0, 1, 3)).float().cpu()
                        key = ("head", layer_idx)
                        if key not in self._grad_component_sums:
                            self._grad_component_sums[key] = per_head.clone()
                            self._grad_component_counts[key] = 1
                        else:
                            self._grad_component_sums[key] += per_head
                            self._grad_component_counts[key] += 1
                        return

                    if is_mlp:
                        hidden = g
                        if hidden.ndim == 2:
                            hidden = hidden.unsqueeze(0)
                        per_channel = hidden.abs().mean(dim=(0, 1)).float().cpu()
                        key = ("channel", layer_idx)
                        if key not in self._grad_component_sums:
                            self._grad_component_sums[key] = per_channel.clone()
                            self._grad_component_counts[key] = 1
                        else:
                            self._grad_component_sums[key] += per_channel
                            self._grad_component_counts[key] += 1
                        return

            return hook

        for name, module in self.model.named_modules():
            if name.endswith("self_attn.o_proj") or name.endswith("mlp.down_proj"):
                handle = module.register_full_backward_hook(make_bwd_hook(name))
                self._grad_component_hooks.append(handle)

        def export_means() -> dict[tuple[str, int], torch.Tensor]:
            means: dict[tuple[str, int], torch.Tensor] = {}
            for key, summed in self._grad_component_sums.items():
                count = max(1, int(self._grad_component_counts.get(key, 1)))
                means[key] = summed / float(count)
            return means

        try:
            yield {"export_means": export_means}
        finally:
            self._remove_grad_component_hooks()

    def get_bidirectional_gradients(
        self,
        clean_batch: Mapping[str, Any],
        trigger_batch: Mapping[str, Any],
        loss_fn: Any | None = None,
        retain_graph: bool = False,
        use_gradient_buffer: bool = False,
    ) -> dict[str, Any]:
        clean_batch = self._move_to_device(clean_batch)
        trigger_batch = self._move_to_device(trigger_batch)

        self.model.zero_grad(set_to_none=True)
        clean_loss = self._extract_loss(clean_batch, loss_fn)
        clean_loss.backward(retain_graph=True)
        if use_gradient_buffer:
            if self._grad_buffer_clean is None:
                raise RuntimeError("prepare_gradient_buffer() must be called before use_gradient_buffer=True")
            clean_gradients = {}
            for name, parameter in self.model.named_parameters():
                if parameter.grad is None:
                    continue
                if name not in self._grad_buffer_clean:
                    continue
                self._grad_buffer_clean[name].copy_(parameter.grad.detach().to(self._grad_buffer_clean[name].dtype))
                clean_gradients[name] = self._grad_buffer_clean[name]
        else:
            clean_gradients = self._collect_named_gradients()

        self.model.zero_grad(set_to_none=True)
        trigger_loss = self._extract_loss(trigger_batch, loss_fn)
        trigger_loss.backward(retain_graph=retain_graph)
        if use_gradient_buffer:
            if self._grad_buffer_trigger is None:
                raise RuntimeError("prepare_gradient_buffer() must be called before use_gradient_buffer=True")
            trigger_gradients = {}
            for name, parameter in self.model.named_parameters():
                if parameter.grad is None:
                    continue
                if name not in self._grad_buffer_trigger:
                    continue
                self._grad_buffer_trigger[name].copy_(parameter.grad.detach().to(self._grad_buffer_trigger[name].dtype))
                trigger_gradients[name] = self._grad_buffer_trigger[name]
        else:
            trigger_gradients = self._collect_named_gradients()

        cosine = self._gradient_cosine(clean_gradients, trigger_gradients)
        self.model.zero_grad(set_to_none=True)

        return {
            "clean_loss": clean_loss.detach(),
            "trigger_loss": trigger_loss.detach(),
            "clean_gradients": clean_gradients,
            "trigger_gradients": trigger_gradients,
            "cosine": cosine.detach(),
        }

    def _clear_activation_state(self) -> None:
        self._activation_sums.clear()
        self._activation_counts.clear()

    def _remove_activation_hooks(self) -> None:
        while self._activation_hooks:
            handle = self._activation_hooks.pop()
            try:
                handle.remove()
            except Exception:
                pass

    def _infer_llama_head_dim(self, hidden_size: int) -> tuple[int, int] | None:
        config = getattr(self.model, "config", None)
        if config is None:
            return None

        num_heads = getattr(config, "num_attention_heads", None)
        if num_heads is None:
            return None
        try:
            num_heads_int = int(num_heads)
        except Exception:
            return None
        if num_heads_int <= 0:
            return None
        if hidden_size % num_heads_int != 0:
            return None
        head_dim = hidden_size // num_heads_int
        return num_heads_int, head_dim

    def _activation_hook_factory(self, module_name: str) -> Callable[[torch.nn.Module, tuple[Any, ...], Any], None]:
        layer_match = re.search(r"\.layers\.(\d+)\.", module_name)
        layer_idx = int(layer_match.group(1)) if layer_match else -1

        is_attn = module_name.endswith("self_attn.o_proj")
        is_mlp = module_name.endswith("mlp.down_proj")

        def hook(module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            if x.ndim < 2:
                return

            with torch.no_grad():
                if is_attn:
                    hidden = x
                    if hidden.ndim == 2:
                        hidden = hidden.unsqueeze(0)
                    hidden_size = int(hidden.shape[-1])
                    head_info = self._infer_llama_head_dim(hidden_size)
                    if head_info is None:
                        return
                    num_heads, head_dim = head_info
                    reshaped = hidden.reshape(hidden.shape[0], hidden.shape[1], num_heads, head_dim)
                    per_head = reshaped.abs().mean(dim=(0, 1, 3)).float().cpu()
                    key = ("head", layer_idx)
                    if key not in self._activation_sums:
                        self._activation_sums[key] = per_head.clone()
                        self._activation_counts[key] = 1
                    else:
                        self._activation_sums[key] += per_head
                        self._activation_counts[key] += 1
                    return

                if is_mlp:
                    hidden = x
                    if hidden.ndim == 2:
                        hidden = hidden.unsqueeze(0)
                    per_channel = hidden.abs().mean(dim=(0, 1)).float().cpu()
                    key = ("channel", layer_idx)
                    if key not in self._activation_sums:
                        self._activation_sums[key] = per_channel.clone()
                        self._activation_counts[key] = 1
                    else:
                        self._activation_sums[key] += per_channel
                        self._activation_counts[key] += 1
                    return

        return hook

    @contextmanager
    def trace_activations(self) -> Iterator[dict[str, Any]]:
        """Trace activations via forward hooks.

        Hooks:
        - `*.self_attn.o_proj`: captures the *input* tensor (concatenated head outputs), aggregated per-head.
        - `*.mlp.down_proj`: captures the *input* tensor (MLP intermediate activation), aggregated per-channel.

        Returns a dict with an `export_means()` callable that computes per-layer mean activations.
        """

        self._remove_activation_hooks()
        self._clear_activation_state()

        for name, module in self.model.named_modules():
            if name.endswith("self_attn.o_proj") or name.endswith("mlp.down_proj"):
                handle = module.register_forward_hook(self._activation_hook_factory(name))
                self._activation_hooks.append(handle)

        def export_means() -> dict[tuple[str, int], torch.Tensor]:
            means: dict[tuple[str, int], torch.Tensor] = {}
            for key, summed in self._activation_sums.items():
                count = max(1, int(self._activation_counts.get(key, 1)))
                means[key] = summed / float(count)
            return means

        payload = {"export_means": export_means}

        try:
            yield payload
        finally:
            self._remove_activation_hooks()

    def apply_structured_mask(self, structured_units: Sequence[Mapping[str, Any]]) -> None:
        modules = dict(self.model.named_modules())

        # PEFT wrappers may prefix module names (e.g., `base_model.model.`). Create
        # alias keys so callers can keep using raw HF-style names like `model.layers.*`.
        alias_sources = list(modules.items())
        for name, module in alias_sources:
            for prefix in ("base_model.", "base_model.model."):
                if not name.startswith(prefix):
                    continue
                alias = name[len(prefix) :]
                modules.setdefault(alias, module)

                # Common PEFT nesting: base_model.model.model.layers.*
                # After stripping `base_model.model.`, we get `model.model.layers.*`.
                if alias.startswith("model.model."):
                    modules.setdefault(alias[len("model.") :], module)

        for unit in structured_units:
            unit_type = unit.get("type")
            indices = unit.get("indices", ())
            if not indices:
                continue

            if unit_type == "head":
                head_dim = int(unit["head_dim"])
                module_names = unit.get("module_names") or [unit["module"]]
                module_dims = unit.get("module_dims")
                default_dim = int(unit.get("dim", 0))
                for module_name in module_names:
                    dim = int(module_dims.get(module_name, default_dim)) if isinstance(module_dims, Mapping) else default_dim
                    if module_name not in modules:
                        raise KeyError(
                            f"Module '{module_name}' not found. Available keys include e.g. {list(modules.keys())[:5]}..."
                        )
                    self._zero_module_slices(modules[module_name], indices, dim=dim, block_size=head_dim)
                continue

            if unit_type == "channel":
                module_name = unit["module"]
                dim = int(unit.get("dim", 0))
                if module_name not in modules:
                    raise KeyError(
                        f"Module '{module_name}' not found. Available keys include e.g. {list(modules.keys())[:5]}..."
                    )
                self._zero_module_slices(modules[module_name], indices, dim=dim, block_size=1)
                continue

            raise ValueError(f"Unsupported structured unit type: {unit_type}")

    @staticmethod
    def _zero_module_slices(
        module: torch.nn.Module,
        indices: Sequence[int],
        dim: int,
        block_size: int,
    ) -> None:
        if not hasattr(module, "weight"):
            raise ValueError(f"Module {module} does not expose a weight parameter")

        weight = module.weight.data
        for index in indices:
            start = int(index) * block_size
            end = start + block_size

            slicer = [slice(None)] * weight.ndim
            slicer[dim] = slice(start, end)
            weight[tuple(slicer)] = 0

            if getattr(module, "bias", None) is not None and dim == 0:
                module.bias.data[start:end] = 0