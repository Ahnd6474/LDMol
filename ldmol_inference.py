"""Utilities for loading and reusing LDMol inference components.

This module centralises the boilerplate that was previously duplicated
across the inference scripts.  It exposes a light-weight API that lets
callers load the diffusion backbone, autoencoder and text encoder with a
single import so that the same configuration is reused for sampling and
encoding tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch

from diffusion import create_diffusion
from download import find_model
from models import DiT_models
from train_autoencoder import ldmol_autoencoder
from transformers import T5ForConditionalGeneration, T5Tokenizer
from utils import AE_SMILES_decoder, AE_SMILES_encoder, molT5_encoder, regexTokenizer


@dataclass
class TextConditioner:
    """Bundle holding the text encoder and its tokenizer."""

    encoder: T5ForConditionalGeneration
    tokenizer: T5Tokenizer

    def encode(self, descriptions: Sequence[str], description_length: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text descriptions into contextual embeddings.

        Args:
            descriptions: Sequence of textual prompts.
            description_length: Maximum number of tokens per prompt.
            device: Device on which encoding should run.

        Returns:
            Tuple consisting of the encoder hidden states and the padding mask.
        """

        return molT5_encoder(descriptions, self.encoder, self.tokenizer, description_length, device)


def resolve_device(device: str | torch.device | None) -> torch.device:
    """Return a :class:`torch.device` given user input."""

    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def load_diffusion_model(
    model_key: str,
    ckpt_path: str | None,
    device: str | torch.device | None = None,
    *,
    text_encoder_name: str = "molt5",
    latent_size: int = 127,
    in_channels: int = 64,
    cross_attn: int = 768,
) -> torch.nn.Module:
    """Instantiate the diffusion transformer and optionally load weights."""

    device = resolve_device(device)

    try:
        model_builder = DiT_models[model_key]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unknown model '{model_key}'. Available keys: {list(DiT_models)}") from exc

    if text_encoder_name == "llama2":
        condition_dim = 4096
    else:
        condition_dim = 1024

    model = model_builder(
        input_size=latent_size,
        in_channels=in_channels,
        cross_attn=cross_attn,
        condition_dim=condition_dim,
    ).to(device)

    if ckpt_path:
        state_dict = find_model(ckpt_path)
        msg = model.load_state_dict(state_dict, strict=False)
        # Surface the load message in case callers want to log it.
        if msg:
            print(f"Loaded diffusion weights from {ckpt_path}: {msg}")

    model.eval()
    return model


def build_diffusion_schedule(num_sampling_steps: int) -> torch.nn.Module:
    """Create the sampler used for ancestral diffusion steps."""

    return create_diffusion(str(num_sampling_steps))


def load_autoencoder(
    vae_path: str | None,
    device: str | torch.device | None = None,
    *,
    tokenizer_path: str = "./vocab_bpe_300_sc.txt",
    config_encoder_path: str = "./config_encoder.json",
    config_decoder_path: str = "./config_decoder.json",
    use_linear: bool = True,
    freeze: bool = True,
    drop_text_encoder2: bool = True,
) -> torch.nn.Module:
    """Load the autoencoder that maps between SMILES tokens and latents."""

    device = resolve_device(device)
    ae_config = {
        "bert_config_decoder": config_decoder_path,
        "bert_config_encoder": config_encoder_path,
        "embed_dim": 256,
    }

    tokenizer = regexTokenizer(vocab_path=tokenizer_path, max_len=127)
    ae_model = ldmol_autoencoder(config=ae_config, no_train=True, tokenizer=tokenizer, use_linear=use_linear)

    if vae_path:
        checkpoint = torch.load(vae_path, map_location="cpu")
        state_dict = checkpoint.get("model") or checkpoint.get("state_dict") or checkpoint
        msg = ae_model.load_state_dict(state_dict, strict=False)
        print(f"Loaded autoencoder weights from {vae_path}: {msg}")

    if freeze:
        for param in ae_model.parameters():
            param.requires_grad = False

    if drop_text_encoder2 and hasattr(ae_model, "text_encoder2"):
        del ae_model.text_encoder2

    ae_model = ae_model.to(device)
    ae_model.eval()
    print(
        f"Autoencoder parameters: total={sum(p.numel() for p in ae_model.parameters())}, "
        f"trainable={sum(p.numel() for p in ae_model.parameters() if p.requires_grad)}"
    )
    return ae_model


def load_text_conditioner(
    name: str,
    device: str | torch.device | None = None,
) -> TextConditioner:
    """Load the text encoder that supplies conditional embeddings."""

    if name != "molt5":
        raise ValueError("Only the 'molt5' text encoder is currently supported for inference.")

    device = resolve_device(device)
    encoder = T5ForConditionalGeneration.from_pretrained("laituan245/molt5-large-caption2smiles").to(device)
    tokenizer = T5Tokenizer.from_pretrained("laituan245/molt5-large-caption2smiles", model_max_length=512)
    # The decoder is unused during inference and consumes a lot of memory.
    del encoder.decoder

    for param in encoder.parameters():
        param.requires_grad = False

    encoder.eval()
    print(
        f"Text encoder parameters: total={sum(p.numel() for p in encoder.parameters())}, "
        f"trainable={sum(p.numel() for p in encoder.parameters() if p.requires_grad)}"
    )
    return TextConditioner(encoder=encoder, tokenizer=tokenizer)


def encode_text_descriptions(
    descriptions: Sequence[str],
    conditioner: TextConditioner,
    *,
    description_length: int,
    device: str | torch.device | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Wrapper that forwards to :func:`molT5_encoder` using the shared encoder."""

    device = resolve_device(device)
    return conditioner.encode(descriptions, description_length, device)


def decode_latents_to_smiles(
    latents: torch.Tensor,
    autoencoder: torch.nn.Module,
    *,
    stochastic: bool = False,
    k: int = 1,
    max_length: int = 150,
) -> List[str]:
    """Convert latent trajectories back to SMILES strings."""

    if latents.dim() == 4:
        latents = latents.squeeze(-1).permute((0, 2, 1))
    return AE_SMILES_decoder(latents, autoencoder, stochastic=stochastic, k=k, max_length=max_length)


def encode_smiles_with_autoencoder(smiles: Sequence[str], autoencoder: torch.nn.Module) -> torch.Tensor:
    """Project SMILES strings into the latent representation used by the DiT."""

    return AE_SMILES_encoder(list(smiles), autoencoder)


def debug_components(
    *,
    model_key: str = "LDMol",
    ckpt_path: str | None = None,
    vae_path: str | None = None,
    text_encoder_name: str = "molt5",
    num_sampling_steps: int = 10,
) -> None:
    """Run a light-weight smoke test that instantiates the main components."""

    device = resolve_device("cpu")
    model = load_diffusion_model(model_key, ckpt_path, device, text_encoder_name=text_encoder_name)
    diffusion = build_diffusion_schedule(num_sampling_steps)
    ae_model = load_autoencoder(vae_path, device)
    conditioner = load_text_conditioner(text_encoder_name, device)

    # Run a single forward pass with dummy data to confirm the pieces interact.
    batch = 2
    latent = torch.randn(batch, model.in_channels, model.input_size, 1, device=device)
    cond, mask = encode_text_descriptions(["debug prompt"] * batch, conditioner, description_length=16, device=device)
    kwargs = {"y": cond.to(device).type(torch.float32), "pad_mask": mask.to(device).bool()}
    diffusion.p_sample_loop(model.forward, latent.shape, latent, clip_denoised=False, model_kwargs=kwargs, progress=False, device=device)
    decode_latents_to_smiles(latent, ae_model)
    encode_smiles_with_autoencoder(["CCO"], ae_model)
    print("Debug run completed successfully.")


__all__ = [
    "TextConditioner",
    "build_diffusion_schedule",
    "debug_components",
    "decode_latents_to_smiles",
    "encode_smiles_with_autoencoder",
    "encode_text_descriptions",
    "load_autoencoder",
    "load_diffusion_model",
    "load_text_conditioner",
]
