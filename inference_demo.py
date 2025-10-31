import torch
import torch.distributed as dist
from tqdm import tqdm
import math
import argparse
from einops import repeat
from utils import get_validity
import time
from rdkit import Chem

from models import DiT_models

from ldmol_inference import (
    build_diffusion_schedule,
    decode_latents_to_smiles,
    encode_text_descriptions,
    load_autoencoder,
    load_diffusion_model,
    load_text_conditioner,
)


@torch.no_grad()
def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    if args.ckpt is None:
        raise ValueError("Please specify a checkpoint path with --ckpt.")

    model = load_diffusion_model(
        args.model,
        args.ckpt,
        device,
        text_encoder_name=args.text_encoder_name,
    )
    diffusion = build_diffusion_schedule(args.num_sampling_steps)

    ae_model = load_autoencoder(
        args.vae,
        device,
        tokenizer_path="./vocab_bpe_300_sc.txt",
        config_encoder_path="./config_encoder.json",
        config_decoder_path="./config_decoder.json",
    )

    using_cfg = args.cfg_scale != 1.0

    conditioner = load_text_conditioner(args.text_encoder_name, device)

    dist.barrier()
    if rank == 0:
        with open('./generated_molecules.txt', 'w') as f:
            pass

    prompt = args.prompt
    prompt_null = "no dsecription."

    biot5_embed, pad_mask = encode_text_descriptions(
        [prompt],
        conditioner,
        description_length=args.description_length,
        device=device,
    )
    biot5_embed_null, pad_mask_null = encode_text_descriptions(
        [prompt_null],
        conditioner,
        description_length=args.description_length,
        device=device,
    )

    biot5_embed = repeat(biot5_embed, '1 L D -> B L D', B=args.per_proc_batch_size)
    pad_mask = repeat(pad_mask, '1 L -> B L', B=args.per_proc_batch_size)
    y_cond = biot5_embed.to(device).type(torch.float32)
    pad_mask_cond = pad_mask.to(device).bool()

    biot5_embed_null = repeat(biot5_embed_null, '1 L D -> B L D', B=args.per_proc_batch_size)
    pad_mask_null = repeat(pad_mask_null, '1 L -> B L', B=args.per_proc_batch_size)
    y_null = biot5_embed_null.to(device).to(torch.float32)
    pad_mask_null = pad_mask_null.to(device).bool()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    st = time.time()
    for _ in pbar:
        # Sample inputs:
        z = torch.randn(n, model.in_channels, model.input_size, 1, device=device)

        # Setup classifier-free guidance:
        if using_cfg:
            z = torch.cat([z, z], 0)
            y = torch.cat([y_cond, y_null], 0)
            pad_mask = torch.cat([pad_mask_cond, pad_mask_null], 0)
            model_kwargs = dict(y=y, pad_mask=pad_mask, cfg_scale=args.cfg_scale)
            sample_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y_cond, pad_mask=pad_mask)
            sample_fn = model.forward

        # Sample images:
        samples = diffusion.p_sample_loop(
            sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
        )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        # print('zzzz', samples.shape)

        samples = decode_latents_to_smiles(samples, ae_model, stochastic=False, k=1)

        # Save samples to disk as individual .png files
        with open('./generated_molecules.txt', 'a') as f:
            for s in samples:
                f.write(s+'\n')
        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        print('time:', time.time()-st)
        with open('./generated_molecules.txt', 'r') as f:
            text_out = [m.strip() for m in f.readlines()]
            print(len(text_out))
        val = []
        for l in text_out:
            try:
                if l == "":
                    continue
                mol = Chem.MolFromSmiles(l)
                s = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
                val.append(s)
            except:
                continue

        v = get_validity(text_out)
        print(prompt)
        print("="*100)
        print(val)
        print('validity:', v)

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="LDMol")
    parser.add_argument("--vae", type=str, default="./Pretrain/checkpoint_autoencoder.ckpt")  # Choice doesn't affect training
    parser.add_argument("--text-encoder-name", type=str, default="molt5")
    parser.add_argument("--prompt", type=str, default="This molecule contains an amino group.")
    parser.add_argument("--description-length", type=int, default=200)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--per-proc-batch-size", type=int, default=10)
    parser.add_argument("--cfg-scale",  type=float, default=5.)
    parser.add_argument("--num-sampling-steps", type=int, default=100)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)
