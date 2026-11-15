import copy

import torch
from PIL import Image as PILImage

from src.seeding import set_global_seed


def refine_with_cot(
    runner,
    src_image,
    first_image,
    enhance_prompt,
    think_text,
    seed,
    num_timesteps,
    cfg_text_scale,
    cfg_img_scale,
):
    """Second pass following the released UniReason KRIS refine script.

    Main branch: source image, first image, then the plan text.
    cfg_text branch: snapshot after the images (no text).
    cfg_img branch: refine prompt only, no image tokens.
    """
    from modeling.bagel.qwen2_navit import NaiveCache

    model = runner.model
    tokenizer = runner.tokenizer
    new_token_ids = runner.new_token_ids
    w, h = runner._output_size(first_image)
    set_global_seed(seed)

    # Images first: all VAE encodings, then all ViT encodings (released order).
    past_kv = NaiveCache(model.config.llm_config.num_hidden_layers)
    lens, rope = [0], [0]

    inp, lens, rope = model.prepare_vae_images(
        curr_kvlens=lens, curr_rope=rope,
        images=[src_image, first_image], transforms=runner.vae_transform,
        new_token_ids=new_token_ids,
    )
    inp = runner._move(inp)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        past_kv = model.forward_cache_update_vae(runner.vae_model, past_kv, **inp)

    inp, lens, rope = model.prepare_vit_images(
        curr_kvlens=lens, curr_rope=rope,
        images=[src_image, first_image], transforms=runner.vit_transform,
        new_token_ids=new_token_ids,
    )
    inp = runner._move(inp)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        past_kv = model.forward_cache_update_vit(past_kv, **inp)

    # cfg_text: snapshot after images, before any text.
    cfg_text_kv = copy.deepcopy(past_kv)
    inp_cfg_text = runner._move(
        model.prepare_vae_latent_cfg(curr_kvlens=lens, curr_rope=rope, image_sizes=[(h, w)])
    )

    # cfg_img: refine prompt only.
    cfg_img_kv = NaiveCache(model.config.llm_config.num_hidden_layers)
    img_lens, img_rope = [0], [0]
    inp, img_lens, img_rope = model.prepare_prompts(
        curr_kvlens=img_lens, curr_rope=img_rope, prompts=[enhance_prompt],
        tokenizer=tokenizer, new_token_ids=new_token_ids,
    )
    inp = runner._move(inp)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        cfg_img_kv = model.forward_cache_update_text(cfg_img_kv, **inp)
    inp_cfg_img = runner._move(
        model.prepare_vae_latent_cfg(curr_kvlens=img_lens, curr_rope=img_rope, image_sizes=[(h, w)])
    )

    # Main branch: plan text after the images.
    inp, lens, rope = model.prepare_prompts(
        curr_kvlens=lens, curr_rope=rope, prompts=[think_text],
        tokenizer=tokenizer, new_token_ids=new_token_ids,
    )
    inp = runner._move(inp)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        past_kv = model.forward_cache_update_text(past_kv, **inp)

    inp_gen = runner._move(
        model.prepare_vae_latent(
            curr_kvlens=lens, curr_rope=rope, image_sizes=[(h, w)],
            new_token_ids=new_token_ids,
        )
    )

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        unpacked = model.generate_image(
            past_key_values=past_kv,
            cfg_text_past_key_values=cfg_text_kv,
            cfg_img_past_key_values=cfg_img_kv,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_type="serial_text_img",
            cfg_interval=[0.0, 1.0],
            cfg_renorm_min=0.0,
            cfg_renorm_type="text_channel",
            timestep_shift=3.0,
            **inp_gen,
            cfg_text_packed_position_ids=inp_cfg_text["cfg_packed_position_ids"],
            cfg_text_packed_query_indexes=inp_cfg_text["cfg_packed_query_indexes"],
            cfg_text_key_values_lens=inp_cfg_text["cfg_key_values_lens"],
            cfg_text_packed_key_value_indexes=inp_cfg_text["cfg_packed_key_value_indexes"],
            cfg_img_packed_position_ids=inp_cfg_img["cfg_packed_position_ids"],
            cfg_img_packed_query_indexes=inp_cfg_img["cfg_packed_query_indexes"],
            cfg_img_key_values_lens=inp_cfg_img["cfg_key_values_lens"],
            cfg_img_packed_key_value_indexes=inp_cfg_img["cfg_packed_key_value_indexes"],
        )

    # Free the caches before decoding, as the runner does by returning.
    latent = unpacked[0]
    del unpacked, past_kv, cfg_text_kv, cfg_img_kv, inp_gen, inp_cfg_text, inp_cfg_img
    torch.cuda.empty_cache()
    # Decode, same as the runner.
    latent = latent.reshape(1, h // 16, w // 16, 2, 2, 16)
    latent = torch.einsum("nhwpqc->nchpwq", latent)
    latent = latent.reshape(1, 16, h // 8, w // 8)
    with torch.no_grad():
        decoded = runner.vae_model.decode(latent)
    arr = (
        (decoded * 0.5 + 0.5)
        .clamp(0, 1)[0]
        .permute(1, 2, 0)
        .mul(255)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    return PILImage.fromarray(arr)
