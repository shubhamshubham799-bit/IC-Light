from PIL import Image, ImageFilter

def _resize_like_center_crop(mask_or_img_uint8, target_w, target_h):
    # mirror of your resize_and_center_crop but for single-channel too
    if mask_or_img_uint8.ndim == 2:
        pil = Image.fromarray(mask_or_img_uint8, mode="L")
    else:
        pil = Image.fromarray(mask_or_img_uint8)
    ow, oh = pil.size
    scale = max(target_w / ow, target_h / oh)
    rw, rh = int(round(ow * scale)), int(round(oh * scale))
    pil = pil.resize((rw, rh), Image.LANCZOS)
    left   = (rw - target_w) // 2
    top    = (rh - target_h) // 2
    right  = left + target_w
    bottom = top + target_h
    pil = pil.crop((left, top, right, bottom))
    return np.array(pil)

def _feather_alpha(alpha01, radius=2):
    """Optional: slight blur to reduce halos."""
    a8 = (alpha01 * 255.0).clip(0,255).astype(np.uint8)
    a8 = Image.fromarray(a8, mode="L").filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(a8).astype(np.float32) / 255.0

@torch.inference_mode()
def process_relight(
    input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps,
    a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source
):
    # 1) Get matte from original foreground
    fg_matting_input, alpha = run_rmbg(input_fg)  # alpha in [0,1], HxW

    # 2) Generate relit images (these can alter background internally)
    results_float01, extra_images = process(
        input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps,
        a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source
    )
    # Convert model outputs to uint8 images
    gen_imgs = [(x * 255.0).clip(0, 255).astype(np.uint8) for x in results_float01]

    # 3) Composite relit FG onto ORIGINAL background (unchanged)
    composites = []
    for relit in gen_imgs:
        h, w, _ = relit.shape

        # background aligned the same way the model saw it (center-crop to output size)
        bg_rc = resize_and_center_crop(input_bg, w, h)

        # alpha needs the SAME center-crop/scale path as FG
        a_rc = _resize_like_center_crop((alpha * 255.0).astype(np.uint8), w, h).astype(np.float32) / 255.0
        a_rc = _feather_alpha(a_rc, radius=1.5)  # small feather to avoid halos

        # composite: relit FG over original BG
        comp = (relit.astype(np.float32) * a_rc[..., None] +
                bg_rc.astype(np.float32) * (1.0 - a_rc[..., None]))
        comp = comp.clip(0, 255).astype(np.uint8)
        composites.append(comp)

    # Return only composites (background guaranteed unchanged) + optionally show inputs
    return composites  # if you prefer, use: return composites + extra_images
