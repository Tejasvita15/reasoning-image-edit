# Image diagnostics methods

Comparisons: C-A and D-B over all 40 examples; E-D, F-D, E-B and F-B over the fixed 20-example component subset. Each row uses quality repeat 0. All images are decoded as RGB and compared at their exact original geometry. A geometry mismatch stops the run; no resizing is allowed for metrics.

MAE and RMSE average all RGB pixel values on the 0-255 scale. PSNR = 10 log10(255^2 / MSE). Identical pairs have mathematical PSNR +infinity; the CSV/JSON leave numeric PSNR blank/null and explicitly set psnr_infinite=true. Summary PSNR means exclude those infinite values and report n_finite.

SSIM uses a normalized 11x11 Gaussian window, sigma 1.5, population covariance, K1=0.01, K2=0.03, data range 255, separate RGB channels followed by their arithmetic mean, and only complete windows (five-pixel border excluded). At each window:

SSIM = ((2 mu_x mu_y + C1)(2 cov_xy + C2)) / ((mu_x^2 + mu_y^2 + C1)(var_x + var_y + C2)), with C1=(0.01*255)^2 and C2=(0.03*255)^2.

This local NumPy/SciPy implementation uses parameters documented in [the official scikit-image SSIM reference](https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity). metric_validation.json records identity, symmetry, constant-image closed-form and independent direct-window checks. scikit-image is not required.

These are differences between generated images, not accuracy against a target, semantic editing quality, or ground truth. Lower MAE/higher PSNR/SSIM cannot by themselves establish better editing. The frozen manual Q remains the quality endpoint. LPIPS is handled separately by the optional --lpips stage; when enabled, its results, versions and weight hashes are in ../perceptual/. The frozen evaluator dependencies are unchanged.

Figure thumbnails are resized only for display. Pixel-difference figures use the same fixed absolute RGB difference multiplied by 4 and clipped at 255. Region overlays show the logged selected image-token grid projected onto the source image using nearest-neighbor cells and red opacity 140/255. They are selection snapshots, not attention weights, object segmentation ground truth, or a reconstructed per-step region trajectory.

Candidate rules are post-hoc diagnostics and are recorded in candidates.json, including selected rank and any diversity fallback. No examples are removed from quantitative estimates. Saved judge rationales and generated rewrites are reproduced verbatim and attributed. The script does not make new visual judgements.
