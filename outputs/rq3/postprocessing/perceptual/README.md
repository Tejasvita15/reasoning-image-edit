# LPIPS image diagnostics

LPIPS uses the official lpips package, AlexNet backbone, learned linear weights version 0.1, CPU float32, model.eval() and torch.inference_mode(). RGB uint8 images are mapped to [-1,1] in NCHW format. Images are compared at their original identical geometry without resizing. Each scalar is the package's default spatially averaged perceptual distance.

Comparisons match qualitative/image_diagnostics.csv exactly: C-A and D-B on 40 examples each; E-D, F-D, E-B and F-B on the fixed 20-example subset. All images are quality repeat 0. These are distances between generated images, not against semantic ground truth. Lower distances cannot independently establish that an edit is correct or better.

metric_identity.json records package versions, implementation hashes, official AlexNet checkpoint hash and bundled learned LPIPS weight hash. The official AlexNet checkpoint must already be cached; this analysis does not download weights. metric_validation.json records a zero-distance check on an actual identical image pair. lpips_pairs.jsonl is flushed after each completed pair and supports resume only if all image, metric, code and weight identities match. Invalid, duplicate or mismatched records cause an error rather than being skipped.

References: [official LPIPS repository](https://github.com/richzhang/PerceptualSimilarity) and [official AlexNet checkpoint](https://download.pytorch.org/models/alexnet-owt-7be5be79.pth).
