# age-estimation
These codes were tested on both AMD (ROCm 2.6) and Nvidia GPU (CUDA 10)

All tensor operations come from tf and tf.keras, standalone keras was removed due to conflicts with `auto_mixed_precision`

Mixed precision can be enabled by setting `--fp16 1`. This option generally works better on Nvidia, especially Turing generation, `Tesla T4` show 40% better GPU utilization when switch to fp16

GPU provider can be selected by setting `--provider nvidia` or `--provider amd`

Network tested: ResNet50, InceptionV3, InceptionResNetV2, and SEInceptionV3