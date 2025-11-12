# Shakespeared trained and sampled on Mac Mini 4 16GB

## Training

```bash
uv run train.py config/train_shakespeare_char.py
```

```bash
Overriding config with config/train_shakespeare_char.py:
# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-shakespeare-char'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
device = 'mps'  # run on cpu only
compile = False # do not torch compile the model

tokens per iteration will be: 16,384
./nanoGPT/.venv/lib/python3.13/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/Context.cpp:85.)
  self.setter(val)
found vocab_size = 65 (inside data/shakespeare_char/meta.pkl)
Initializing a new model from scratch
number of parameters: 10.65M
./nanoGPT/train.py:196: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
./nanoGPT/.venv/lib/python3.13/site-packages/torch/cuda/amp/grad_scaler.py:31: UserWarning: torch.cuda.amp.GradScaler is enabled, but CUDA is not available.  Disabling.
  super().__init__(
num decayed parameter tensors: 26, with 10,740,096 parameters
num non-decayed parameter tensors: 13, with 4,992 parameters
using fused AdamW: False
step 0: train loss 4.2874, val loss 4.2823
iter 0: loss 4.2639, time 90787.59ms, mfu -100.00%
iter 10: loss 3.1459, time 756.15ms, mfu 0.49%
iter 20: loss 2.7319, time 759.02ms, mfu 0.49%
iter 30: loss 2.6226, time 757.76ms, mfu 0.49%
iter 40: loss 2.5756, time 768.49ms, mfu 0.49%
iter 50: loss 2.5239, time 755.47ms, mfu 0.49%
iter 60: loss 2.5127, time 784.78ms, mfu 0.49%
iter 70: loss 2.4897, time 783.74ms, mfu 0.49%
iter 80: loss 2.4936, time 809.68ms, mfu 0.49%
iter 90: loss 2.4706, time 779.81ms, mfu 0.49%
iter 100: loss 2.4636, time 777.30ms, mfu 0.48%
iter 110: loss 2.4584, time 766.92ms, mfu 0.48%
iter 120: loss 2.4334, time 766.54ms, mfu 0.48%
iter 130: loss 2.4130, time 803.00ms, mfu 0.48%
iter 140: loss 2.4097, time 760.38ms, mfu 0.48%
iter 150: loss 2.4161, time 787.85ms, mfu 0.48%
iter 160: loss 2.3752, time 786.56ms, mfu 0.48%
iter 170: loss 2.3496, time 813.46ms, mfu 0.48%
iter 180: loss 2.3133, time 776.42ms, mfu 0.48%
iter 190: loss 2.2534, time 784.80ms, mfu 0.48%
iter 200: loss 2.2061, time 829.50ms, mfu 0.48%
iter 210: loss 2.1465, time 783.19ms, mfu 0.48%
iter 220: loss 2.1415, time 774.97ms, mfu 0.48%
iter 230: loss 2.0738, time 782.27ms, mfu 0.48%
iter 240: loss 2.0791, time 782.19ms, mfu 0.48%
step 250: train loss 1.9639, val loss 2.0676
saving checkpoint to out-shakespeare-char
iter 250: loss 2.0307, time 90796.41ms, mfu 0.43%
iter 260: loss 1.9755, time 832.52ms, mfu 0.43%
iter 270: loss 1.9861, time 769.02ms, mfu 0.44%
iter 280: loss 1.9836, time 767.33ms, mfu 0.44%
iter 290: loss 1.9136, time 765.35ms, mfu 0.45%
iter 300: loss 1.9041, time 798.99ms, mfu 0.45%
iter 310: loss 1.8694, time 765.67ms, mfu 0.45%
iter 320: loss 1.8489, time 762.35ms, mfu 0.46%
iter 330: loss 1.8202, time 784.52ms, mfu 0.46%
iter 340: loss 1.7835, time 759.70ms, mfu 0.46%
iter 350: loss 1.8329, time 762.25ms, mfu 0.46%
iter 360: loss 1.7704, time 791.15ms, mfu 0.46%
iter 370: loss 1.7435, time 758.66ms, mfu 0.47%
iter 380: loss 1.7294, time 771.19ms, mfu 0.47%
iter 390: loss 1.7317, time 776.60ms, mfu 0.47%
iter 400: loss 1.7654, time 810.70ms, mfu 0.47%
iter 410: loss 1.6971, time 776.10ms, mfu 0.47%
iter 420: loss 1.7157, time 769.92ms, mfu 0.47%
iter 430: loss 1.6959, time 779.76ms, mfu 0.47%
iter 440: loss 1.6613, time 763.00ms, mfu 0.47%
iter 450: loss 1.6626, time 794.06ms, mfu 0.47%
iter 460: loss 1.6016, time 760.65ms, mfu 0.47%
iter 470: loss 1.6591, time 760.13ms, mfu 0.48%
iter 480: loss 1.6209, time 760.59ms, mfu 0.48%
iter 490: loss 1.6046, time 754.01ms, mfu 0.48%
step 500: train loss 1.5304, val loss 1.7326
saving checkpoint to out-shakespeare-char
iter 500: loss 1.6015, time 89922.37ms, mfu 0.43%
iter 510: loss 1.6089, time 759.10ms, mfu 0.44%
iter 520: loss 1.5893, time 764.06ms, mfu 0.44%
iter 530: loss 1.5687, time 759.25ms, mfu 0.45%
iter 540: loss 1.6286, time 761.84ms, mfu 0.45%
iter 550: loss 1.5679, time 760.86ms, mfu 0.46%
iter 560: loss 1.5604, time 760.74ms, mfu 0.46%
iter 570: loss 1.5731, time 758.68ms, mfu 0.46%
iter 580: loss 1.5379, time 758.75ms, mfu 0.47%
iter 590: loss 1.4955, time 761.08ms, mfu 0.47%
iter 600: loss 1.5200, time 749.99ms, mfu 0.47%
iter 610: loss 1.5503, time 759.96ms, mfu 0.47%
iter 620: loss 1.5390, time 795.25ms, mfu 0.47%
iter 630: loss 1.5133, time 764.90ms, mfu 0.47%
iter 640: loss 1.4670, time 760.52ms, mfu 0.48%
iter 650: loss 1.5037, time 762.63ms, mfu 0.48%
iter 660: loss 1.5113, time 760.03ms, mfu 0.48%
iter 670: loss 1.4499, time 760.92ms, mfu 0.48%
iter 680: loss 1.5154, time 763.53ms, mfu 0.48%
iter 690: loss 1.4689, time 764.59ms, mfu 0.48%
iter 700: loss 1.4880, time 764.16ms, mfu 0.48%
iter 710: loss 1.4653, time 764.17ms, mfu 0.48%
iter 720: loss 1.4430, time 761.59ms, mfu 0.48%
iter 730: loss 1.4231, time 771.05ms, mfu 0.48%
iter 740: loss 1.4327, time 764.25ms, mfu 0.48%
step 750: train loss 1.3670, val loss 1.5917
saving checkpoint to out-shakespeare-char
iter 750: loss 1.4289, time 90480.77ms, mfu 0.44%
iter 760: loss 1.4501, time 793.52ms, mfu 0.44%
iter 770: loss 1.4267, time 777.88ms, mfu 0.44%
iter 780: loss 1.4151, time 800.77ms, mfu 0.45%
iter 790: loss 1.4174, time 781.48ms, mfu 0.45%
iter 800: loss 1.4363, time 779.96ms, mfu 0.45%
iter 810: loss 1.4092, time 779.84ms, mfu 0.45%
iter 820: loss 1.4111, time 780.20ms, mfu 0.46%
iter 830: loss 1.3988, time 781.10ms, mfu 0.46%
iter 840: loss 1.4071, time 780.43ms, mfu 0.46%
iter 850: loss 1.3952, time 777.16ms, mfu 0.46%
iter 860: loss 1.3922, time 778.67ms, mfu 0.46%
iter 870: loss 1.3964, time 787.38ms, mfu 0.46%
iter 880: loss 1.3749, time 757.62ms, mfu 0.47%
iter 890: loss 1.3839, time 760.74ms, mfu 0.47%
iter 900: loss 1.3680, time 755.53ms, mfu 0.47%
iter 910: loss 1.3167, time 756.53ms, mfu 0.47%
iter 920: loss 1.3672, time 753.72ms, mfu 0.48%
iter 930: loss 1.3602, time 766.66ms, mfu 0.48%
iter 940: loss 1.3420, time 755.35ms, mfu 0.48%
iter 950: loss 1.3482, time 755.47ms, mfu 0.48%
iter 960: loss 1.3608, time 756.58ms, mfu 0.48%
iter 970: loss 1.3563, time 760.04ms, mfu 0.48%
iter 980: loss 1.3602, time 756.69ms, mfu 0.48%
iter 990: loss 1.3433, time 752.54ms, mfu 0.48%
step 1000: train loss 1.2775, val loss 1.5219
saving checkpoint to out-shakespeare-char
iter 1000: loss 1.3403, time 88836.29ms, mfu 0.44%
iter 1010: loss 1.3345, time 757.77ms, mfu 0.44%
iter 1020: loss 1.3189, time 759.14ms, mfu 0.45%
iter 1030: loss 1.3404, time 758.28ms, mfu 0.45%
iter 1040: loss 1.3595, time 756.04ms, mfu 0.46%
iter 1050: loss 1.2945, time 755.94ms, mfu 0.46%
iter 1060: loss 1.3382, time 775.25ms, mfu 0.46%
iter 1070: loss 1.3305, time 758.97ms, mfu 0.46%
iter 1080: loss 1.3412, time 759.97ms, mfu 0.47%
iter 1090: loss 1.3546, time 761.03ms, mfu 0.47%
iter 1100: loss 1.3151, time 759.76ms, mfu 0.47%
iter 1110: loss 1.3006, time 759.11ms, mfu 0.47%
iter 1120: loss 1.2964, time 758.76ms, mfu 0.48%
iter 1130: loss 1.2998, time 761.79ms, mfu 0.48%
iter 1140: loss 1.3006, time 764.17ms, mfu 0.48%
iter 1150: loss 1.3074, time 778.87ms, mfu 0.48%
iter 1160: loss 1.3320, time 774.38ms, mfu 0.48%
iter 1170: loss 1.2922, time 771.34ms, mfu 0.48%
iter 1180: loss 1.3189, time 767.86ms, mfu 0.48%
iter 1190: loss 1.2716, time 765.73ms, mfu 0.48%
iter 1200: loss 1.2880, time 761.42ms, mfu 0.48%
iter 1210: loss 1.2660, time 761.21ms, mfu 0.48%
iter 1220: loss 1.3042, time 763.18ms, mfu 0.48%
iter 1230: loss 1.3000, time 762.07ms, mfu 0.48%
iter 1240: loss 1.2970, time 759.08ms, mfu 0.48%
step 1250: train loss 1.2056, val loss 1.4919
saving checkpoint to out-shakespeare-char
iter 1250: loss 1.2854, time 89016.63ms, mfu 0.44%
iter 1260: loss 1.2828, time 762.39ms, mfu 0.44%
iter 1270: loss 1.2730, time 761.89ms, mfu 0.45%
iter 1280: loss 1.2532, time 760.54ms, mfu 0.45%
iter 1290: loss 1.2937, time 764.28ms, mfu 0.45%
iter 1300: loss 1.3072, time 761.29ms, mfu 0.46%
iter 1310: loss 1.2386, time 761.80ms, mfu 0.46%
iter 1320: loss 1.2962, time 759.60ms, mfu 0.46%
iter 1330: loss 1.2620, time 760.74ms, mfu 0.47%
iter 1340: loss 1.3001, time 758.46ms, mfu 0.47%
iter 1350: loss 1.2603, time 759.78ms, mfu 0.47%
iter 1360: loss 1.2773, time 759.28ms, mfu 0.47%
iter 1370: loss 1.2600, time 760.70ms, mfu 0.47%
iter 1380: loss 1.2685, time 763.42ms, mfu 0.48%
iter 1390: loss 1.2481, time 756.01ms, mfu 0.48%
iter 1400: loss 1.2612, time 768.26ms, mfu 0.48%
iter 1410: loss 1.2509, time 762.87ms, mfu 0.48%
iter 1420: loss 1.2734, time 770.20ms, mfu 0.48%
iter 1430: loss 1.2452, time 761.12ms, mfu 0.48%
iter 1440: loss 1.2607, time 763.24ms, mfu 0.48%
iter 1450: loss 1.2301, time 758.78ms, mfu 0.48%
iter 1460: loss 1.2399, time 763.41ms, mfu 0.48%
iter 1470: loss 1.2191, time 761.96ms, mfu 0.48%
iter 1480: loss 1.2120, time 742.79ms, mfu 0.49%
iter 1490: loss 1.2405, time 757.44ms, mfu 0.49%
step 1500: train loss 1.1567, val loss 1.4852
saving checkpoint to out-shakespeare-char
iter 1500: loss 1.1835, time 89043.12ms, mfu 0.44%
iter 1510: loss 1.2307, time 769.64ms, mfu 0.44%
iter 1520: loss 1.2208, time 765.25ms, mfu 0.45%
iter 1530: loss 1.2543, time 762.49ms, mfu 0.45%
iter 1540: loss 1.1934, time 764.65ms, mfu 0.45%
iter 1550: loss 1.2308, time 761.64ms, mfu 0.46%
iter 1560: loss 1.2080, time 765.12ms, mfu 0.46%
iter 1570: loss 1.2361, time 765.18ms, mfu 0.46%
iter 1580: loss 1.2088, time 791.66ms, mfu 0.46%
iter 1590: loss 1.1936, time 768.87ms, mfu 0.47%
iter 1600: loss 1.1932, time 764.69ms, mfu 0.47%
iter 1610: loss 1.2386, time 763.72ms, mfu 0.47%
iter 1620: loss 1.1853, time 765.40ms, mfu 0.47%
iter 1630: loss 1.2056, time 766.69ms, mfu 0.47%
iter 1640: loss 1.2083, time 761.38ms, mfu 0.48%
iter 1650: loss 1.1762, time 765.19ms, mfu 0.48%
iter 1660: loss 1.2210, time 762.57ms, mfu 0.48%
iter 1670: loss 1.1914, time 763.61ms, mfu 0.48%
iter 1680: loss 1.2024, time 763.91ms, mfu 0.48%
iter 1690: loss 1.1998, time 764.72ms, mfu 0.48%
iter 1700: loss 1.1911, time 764.47ms, mfu 0.48%
iter 1710: loss 1.1795, time 766.98ms, mfu 0.48%
iter 1720: loss 1.1808, time 766.50ms, mfu 0.48%
iter 1730: loss 1.1974, time 763.45ms, mfu 0.48%
iter 1740: loss 1.1681, time 764.13ms, mfu 0.48%
step 1750: train loss 1.1025, val loss 1.4723
saving checkpoint to out-shakespeare-char
iter 1750: loss 1.1854, time 88905.94ms, mfu 0.44%
iter 1760: loss 1.1871, time 770.71ms, mfu 0.44%
iter 1770: loss 1.1981, time 789.39ms, mfu 0.44%
iter 1780: loss 1.1917, time 770.98ms, mfu 0.45%
iter 1790: loss 1.1928, time 766.66ms, mfu 0.45%
iter 1800: loss 1.1811, time 763.75ms, mfu 0.45%
iter 1810: loss 1.1672, time 762.26ms, mfu 0.46%
iter 1820: loss 1.1688, time 761.16ms, mfu 0.46%
iter 1830: loss 1.1612, time 762.49ms, mfu 0.46%
iter 1840: loss 1.1612, time 766.78ms, mfu 0.47%
iter 1850: loss 1.1611, time 776.98ms, mfu 0.47%
iter 1860: loss 1.1789, time 778.04ms, mfu 0.47%
iter 1870: loss 1.1398, time 766.83ms, mfu 0.47%
iter 1880: loss 1.1833, time 769.44ms, mfu 0.47%
iter 1890: loss 1.1780, time 769.55ms, mfu 0.47%
iter 1900: loss 1.1365, time 772.74ms, mfu 0.47%
iter 1910: loss 1.1647, time 762.08ms, mfu 0.48%
iter 1920: loss 1.1755, time 766.68ms, mfu 0.48%
iter 1930: loss 1.1427, time 764.07ms, mfu 0.48%
iter 1940: loss 1.1226, time 782.14ms, mfu 0.48%
iter 1950: loss 1.1380, time 766.50ms, mfu 0.48%
iter 1960: loss 1.1598, time 763.42ms, mfu 0.48%
iter 1970: loss 1.1502, time 763.73ms, mfu 0.48%
iter 1980: loss 1.1474, time 764.00ms, mfu 0.48%
iter 1990: loss 1.1485, time 762.12ms, mfu 0.48%
step 2000: train loss 1.0549, val loss 1.4811
iter 2000: loss 1.1275, time 88782.56ms, mfu 0.43%
iter 2010: loss 1.1348, time 768.47ms, mfu 0.44%
iter 2020: loss 1.1218, time 764.47ms, mfu 0.44%
iter 2030: loss 1.1530, time 765.56ms, mfu 0.45%
iter 2040: loss 1.1403, time 768.81ms, mfu 0.45%
iter 2050: loss 1.1144, time 764.75ms, mfu 0.46%
iter 2060: loss 1.1003, time 764.42ms, mfu 0.46%
iter 2070: loss 1.1220, time 766.73ms, mfu 0.46%
iter 2080: loss 1.1213, time 764.03ms, mfu 0.46%
iter 2090: loss 1.1297, time 767.40ms, mfu 0.47%
iter 2100: loss 1.1348, time 761.16ms, mfu 0.47%
iter 2110: loss 1.1257, time 763.90ms, mfu 0.47%
iter 2120: loss 1.1325, time 765.43ms, mfu 0.47%
iter 2130: loss 1.1356, time 768.53ms, mfu 0.47%
iter 2140: loss 1.1363, time 764.02ms, mfu 0.47%
iter 2150: loss 1.1213, time 764.28ms, mfu 0.48%
iter 2160: loss 1.1432, time 764.64ms, mfu 0.48%
iter 2170: loss 1.1355, time 761.73ms, mfu 0.48%
iter 2180: loss 1.1126, time 766.24ms, mfu 0.48%
iter 2190: loss 1.1065, time 765.54ms, mfu 0.48%
iter 2200: loss 1.1265, time 763.92ms, mfu 0.48%
iter 2210: loss 1.1084, time 764.75ms, mfu 0.48%
iter 2220: loss 1.1220, time 768.67ms, mfu 0.48%
iter 2230: loss 1.1189, time 768.14ms, mfu 0.48%
iter 2240: loss 1.1290, time 766.11ms, mfu 0.48%
step 2250: train loss 1.0101, val loss 1.4883
iter 2250: loss 1.1069, time 89221.30ms, mfu 0.43%
iter 2260: loss 1.1022, time 772.16ms, mfu 0.44%
iter 2270: loss 1.1337, time 769.44ms, mfu 0.44%
iter 2280: loss 1.0995, time 763.04ms, mfu 0.45%
iter 2290: loss 1.1357, time 767.49ms, mfu 0.45%
iter 2300: loss 1.1214, time 765.23ms, mfu 0.46%
iter 2310: loss 1.0972, time 768.72ms, mfu 0.46%
iter 2320: loss 1.0882, time 764.15ms, mfu 0.46%
iter 2330: loss 1.0942, time 766.80ms, mfu 0.46%
iter 2340: loss 1.1218, time 766.91ms, mfu 0.47%
iter 2350: loss 1.1028, time 765.39ms, mfu 0.47%
iter 2360: loss 1.1030, time 761.76ms, mfu 0.47%
iter 2370: loss 1.0892, time 760.61ms, mfu 0.47%
iter 2380: loss 1.0853, time 763.05ms, mfu 0.47%
iter 2390: loss 1.0760, time 761.86ms, mfu 0.48%
iter 2400: loss 1.0834, time 763.93ms, mfu 0.48%
iter 2410: loss 1.0692, time 764.55ms, mfu 0.48%
iter 2420: loss 1.0769, time 764.58ms, mfu 0.48%
iter 2430: loss 1.0545, time 763.06ms, mfu 0.48%
iter 2440: loss 1.0538, time 764.51ms, mfu 0.48%
iter 2450: loss 1.0685, time 764.81ms, mfu 0.48%
iter 2460: loss 1.0883, time 764.05ms, mfu 0.48%
iter 2470: loss 1.0889, time 761.43ms, mfu 0.48%
iter 2480: loss 1.0783, time 770.91ms, mfu 0.48%
iter 2490: loss 1.0619, time 764.06ms, mfu 0.48%
step 2500: train loss 0.9583, val loss 1.4988
iter 2500: loss 1.0838, time 88814.25ms, mfu 0.44%
iter 2510: loss 1.0697, time 792.84ms, mfu 0.44%
iter 2520: loss 1.0461, time 767.25ms, mfu 0.44%
iter 2530: loss 1.0497, time 763.83ms, mfu 0.45%
iter 2540: loss 1.0460, time 765.56ms, mfu 0.45%
iter 2550: loss 1.0731, time 770.39ms, mfu 0.45%
iter 2560: loss 1.0571, time 765.38ms, mfu 0.46%
iter 2570: loss 1.0762, time 766.11ms, mfu 0.46%
iter 2580: loss 1.0788, time 768.30ms, mfu 0.46%
iter 2590: loss 1.0689, time 762.82ms, mfu 0.47%
iter 2600: loss 1.0664, time 769.98ms, mfu 0.47%
iter 2610: loss 1.0501, time 762.06ms, mfu 0.47%
iter 2620: loss 1.0419, time 765.89ms, mfu 0.47%
iter 2630: loss 1.0177, time 769.23ms, mfu 0.47%
iter 2640: loss 1.0490, time 761.70ms, mfu 0.47%
iter 2650: loss 1.0594, time 765.34ms, mfu 0.48%
iter 2660: loss 1.0406, time 763.12ms, mfu 0.48%
iter 2670: loss 1.0136, time 769.97ms, mfu 0.48%
iter 2680: loss 1.0467, time 764.77ms, mfu 0.48%
iter 2690: loss 1.0531, time 763.14ms, mfu 0.48%
iter 2700: loss 1.0206, time 763.33ms, mfu 0.48%
iter 2710: loss 1.0448, time 763.77ms, mfu 0.48%
iter 2720: loss 1.0513, time 766.34ms, mfu 0.48%
iter 2730: loss 1.0645, time 761.04ms, mfu 0.48%
iter 2740: loss 1.0341, time 760.09ms, mfu 0.48%
step 2750: train loss 0.9113, val loss 1.5231
iter 2750: loss 1.0294, time 88750.78ms, mfu 0.44%
iter 2760: loss 1.0266, time 770.24ms, mfu 0.44%
iter 2770: loss 1.0236, time 776.24ms, mfu 0.44%
iter 2780: loss 1.0195, time 770.29ms, mfu 0.45%
iter 2790: loss 1.0363, time 773.80ms, mfu 0.45%
iter 2800: loss 1.0032, time 775.78ms, mfu 0.45%
iter 2810: loss 1.0393, time 765.77ms, mfu 0.46%
iter 2820: loss 1.0250, time 759.64ms, mfu 0.46%
iter 2830: loss 1.0281, time 763.72ms, mfu 0.46%
iter 2840: loss 0.9877, time 766.91ms, mfu 0.47%
iter 2850: loss 1.0251, time 768.34ms, mfu 0.47%
iter 2860: loss 1.0272, time 761.83ms, mfu 0.47%
iter 2870: loss 0.9962, time 818.67ms, mfu 0.47%
iter 2880: loss 1.0371, time 768.51ms, mfu 0.47%
iter 2890: loss 0.9991, time 764.55ms, mfu 0.47%
iter 2900: loss 0.9856, time 759.84ms, mfu 0.47%
iter 2910: loss 1.0297, time 760.32ms, mfu 0.48%
iter 2920: loss 1.0099, time 762.98ms, mfu 0.48%
iter 2930: loss 0.9962, time 763.19ms, mfu 0.48%
iter 2940: loss 0.9927, time 762.32ms, mfu 0.48%
iter 2950: loss 1.0174, time 760.57ms, mfu 0.48%
iter 2960: loss 0.9995, time 760.98ms, mfu 0.48%
iter 2970: loss 0.9914, time 763.70ms, mfu 0.48%
iter 2980: loss 0.9968, time 760.66ms, mfu 0.48%
iter 2990: loss 0.9719, time 758.40ms, mfu 0.48%
step 3000: train loss 0.8658, val loss 1.5323
iter 3000: loss 0.9881, time 89544.40ms, mfu 0.44%
iter 3010: loss 0.9981, time 766.32ms, mfu 0.44%
iter 3020: loss 0.9965, time 765.43ms, mfu 0.45%
iter 3030: loss 0.9990, time 770.22ms, mfu 0.45%
iter 3040: loss 1.0214, time 763.25ms, mfu 0.45%
iter 3050: loss 0.9768, time 762.89ms, mfu 0.46%
iter 3060: loss 0.9970, time 762.83ms, mfu 0.46%
iter 3070: loss 1.0053, time 762.93ms, mfu 0.46%
iter 3080: loss 0.9953, time 764.46ms, mfu 0.47%
iter 3090: loss 0.9832, time 764.52ms, mfu 0.47%
iter 3100: loss 0.9933, time 787.22ms, mfu 0.47%
iter 3110: loss 0.9714, time 764.52ms, mfu 0.47%
iter 3120: loss 0.9948, time 763.31ms, mfu 0.47%
iter 3130: loss 0.9790, time 760.68ms, mfu 0.47%
iter 3140: loss 0.9779, time 766.33ms, mfu 0.47%
iter 3150: loss 0.9930, time 762.99ms, mfu 0.48%
iter 3160: loss 1.0104, time 762.74ms, mfu 0.48%
iter 3170: loss 0.9615, time 765.08ms, mfu 0.48%
iter 3180: loss 0.9744, time 764.29ms, mfu 0.48%
iter 3190: loss 0.9932, time 766.75ms, mfu 0.48%
iter 3200: loss 0.9627, time 764.33ms, mfu 0.48%
iter 3210: loss 0.9607, time 761.90ms, mfu 0.48%
iter 3220: loss 0.9607, time 765.40ms, mfu 0.48%
iter 3230: loss 0.9506, time 764.91ms, mfu 0.48%
iter 3240: loss 0.9626, time 765.50ms, mfu 0.48%
step 3250: train loss 0.8192, val loss 1.5645
iter 3250: loss 0.9639, time 94706.32ms, mfu 0.44%
iter 3260: loss 0.9562, time 797.48ms, mfu 0.44%
iter 3270: loss 0.9703, time 816.05ms, mfu 0.44%
iter 3280: loss 0.9486, time 789.35ms, mfu 0.44%
iter 3290: loss 0.9381, time 780.74ms, mfu 0.45%
iter 3300: loss 0.9351, time 775.87ms, mfu 0.45%
iter 3310: loss 0.9564, time 782.38ms, mfu 0.45%
iter 3320: loss 0.9575, time 779.06ms, mfu 0.46%
iter 3330: loss 0.9565, time 781.94ms, mfu 0.46%
iter 3340: loss 0.9481, time 779.68ms, mfu 0.46%
iter 3350: loss 0.9482, time 774.10ms, mfu 0.46%
iter 3360: loss 0.9252, time 769.16ms, mfu 0.46%
iter 3370: loss 0.9502, time 790.72ms, mfu 0.46%
iter 3380: loss 0.9482, time 859.93ms, mfu 0.46%
iter 3390: loss 0.9468, time 785.12ms, mfu 0.46%
iter 3400: loss 0.9541, time 837.16ms, mfu 0.46%
iter 3410: loss 0.9357, time 774.25ms, mfu 0.46%
iter 3420: loss 0.9455, time 773.59ms, mfu 0.46%
iter 3430: loss 0.9423, time 765.35ms, mfu 0.47%
iter 3440: loss 0.9737, time 792.23ms, mfu 0.47%
iter 3450: loss 0.9550, time 776.90ms, mfu 0.47%
iter 3460: loss 0.9439, time 766.39ms, mfu 0.47%
iter 3470: loss 0.9385, time 766.46ms, mfu 0.47%
iter 3480: loss 0.9434, time 766.61ms, mfu 0.47%
iter 3490: loss 0.9157, time 770.34ms, mfu 0.47%
step 3500: train loss 0.7781, val loss 1.5769
iter 3500: loss 0.9019, time 90869.83ms, mfu 0.43%
iter 3510: loss 0.9144, time 780.58ms, mfu 0.43%
iter 3520: loss 0.9174, time 775.75ms, mfu 0.44%
iter 3530: loss 0.9494, time 764.67ms, mfu 0.44%
iter 3540: loss 0.9288, time 780.71ms, mfu 0.45%
iter 3550: loss 0.9227, time 769.29ms, mfu 0.45%
iter 3560: loss 0.9469, time 767.41ms, mfu 0.45%
iter 3570: loss 0.9340, time 771.33ms, mfu 0.46%
iter 3580: loss 0.9337, time 797.18ms, mfu 0.46%
iter 3590: loss 0.9256, time 766.42ms, mfu 0.46%
iter 3600: loss 0.9235, time 771.00ms, mfu 0.46%
iter 3610: loss 0.9060, time 771.09ms, mfu 0.46%
iter 3620: loss 0.9090, time 767.89ms, mfu 0.47%
iter 3630: loss 0.9236, time 770.49ms, mfu 0.47%
iter 3640: loss 0.9057, time 770.61ms, mfu 0.47%
iter 3650: loss 0.9153, time 765.55ms, mfu 0.47%
iter 3660: loss 0.9407, time 805.50ms, mfu 0.47%
iter 3670: loss 0.9281, time 770.75ms, mfu 0.47%
iter 3680: loss 0.9055, time 769.85ms, mfu 0.47%
iter 3690: loss 0.9294, time 780.05ms, mfu 0.47%
iter 3700: loss 0.8663, time 808.42ms, mfu 0.47%
iter 3710: loss 0.8763, time 798.28ms, mfu 0.47%
iter 3720: loss 0.9011, time 824.18ms, mfu 0.47%
iter 3730: loss 0.9039, time 775.43ms, mfu 0.47%
iter 3740: loss 0.8997, time 791.39ms, mfu 0.47%
step 3750: train loss 0.7360, val loss 1.6074
iter 3750: loss 0.8986, time 91971.63ms, mfu 0.42%
iter 3760: loss 0.9264, time 821.35ms, mfu 0.43%
iter 3770: loss 0.9320, time 777.67ms, mfu 0.43%
iter 3780: loss 0.9129, time 844.16ms, mfu 0.43%
iter 3790: loss 0.8908, time 774.95ms, mfu 0.44%
iter 3800: loss 0.9085, time 781.01ms, mfu 0.44%
iter 3810: loss 0.9074, time 789.81ms, mfu 0.44%
iter 3820: loss 0.8872, time 767.95ms, mfu 0.45%
iter 3830: loss 0.9014, time 787.62ms, mfu 0.45%
iter 3840: loss 0.8816, time 807.49ms, mfu 0.45%
iter 3850: loss 0.8778, time 800.56ms, mfu 0.45%
iter 3860: loss 0.8761, time 774.37ms, mfu 0.46%
iter 3870: loss 0.8868, time 765.38ms, mfu 0.46%
iter 3880: loss 0.8816, time 791.02ms, mfu 0.46%
iter 3890: loss 0.8937, time 767.78ms, mfu 0.46%
iter 3900: loss 0.8883, time 764.94ms, mfu 0.47%
iter 3910: loss 0.8821, time 780.19ms, mfu 0.47%
iter 3920: loss 0.8617, time 766.48ms, mfu 0.47%
iter 3930: loss 0.8906, time 781.94ms, mfu 0.47%
iter 3940: loss 0.8762, time 768.42ms, mfu 0.47%
iter 3950: loss 0.8806, time 762.94ms, mfu 0.47%
iter 3960: loss 0.9053, time 763.33ms, mfu 0.47%
iter 3970: loss 0.8929, time 763.68ms, mfu 0.48%
iter 3980: loss 0.9000, time 766.02ms, mfu 0.48%
iter 3990: loss 0.8738, time 764.78ms, mfu 0.48%
step 4000: train loss 0.7034, val loss 1.6342
iter 4000: loss 0.8590, time 89364.17ms, mfu 0.43%
iter 4010: loss 0.8781, time 764.38ms, mfu 0.44%
iter 4020: loss 0.8816, time 764.92ms, mfu 0.44%
iter 4030: loss 0.8774, time 764.12ms, mfu 0.45%
iter 4040: loss 0.8792, time 765.87ms, mfu 0.45%
iter 4050: loss 0.8731, time 764.40ms, mfu 0.45%
iter 4060: loss 0.8570, time 765.33ms, mfu 0.46%
iter 4070: loss 0.8556, time 765.59ms, mfu 0.46%
iter 4080: loss 0.8760, time 766.35ms, mfu 0.46%
iter 4090: loss 0.8507, time 765.01ms, mfu 0.47%
iter 4100: loss 0.9000, time 762.78ms, mfu 0.47%
iter 4110: loss 0.8646, time 764.20ms, mfu 0.47%
iter 4120: loss 0.8727, time 765.67ms, mfu 0.47%
iter 4130: loss 0.8542, time 766.51ms, mfu 0.47%
iter 4140: loss 0.8699, time 781.01ms, mfu 0.47%
iter 4150: loss 0.8724, time 770.61ms, mfu 0.47%
iter 4160: loss 0.8563, time 773.55ms, mfu 0.47%
iter 4170: loss 0.8696, time 767.49ms, mfu 0.48%
iter 4180: loss 0.8666, time 776.46ms, mfu 0.48%
iter 4190: loss 0.8732, time 766.99ms, mfu 0.48%
iter 4200: loss 0.8503, time 763.08ms, mfu 0.48%
iter 4210: loss 0.8701, time 765.49ms, mfu 0.48%
iter 4220: loss 0.8592, time 769.91ms, mfu 0.48%
iter 4230: loss 0.8744, time 766.23ms, mfu 0.48%
iter 4240: loss 0.8678, time 899.85ms, mfu 0.47%
step 4250: train loss 0.6746, val loss 1.6593
iter 4250: loss 0.8587, time 90109.97ms, mfu 0.43%
iter 4260: loss 0.8589, time 772.54ms, mfu 0.43%
iter 4270: loss 0.8615, time 769.50ms, mfu 0.44%
iter 4280: loss 0.8513, time 767.60ms, mfu 0.44%
iter 4290: loss 0.8299, time 767.44ms, mfu 0.45%
iter 4300: loss 0.8264, time 769.06ms, mfu 0.45%
iter 4310: loss 0.8550, time 766.96ms, mfu 0.45%
iter 4320: loss 0.8405, time 766.16ms, mfu 0.46%
iter 4330: loss 0.8516, time 767.95ms, mfu 0.46%
iter 4340: loss 0.8313, time 769.18ms, mfu 0.46%
iter 4350: loss 0.8365, time 765.96ms, mfu 0.46%
iter 4360: loss 0.8638, time 764.79ms, mfu 0.47%
iter 4370: loss 0.8499, time 774.58ms, mfu 0.47%
iter 4380: loss 0.8397, time 768.31ms, mfu 0.47%
iter 4390: loss 0.8625, time 771.60ms, mfu 0.47%
iter 4400: loss 0.8457, time 765.52ms, mfu 0.47%
iter 4410: loss 0.8644, time 767.67ms, mfu 0.47%
iter 4420: loss 0.8560, time 770.97ms, mfu 0.48%
iter 4430: loss 0.8458, time 768.76ms, mfu 0.48%
iter 4440: loss 0.8395, time 769.60ms, mfu 0.48%
iter 4450: loss 0.8420, time 775.74ms, mfu 0.48%
iter 4460: loss 0.8298, time 766.77ms, mfu 0.48%
iter 4470: loss 0.8444, time 767.64ms, mfu 0.48%
iter 4480: loss 0.8243, time 766.65ms, mfu 0.48%
iter 4490: loss 0.8395, time 766.99ms, mfu 0.48%
step 4500: train loss 0.6482, val loss 1.6789
iter 4500: loss 0.8548, time 89243.32ms, mfu 0.43%
iter 4510: loss 0.8535, time 776.49ms, mfu 0.44%
iter 4520: loss 0.8343, time 786.44ms, mfu 0.44%
iter 4530: loss 0.8378, time 767.57ms, mfu 0.45%
iter 4540: loss 0.8394, time 761.96ms, mfu 0.45%
iter 4550: loss 0.8616, time 679.65ms, mfu 0.46%
iter 4560: loss 0.8336, time 819.33ms, mfu 0.46%
iter 4570: loss 0.8451, time 775.99ms, mfu 0.46%
iter 4580: loss 0.8510, time 821.07ms, mfu 0.46%
iter 4590: loss 0.8454, time 771.13ms, mfu 0.46%
iter 4600: loss 0.8206, time 771.36ms, mfu 0.46%
iter 4610: loss 0.8557, time 770.63ms, mfu 0.47%
iter 4620: loss 0.8274, time 772.03ms, mfu 0.47%
iter 4630: loss 0.8253, time 768.52ms, mfu 0.47%
iter 4640: loss 0.8383, time 768.48ms, mfu 0.47%
iter 4650: loss 0.8562, time 767.76ms, mfu 0.47%
iter 4660: loss 0.8434, time 770.06ms, mfu 0.47%
iter 4670: loss 0.8397, time 768.85ms, mfu 0.47%
iter 4680: loss 0.8431, time 768.97ms, mfu 0.48%
iter 4690: loss 0.8355, time 770.27ms, mfu 0.48%
iter 4700: loss 0.8233, time 763.08ms, mfu 0.48%
iter 4710: loss 0.7875, time 773.54ms, mfu 0.48%
iter 4720: loss 0.8362, time 766.37ms, mfu 0.48%
iter 4730: loss 0.8192, time 770.49ms, mfu 0.48%
iter 4740: loss 0.8264, time 769.96ms, mfu 0.48%
step 4750: train loss 0.6307, val loss 1.6910
iter 4750: loss 0.7958, time 92867.44ms, mfu 0.43%
iter 4760: loss 0.8178, time 807.11ms, mfu 0.44%
iter 4770: loss 0.7898, time 783.59ms, mfu 0.44%
iter 4780: loss 0.8074, time 831.81ms, mfu 0.44%
iter 4790: loss 0.8215, time 807.41ms, mfu 0.44%
iter 4800: loss 0.8141, time 840.90ms, mfu 0.44%
iter 4810: loss 0.8376, time 781.90ms, mfu 0.45%
iter 4820: loss 0.8210, time 784.68ms, mfu 0.45%
iter 4830: loss 0.8109, time 779.31ms, mfu 0.45%
iter 4840: loss 0.8309, time 917.62ms, mfu 0.45%
iter 4850: loss 0.8132, time 837.24ms, mfu 0.45%
iter 4860: loss 0.8163, time 809.85ms, mfu 0.45%
iter 4870: loss 0.8047, time 800.44ms, mfu 0.45%
iter 4880: loss 0.8313, time 776.69ms, mfu 0.45%
iter 4890: loss 0.8094, time 778.94ms, mfu 0.46%
iter 4900: loss 0.8066, time 780.65ms, mfu 0.46%
iter 4910: loss 0.8279, time 776.43ms, mfu 0.46%
iter 4920: loss 0.8152, time 781.76ms, mfu 0.46%
iter 4930: loss 0.8064, time 819.07ms, mfu 0.46%
iter 4940: loss 0.8013, time 806.60ms, mfu 0.46%
iter 4950: loss 0.8226, time 768.95ms, mfu 0.46%
iter 4960: loss 0.8119, time 780.67ms, mfu 0.46%
iter 4970: loss 0.7848, time 767.88ms, mfu 0.47%
iter 4980: loss 0.7990, time 768.50ms, mfu 0.47%
iter 4990: loss 0.8228, time 765.76ms, mfu 0.47%
step 5000: train loss 0.6175, val loss 1.7093
iter 5000: loss 0.8108, time 89300.71ms, mfu 0.42%
```

## Sampling

```bash
uv run sample.py --out_dir=out-shakespeare-char
```

```bash
Overriding: out_dir = out-shakespeare-char
./nanoGPT/.venv/lib/python3.13/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/Context.cpp:85.)
  self.setter(val)
number of parameters: 10.65M
Loading meta from data/shakespeare_char/meta.pkl...

Three came I true!

LADY CAPULET:
Go, she's a woman, what Henry,
She call here is there?

CAPULET:
Not a traitor: no flier, but we may all.

LADY CAPULET:
I have worthy throne, he will follow him,
That have shed my letter of her eyes.

LADY CAPULET:
How would you have it for me?

CAPULET:
It is it not a great matter; for now we would have her hand,
For this profane show the myouth of that dog
Have now in her face and poor heaven'd doth off.

CAPULET:
I am a good to the moon: or this is it,
Give 
---------------

DUKE VINCENTIO:
Come, sir, till my body we have forgot thee
As I change him on his majesty.

DUKE VINCENTIO:
Why, rather he are not hood to save the better
Than two best? Thou hast ever done; but so accused
That thou art power to know for new this.

LUCIO:
Why, Any more is thy mind my company to Romeo?

DUKE VINCENTIO:
Nor your wombning after than they have song
To such find the time of your purpose.

LUCIO:
How may you that they are done?

DUKE VINCENTIO:
How doth you not speak for us from hear
---------------

All the jealous moans--then it shall be contented.

First Lady:
No, yet why, remains the royalty of my heart,
That she wakes not made me part here again.

WARWICK:
Nay, like a service of so come fair;
When I am grieved with his afternoon,
Who would she disdresses in Montague,
Where we have done to her wash in the blood
With from the policy roar's heart;
To swear be the souls of Clifford that here was here;
For then we help too for me; which he is this not the world
That was once wash'd to all th
---------------


GLOUCESTER:
Ah, what should you have prop to't?

GLOUCESTER:
Romeo, fair Berkeley, belike the law of Clarence;
And send towards Clarence by Brake and Greatesby.

BUCKINGHAM:
Ha! what happy to proclaim; here's Clarence?

DUKE OF YORK:
His widow, and welcome hither is so her.

LADY CAPULET:
A salute-foolity soldiers; or the bloody time,
Which he hath seen the seaty against of arms:
I am done with us too more expossible,
To the white of our place with child, that thou mayst live,
And brought the s
---------------


ISABELLA:
Ay, a man this, a doth instruct him with his
bear and this outward: let Has it poor
manifests to be malice, that I must know his face
on the like other's natural death.

DUKE VINCENTIO:
Had a milker sir, distill'd the air course,
A bastard of the valour, and place your part
To the dribunes, but that you do too much love,
The other and making to his shadows do him:
Yet offer to our mistrusty
As high as your infections, I would you,
Lest not speak of a gentleman's brother
Worldly to unm
---------------

Look that I have done to show him and orvid my hell,
And all to my fresh as I have wound
By myself will weak what I will stand.
But I command thou wast before my son:
Do when I must fear, do it be commanded.
In thy daughters and princely Margaret,
Some two more with than true rude in the son,
And makes the oracle Harry for her court coursed than I would weeds,
As of thy beaams are a kingdom, and go not to to my tongue.

HASTINGS:
I take it with our gracesters, being eyes,
To make my solemnity to
---------------

IEL:
There is the orator; this is not a while:

LARTIUS:
I shall
A glory of the people hails of the people,
And melling, but the braining citizens,
Our small change them have been deven to hear,
That e'er what we do read of men
Both the take precious in him: see how he is
His ear brought and with a bloody place
Hath no faults rest, he was he with the one.

RICHARD:
He dares but for kill the neck is false banishment.

RICHARD:
We now the treasure no grief; and, expedient with splutter;
For, now a
---------------

Marcius, I go, having he loved my noble fear:
What's the people, as the wisdom of his feet,
And cannot speak against my jest,
Good my pabe in the sun.

CORIOLANUS:
There's no more.

AUFIDIUS:
The matter of our temperation.

First Senator:
Not for the market-place.

Second Servingman:
Who, if you have heard rained him from the world
That debts him to make him of his treason, he
'lian but him once: the hand heaven too far the wisdom,
Of his life or some souls and will not do't. Now these bids
Of w
---------------

BIUTUS:
I will be gone.

MENENIUS:
Not, no.

Citizens:
Even nor Corioli, the boy
Your consent care for a pitiful.

CORIOLANUS:
He had none of breeding; but were you.

BRUTUS:
So, this is not almost interprivellane, to bear
The point of noble pleasure.

MENENIUS:
Have you joy, daughter ours to the place.

CORIOLANUS:
I am a cause whom the people, poor 'battle; a
more good added man special whose mocks here out the stock.

CORIOLANUS:
This is the gods good devise stand on the maid of fellow.

VIRG
---------------


VIRGILIA:
Go, but you are the gods and a lady's man
And unwill be a monthless to seem; if you be he were
not say 'twere she was brief for a changed soul.

CORIOLANUS:
What arm you that do we beseem ready?

CORIOLANUS:
Ay, have you to me?

CORIOLANUS:
I talk o' the good sir, he that was a fire?

AUFIDIUS:

MENENIUS:
Say, I have it profound to the people, and there it
was the gods slain with virtuous singled friends,
As I were little no satisfied to charge you,
Have a league and more of your peop
```