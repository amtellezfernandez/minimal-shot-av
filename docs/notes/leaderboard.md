# Waymo E2E Driving Challenge Leaderboard

> Source note: this table was supplied manually by the project owner. The public challenge page at <https://waymo.com/open/challenges/2025/e2e-driving/> exposes challenge text and metrics through the current fetch path, but the live leaderboard rows were not accessible from the fetched page.

Sorted by RFS Overall ascending. All metrics are RFS, where higher is better, except ADE, where lower is better. Top-tier rows are bolded where RFS Overall is at least 7.9.

| Method | RFS Overall | ADE 5s↓ | ADE 3s↓ | Spotlight | Construction | Intersection | Pedestrian | Cyclist | Multi-lane | Single-lane | Cut-ins | FOD | Spec. Vehicles | Others |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GAT-Transformer-LSTM | 4.0659 | 29.2741 | 18.6010 | 4.0664 | 4.2428 | 4.1216 | 4.0000 | 4.0347 | 4.0248 | 4.0000 | 4.0668 | 4.1438 | 4.0238 | 4.0000 |
| magicVLM | 4.6243 | 13.3279 | 8.8614 | 4.4498 | 4.3523 | 4.5227 | 4.7186 | 4.7015 | 4.7686 | 4.8152 | 4.7628 | 4.4063 | 4.7221 | 4.6474 |
| ResTransMLP | 4.8955 | 7.7968 | 4.1559 | 4.7663 | 4.5573 | 4.6187 | 5.0286 | 5.2617 | 4.4093 | 5.5463 | 5.0358 | 4.9533 | 5.0873 | 4.5860 |
| OpenEMMA | 5.1575 | 12.4755 | 6.6842 | 4.7131 | 4.6574 | 5.4567 | 5.6003 | 4.9514 | 5.2768 | 5.3269 | 5.1392 | 4.6742 | 5.6639 | 5.2727 |
| ViT-GRU with Modality Fusion | 5.3000 | 12.7350 | 7.5876 | 4.6991 | 4.9490 | 5.7059 | 5.6801 | 5.2028 | 5.4810 | 5.7028 | 5.0668 | 4.8020 | 5.7123 | 5.2981 |
| AgentDrive RouterVLM | 6.0679 | 6.4411 | 3.7727 | 5.1517 | 6.3768 | 6.3019 | 6.3575 | 5.8957 | 6.0994 | 6.2941 | 6.1757 | 5.9466 | 6.4811 | 5.6664 |
| Baseline | 6.2365 | 3.6104 | 1.5713 | 5.4946 | 6.7708 | 6.1546 | 6.3147 | 6.3929 | 5.8628 | 6.3946 | 6.6947 | 6.6364 | 6.3368 | 5.5486 |
| OpenEMMA w/ QwenVLM | 6.3162 | 4.1078 | 1.8663 | 5.5054 | 6.3312 | 6.7374 | 6.6844 | 5.9696 | 6.1858 | 6.5395 | 6.4842 | 6.0835 | 6.7792 | 6.1786 |
| FrozenResNet50 | 6.4177 | 4.2010 | 2.0871 | 6.1057 | 5.7196 | 6.3238 | 6.8007 | 6.7018 | 6.0340 | 7.4715 | 6.8725 | 6.1958 | 6.6946 | 5.6750 |
| E2E Bottleneck Scaling 250 | 6.4674 | 3.9941 | 1.7252 | 5.0525 | 7.1040 | 6.4797 | 6.7374 | 6.5615 | 6.4576 | 6.9367 | 6.7319 | 6.9426 | 6.4063 | 5.7315 |
| FrozenResNet50 | 6.4719 | 3.9148 | 1.9446 | 5.7977 | 5.9802 | 6.5245 | 6.6199 | 6.7997 | 6.0212 | 7.3880 | 6.9099 | 6.4508 | 6.7443 | 5.9553 |
| LightEMMA | 6.5169 | 3.7395 | 1.7052 | 5.7103 | 7.1768 | 6.5039 | 6.3576 | 6.5233 | 6.2180 | 7.2659 | 6.7744 | 7.0588 | 6.3521 | 5.7453 |
| Poutine-Concordia-v4 | 6.5384 | 6.8811 | 3.6696 | 5.7702 | 6.4620 | 6.7749 | 6.9227 | 6.3982 | 6.5523 | 7.0484 | 6.4581 | 6.0508 | 7.1928 | 6.2917 |
| ViT-GRU with Modality Fusion | 6.7425 | 3.2819 | 1.4491 | 5.6726 | 7.5686 | 7.2661 | 6.8617 | 6.7284 | 6.5956 | 6.8024 | 6.8430 | 6.9288 | 6.7916 | 6.1087 |
| ResTransMLP | 6.8440 | 3.7425 | 2.2075 | 5.9202 | 7.3945 | 7.0557 | 7.0417 | 6.9795 | 6.5297 | 7.0732 | 7.2192 | 6.9753 | 6.7670 | 6.3278 |
| CTL-Drive | 6.9843 | 4.3040 | 1.9722 | 5.8676 | 7.2170 | 7.2067 | 7.3233 | 6.9470 | 6.8361 | 7.3808 | 7.1196 | 6.7800 | 7.6391 | 6.5095 |
| CTL-Drive | 6.9969 | 4.2816 | 1.9614 | 5.8980 | 7.2170 | 7.2108 | 7.3295 | 6.9858 | 6.8361 | 7.3808 | 7.1196 | 6.7972 | 7.6391 | 6.5515 |
| MultiModalMamba FULL | 7.0321 | 2.9948 | 1.3145 | 5.5549 | 7.4487 | 7.2043 | 7.1324 | 7.2977 | 6.9092 | 7.8004 | 7.2263 | 7.3803 | 7.1068 | 6.2925 |
| MultiModalMamba FULL | 7.0394 | 2.9960 | 1.3153 | 5.4879 | 7.4748 | 7.2117 | 7.1846 | 7.2864 | 6.9269 | 7.7458 | 7.2832 | 7.3934 | 7.1352 | 6.3039 |
| WayPredict | 7.0641 | 3.5779 | 1.7242 | 5.8562 | 7.4834 | 7.5093 | 7.2743 | 6.8143 | 6.8874 | 7.6518 | 6.9136 | 7.2653 | 7.4440 | 6.6051 |
| ViT-GRU with Modality Fusion | 7.1146 | 3.2102 | 1.4959 | 5.8525 | 7.7763 | 7.3731 | 7.2874 | 7.0467 | 6.7046 | 7.6051 | 7.0391 | 7.3255 | 7.2693 | 6.9807 |
| MultiModal-Transformer-ViT | 7.1651 | 3.2548 | 1.4304 | 6.3154 | 7.6868 | 7.4397 | 7.3129 | 7.2673 | 6.8470 | 7.4929 | 7.1386 | 7.3372 | 7.3614 | 6.6172 |
| MTR-VP | 7.2020 | 3.0941 | 1.3370 | 6.3275 | 7.4720 | 7.4928 | 7.2745 | 7.4339 | 7.1306 | 7.6309 | 7.4163 | 7.3676 | 7.3062 | 6.3693 |
| PredNet | 7.2316 | 2.9130 | 1.2453 | 6.0765 | 7.8144 | 7.3564 | 7.3328 | 7.1789 | 7.1813 | 7.6562 | 7.3651 | 7.5299 | 7.3288 | 6.7270 |
| ViT-GRU with Modality Fusion | 7.2335 | 3.0808 | 1.3926 | 5.9981 | 7.7636 | 7.5276 | 7.5794 | 7.1433 | 7.1263 | 7.8486 | 7.1603 | 7.4625 | 7.2438 | 6.7151 |
| gemma-2-2b-it | 7.2335 | 3.0808 | 1.3926 | 5.9981 | 7.7636 | 7.5276 | 7.5794 | 7.1433 | 7.1263 | 7.8486 | 7.1603 | 7.4625 | 7.2438 | 6.7151 |
| WayPredict-XL | 7.2678 | 3.2370 | 1.4219 | 6.0362 | 7.9728 | 7.5042 | 7.3671 | 7.1296 | 7.1819 | 7.6920 | 7.1393 | 7.4103 | 7.6988 | 6.8139 |
| WayPredict-XL | 7.2922 | 3.2915 | 1.4385 | 6.3083 | 7.8151 | 7.5726 | 7.4202 | 7.2268 | 7.1423 | 7.9033 | 7.1534 | 7.2776 | 7.7262 | 6.6681 |
| zx-method | 7.3034 | 3.1164 | 1.3953 | 6.3970 | 7.9183 | 7.4826 | 7.4161 | 7.2112 | 7.0960 | 7.8717 | 6.9939 | 7.5167 | 7.6952 | 6.7390 |
| MTR-VP | 7.3433 | 3.3485 | 1.4232 | 6.4023 | 7.7111 | 7.6864 | 7.5096 | 7.3138 | 7.2205 | 7.7170 | 7.4170 | 7.3794 | 7.5705 | 6.8492 |
| WayNet | 7.3463 | 2.7977 | 1.2538 | 5.9839 | 7.7444 | 7.5033 | 7.4382 | 7.5305 | 7.2337 | 8.0570 | 7.5527 | 7.7446 | 6.9908 | 7.0302 |
| gemma-2-9b | 7.3635 | 3.0671 | 1.3777 | 5.9239 | 7.9485 | 7.5925 | 7.7056 | 7.2948 | 7.2019 | 8.0365 | 7.2839 | 7.5339 | 7.3792 | 7.0981 |
| WayPredict-XL | 7.3775 | 3.2209 | 1.3622 | 6.1594 | 7.8138 | 7.6323 | 7.4679 | 7.2641 | 7.3424 | 7.9003 | 7.2785 | 7.5350 | 7.8279 | 6.9310 |
| WayPredict-XL | 7.4128 | 3.0848 | 1.3324 | 6.1312 | 8.1471 | 7.6513 | 7.6090 | 7.3450 | 7.2268 | 7.9227 | 7.3815 | 7.5914 | 7.5508 | 6.9840 |
| zx-method | 7.4214 | 3.0414 | 1.3263 | 6.2762 | 8.0673 | 7.5790 | 7.5396 | 7.3842 | 7.1001 | 7.9087 | 7.3757 | 7.5539 | 7.8953 | 6.9553 |
| open-llama | 7.4288 | 3.2165 | 1.3140 | 6.2510 | 7.9137 | 7.7453 | 7.5224 | 7.3280 | 7.2860 | 7.8612 | 7.4274 | 7.6888 | 7.7920 | 6.9015 |
| DrivePI | 7.4526 | 2.9031 | 1.3179 | 6.6093 | 8.0747 | 7.4205 | 7.5987 | 7.4494 | 7.0941 | 7.8834 | 7.7570 | 7.8379 | 7.3799 | 6.8742 |
| E2EDriver | 7.4591 | 2.9072 | 1.2862 | 6.2783 | 8.0985 | 7.4646 | 7.5701 | 7.4894 | 7.4005 | 7.6380 | 7.6818 | 7.7415 | 7.5308 | 7.1571 |
| PoutineE2E | 7.4635 | 3.1858 | 1.3448 | 6.5222 | 7.9679 | 7.8014 | 7.5817 | 7.2899 | 7.3821 | 7.8392 | 7.3947 | 7.5996 | 7.8789 | 6.8415 |
| Swin-Trajectory | 7.4781 | 2.8998 | 1.2137 | 6.5687 | 7.7550 | 7.6162 | 7.6534 | 7.5615 | 7.2678 | 7.8482 | 7.3710 | 7.8838 | 7.6385 | 7.0947 |
| DriveTraj | 7.4957 | 2.9556 | 1.3038 | 6.4101 | 7.9410 | 7.6557 | 7.6210 | 7.5413 | 7.2204 | 8.0220 | 7.4979 | 7.7371 | 7.8075 | 6.9989 |
| ViT-Adapter-GRU | 7.4988 | 2.7024 | 1.1968 | 6.4543 | 8.0011 | 7.5842 | 7.6146 | 7.4462 | 7.2566 | 8.0576 | 7.6036 | 7.6997 | 7.5373 | 7.2319 |
| casual_reason_v1_mot | 7.5050 | 2.7201 | 1.1867 | 6.2509 | 8.1763 | 7.7128 | 7.3067 | 7.7980 | 7.3274 | 7.9884 | 7.5569 | 7.7767 | 7.7529 | 6.9084 |
| Lo-VLM | 7.5090 | 3.1974 | 1.3108 | 6.4416 | 8.0417 | 7.6989 | 7.5848 | 7.3711 | 7.5949 | 7.9848 | 7.3231 | 7.6774 | 8.0727 | 6.8077 |
| DrivePI | 7.5160 | 2.9066 | 1.2526 | 5.9117 | 8.2487 | 7.8302 | 7.9286 | 7.7029 | 7.1695 | 7.9890 | 7.4849 | 7.9575 | 7.5510 | 6.9017 |
| waymo | 7.5281 | 3.0182 | 1.3200 | 6.5953 | 8.2729 | 7.6226 | 7.6651 | 7.5172 | 7.3012 | 7.8515 | 7.8690 | 7.7627 | 7.4597 | 6.8919 |
| Swin-Trajectory | 7.5432 | 2.8135 | 1.2082 | 6.6791 | 7.9920 | 7.8068 | 7.6597 | 7.4274 | 7.1149 | 8.0737 | 7.5512 | 7.8871 | 7.5583 | 7.2244 |
| DiffusionLTF | 7.5447 | 2.9419 | 1.3664 | 6.5605 | 8.0996 | 7.7378 | 7.6098 | 7.4814 | 7.3324 | 8.0768 | 7.5429 | 7.8034 | 7.6150 | 7.1326 |
| AutoVLA | 7.5566 | 2.9580 | 1.3507 | 6.9436 | 7.9556 | 7.7112 | 7.5920 | 7.3208 | 7.5100 | 8.1450 | 7.5256 | 7.9074 | 7.6968 | 6.8151 |
| BBC | 7.5686 | 2.7328 | 1.2194 | 6.2271 | 8.3214 | 7.6235 | 7.5138 | 7.6458 | 7.3267 | 7.9898 | 7.7184 | 7.7794 | 8.0308 | 7.0782 |
| DiffusionLTF | 7.5919 | 2.9768 | 1.3605 | 6.5688 | 8.1589 | 7.6920 | 7.8409 | 7.4625 | 7.1839 | 8.3272 | 7.5844 | 7.9400 | 7.8094 | 6.9428 |
| UniPlan | 7.6317 | 2.8567 | 1.2711 | 6.6023 | 8.5656 | 7.8536 | 7.5056 | 7.4298 | 7.5420 | 7.9557 | 7.6244 | 7.9216 | 7.6402 | 7.3076 |
| dVLM-AD | 7.6331 | 3.0221 | 1.2849 | 6.5597 | 8.1177 | 7.9692 | 7.6775 | 7.5778 | 7.6359 | 8.1150 | 7.4486 | 8.0655 | 7.5812 | 7.2161 |
| Lo-VLM-RL | 7.6345 | 3.3274 | 1.3714 | 6.6662 | 8.2173 | 7.7386 | 7.6182 | 7.4813 | 7.4396 | 8.1842 | 7.7677 | 7.7492 | 7.8971 | 7.2197 |
| UniPlan | 7.6834 | 2.9358 | 1.2934 | 6.7749 | 8.3241 | 7.8455 | 7.7279 | 7.3319 | 7.5659 | 8.1217 | 7.7435 | 7.9270 | 7.6463 | 7.5085 |
| TrajScorer | 7.6836 | 3.0549 | 1.3044 | 6.6936 | 8.3227 | 7.7736 | 7.9432 | 7.6692 | 7.4577 | 8.1176 | 7.5632 | 7.9429 | 8.0196 | 7.0168 |
| UniPlan | 7.6925 | 2.9864 | 1.3083 | 6.6544 | 8.2809 | 7.8824 | 7.8288 | 7.2993 | 7.4702 | 8.1452 | 7.7844 | 7.8437 | 7.7586 | 7.6701 |
| CTL-Drive-V8 | 7.7055 | 2.9986 | 1.2816 | 6.6584 | 7.9469 | 7.8691 | 7.7358 | 7.6204 | 7.7716 | 8.0645 | 7.9513 | 7.9893 | 8.0273 | 7.1258 |
| E2EDriver | 7.7107 | 2.9285 | 1.2709 | 6.6285 | 8.3973 | 7.8531 | 7.9237 | 7.6493 | 7.2990 | 8.2460 | 7.6444 | 7.8918 | 7.9400 | 7.3447 |
| NTR | 7.7146 | 3.0570 | 1.2928 | 6.4265 | 8.4070 | 8.0147 | 8.1016 | 7.6846 | 7.2871 | 8.4444 | 7.4429 | 8.0750 | 7.9708 | 7.0060 |
| DiffusionLTF | 7.7172 | 2.8914 | 1.3561 | 6.4138 | 8.2601 | 7.9269 | 7.9085 | 7.7965 | 7.4163 | 8.2603 | 7.6933 | 8.0938 | 7.7401 | 7.3795 |
| HMVLM | 7.7367 | 3.0715 | 1.3269 | 6.7269 | 8.6663 | 7.9043 | 7.8578 | 7.3925 | 7.5607 | 8.3563 | 7.5826 | 7.8842 | 7.9710 | 7.2013 |
| Traj-Refine | 7.7384 | 3.0011 | 1.3035 | 6.7914 | 8.2909 | 7.9207 | 7.9117 | 7.4041 | 7.6059 | 8.2318 | 7.5225 | 7.9050 | 8.2004 | 7.3375 |
| Traj-Refine-MS-Epoch1 | 7.7387 | 3.0587 | 1.3159 | 6.8245 | 8.1051 | 7.9450 | 7.8054 | 7.4054 | 7.7354 | 8.2797 | 7.5569 | 7.8510 | 8.2199 | 7.3976 |
| Traj-Refine-MS-Epoch1 | 7.7387 | 3.0587 | 1.3159 | 6.8245 | 8.1051 | 7.9450 | 7.8054 | 7.4054 | 7.7354 | 8.2797 | 7.5569 | 7.8510 | 8.2199 | 7.3976 |
| TrajScorer | 7.7646 | 2.9615 | 1.2717 | 6.7489 | 8.3426 | 7.7063 | 7.9766 | 7.5691 | 7.6139 | 8.2993 | 7.7148 | 7.9878 | 8.0785 | 7.3724 |
| BBC | 7.7705 | 2.6710 | 1.1588 | 6.5130 | 8.4049 | 7.7971 | 7.6784 | 7.7686 | 7.6594 | 8.3240 | 7.6554 | 8.0764 | 8.1587 | 7.4393 |
| UniPlan | 7.7795 | 2.8423 | 1.2671 | 6.9174 | 8.5600 | 7.8639 | 7.6384 | 7.7559 | 7.6699 | 8.1599 | 7.7859 | 8.0847 | 7.6702 | 7.4685 |
| Traj-Refine | 7.8345 | 2.9223 | 1.2769 | 6.8894 | 8.6409 | 7.9483 | 7.9522 | 7.5568 | 7.6514 | 8.4570 | 8.0492 | 7.8592 | 8.0075 | 7.1674 |
| IRL-VLA | 7.8372 | 2.8213 | 1.1933 | 6.7801 | 8.3698 | 7.9602 | 7.9235 | 7.7106 | 7.8658 | 8.6436 | 7.9882 | 8.0585 | 7.7399 | 7.1687 |
| BBC | 7.8387 | 2.6545 | 1.0922 | 6.8919 | 8.1347 | 8.0768 | 7.9076 | 7.8454 | 7.8228 | 8.5122 | 7.6003 | 8.1106 | 8.0881 | 7.2357 |
| ViT-Adapter-GRU | 7.8493 | 2.8888 | 1.4434 | 6.6722 | 8.4630 | 8.0471 | 7.8904 | 7.8346 | 7.6132 | 8.3424 | 7.9308 | 8.0682 | 8.0920 | 7.3889 |
| FROST-Drive | 7.8560 | 3.5653 | 2.5373 | 7.0941 | 8.2510 | 7.9658 | 7.9293 | 7.7192 | 7.6509 | 8.3287 | 7.9323 | 8.1297 | 8.0375 | 7.3775 |
| Traj_AR_TEST | 7.8642 | 2.7261 | 1.2200 | 7.0405 | 8.2292 | 7.9403 | 7.9288 | 7.7097 | 7.7446 | 8.5989 | 7.7320 | 8.1144 | 7.8309 | 7.6371 |
| BBC | 7.8663 | 2.6561 | 1.0872 | 6.8853 | 8.1713 | 8.0780 | 7.9019 | 7.8453 | 7.8909 | 8.5493 | 7.6503 | 8.1246 | 8.1447 | 7.2881 |
| qwer | 7.8761 | 3.2699 | 1.8622 | 6.8645 | 8.5262 | 8.0675 | 7.8770 | 7.7323 | 7.9416 | 8.3531 | 7.7839 | 8.0726 | 8.1195 | 7.2991 |
| IRL-VLA | 7.8900 | 2.8235 | 1.2184 | 6.8503 | 8.4401 | 7.9781 | 7.9236 | 7.7757 | 7.9636 | 8.6387 | 8.1261 | 8.0924 | 7.7182 | 7.2831 |
| **Poutine-Base** | **7.9093** | **2.9412** | **1.2717** | **7.0371** | **8.5151** | **8.0440** | **7.8680** | **7.8222** | **7.8880** | **8.3681** | **7.8277** | **8.0496** | **8.2155** | **7.3674** |
| **Poutine** | **7.9860** | **2.7419** | **1.2055** | **6.8929** | **8.3595** | **8.1356** | **7.9529** | **7.8325** | **7.7789** | **8.6101** | **8.2588** | **8.2043** | **8.2965** | **7.5235** |
| **RAP** | **8.0430** | **2.6457** | **1.1741** | **7.2041** | **8.6939** | **8.1798** | **8.0336** | **7.7604** | **8.0999** | **8.5232** | **8.0976** | **8.2741** | **7.8176** | **7.7893** |
| **TTVLM** | **8.0434** | **2.8433** | **1.2577** | **7.0927** | **8.6441** | **8.1223** | **8.0894** | **7.8821** | **8.0519** | **8.3743** | **8.1523** | **8.2666** | **8.1343** | **7.6674** |
| **NTR** | **8.0461** | **2.6379** | **1.1729** | **7.1234** | **8.8159** | **8.1237** | **8.0764** | **7.9158** | **8.0136** | **8.6194** | **8.1160** | **8.2469** | **7.7514** | **7.7040** |

## Key Observations

## Method Metadata

> Source note: this table combines official Waymo reports, public papers/repos/pages, project-owner metadata, and conservative name-based inference. `not found` means no reliable public method report was found in this search pass. Duplicate leaderboard rows are collapsed here into one metadata row per method.

| Method | Best RFS | Entries | Authors | Architecture comments | Source status | Link |
|---|---:|---:|---|---|---|---|
| GAT-Transformer-LSTM | 4.0659 | 1 | TBD | Graph Attention Network + Transformer + LSTM sequence model, inferred from method name. | name-inferred | - |
| magicVLM | 4.6243 | 1 | TBD | Likely VLM-based entry from name; no reliable public method report found. | not found | - |
| ResTransMLP | 6.8440 | 2 | TBD | Residual Transformer + MLP trajectory head, inferred from method name. | name-inferred | - |
| OpenEMMA | 5.1575 | 1 | TBD | Open-source EMMA-style MLLM E2E driving framework with CoT reasoning. Exact WOD-E2E variant not publicly confirmed. | paper listing | [HF](https://huggingface.co/papers?q=diffusion-based+paradigm) |
| ViT-GRU with Modality Fusion | 7.2335 | 4 | TBD | ViT visual encoder + GRU temporal/trajectory decoder with modality fusion, inferred from method name. | name-inferred | - |
| AgentDrive RouterVLM | 6.0679 | 1 | TBD | Likely VLM routing/agentic planning entry from name; no reliable public method report found. | not found | - |
| Baseline | 6.2365 | 1 | Waymo / WOD-E2E authors | NaiveEMMA baseline: simplified Gemini Flash/Gemini1 Nano-style VLM; 8-camera 768x768 montage, 3s ego-status history, high-level route; SFT on WOD-E2E train. | WOD-E2E paper | [WOD-E2E](https://arxiv.org/abs/2510.26125) |
| OpenEMMA w/ QwenVLM | 6.3162 | 1 | TBD | OpenEMMA variant using Qwen VLM backbone; MLLM/CoT driving framework. | inferred from method name | [OpenEMMA listing](https://huggingface.co/papers?q=diffusion-based+paradigm) |
| FrozenResNet50 | 6.4719 | 2 | TBD | Frozen ResNet-50 visual encoder baseline/variant, inferred from method name. | name-inferred | - |
| E2E Bottleneck Scaling 250 | 6.4674 | 1 | TBD | Bottleneck scaling experiment/variant, inferred from method name. | name-inferred | - |
| LightEMMA | 6.5169 | 1 | TBD | Lightweight EMMA/VLM autonomous-driving framework; WOD-E2E paper classifies LightEMMA with Qwen2.5-VL-style VLM methods. | paper listing | [HF](https://huggingface.co/papers?q=End-to-End+Autonomous+Driving) |
| Poutine-Concordia-v4 | 6.5384 | 1 | TBD | Poutine-named Concordia variant; exact relation to Poutine report not confirmed. | not found | - |
| CTL-Drive | 6.9969 | 2 | Xingnan Zhou, Ciprian Alecsandru | VLM-based WOD-E2E method trained on a single RTX 4090 with QLoRA; no RL according to project profile. | author profile | [Profile](https://obsicat.com/) |
| MultiModalMamba FULL | 7.0394 | 2 | TBD | Multimodal Mamba/SSM-family temporal model, inferred from method name. | name-inferred | - |
| WayPredict | 7.0641 | 1 | TBD | Waymo-oriented trajectory prediction/planning model family, exact architecture not publicly found. | not found | - |
| MultiModal-Transformer-ViT | 7.1651 | 1 | TBD | Multimodal Transformer with ViT visual backbone, inferred from method name. | name-inferred | - |
| MTR-VP | 7.3433 | 2 | Maitrayee Keskar, Mohan M. Trivedi, Ross Greer | Vision-based Planning Motion Transformer; ViT/context-driven image encoding + multiple trajectory prediction inspired by MTR. | paper summary | [ChatPaper](https://chatpaper.com/zh-CN/paper/214222) |
| PredNet | 7.2316 | 1 | TBD | Prediction-network style trajectory model, exact architecture not publicly found. | not found | - |
| gemma-2-2b-it | 7.2335 | 1 | TBD | Open LLM/VLM backbone entry, exact driving wrapper not publicly found. | name-inferred | - |
| WayPredict-XL | 7.4128 | 4 | TBD | Waymo-oriented trajectory prediction/planning model family, exact architecture not publicly found. | not found | - |
| zx-method | 7.4214 | 2 | TBD | No public method report found. | not found | - |
| WayNet | 7.3463 | 1 | TBD | WayNet trajectory/planning model, exact architecture not publicly found. | not found | - |
| gemma-2-9b | 7.3635 | 1 | TBD | Open LLM/VLM backbone entry, exact driving wrapper not publicly found. | name-inferred | - |
| open-llama | 7.4288 | 1 | TBD | Open LLM/VLM backbone entry, exact driving wrapper not publicly found. | name-inferred | - |
| DrivePI | 7.5160 | 2 | Zhe Liu, Runhui Huang, Yang Rui, Hengshuang Zhao, et al. | Spatial-aware 4D MLLM / VLA framework for understanding, occupancy, prediction, and planning; WOD entry match uncertain. | paper listing | [HF](https://huggingface.co/papers?q=DRiVE) |
| E2EDriver | 7.7107 | 2 | TBD | TBD | not found | - |
| PoutineE2E | 7.4635 | 1 | Luke Rowe, Rodrigue de Schaetzen, Roger Girgis, Christopher Pal, Liam Paull | Likely Poutine-family entry; exact variant not publicly confirmed. | inferred from method name | [Poutine PDF](https://storage.googleapis.com/waymo-uploads/files/research/2025%20Technical%20Reports/2025%20WOD%20E2E%20Driving%20Challenge%20-%20Special%20Mention%20-%20Poutine.pdf) |
| Swin-Trajectory | 7.5432 | 2 | Sungjin Park, Gwangik Shin, Jaeha Song, Sumin Lee, Hyukju Shon, Byounggun Park, Jinhee Na, Hawook Jeong, Soonmin Hwang | Lightweight Swin Transformer waypoint predictor; single/front-camera or front-camera-focused setup, ego-state encoder, waypoint-query trajectory decoder; 14ms RTX 4090 reported. | official report | [PDF](https://storage.googleapis.com/waymo-uploads/files/research/2025%20Technical%20Reports/2025%20WOD%20E2E%20Driving%20Challenge%20-%203rd%20Place%20-%20Swin-Trajectory.pdf) |
| DriveTraj | 7.4957 | 1 | TBD | Trajectory-focused driving model, exact architecture not publicly found. | not found | - |
| ViT-Adapter-GRU | 7.8493 | 2 | TBD | ViT adapter plus GRU decoder, inferred from method name. | name-inferred | - |
| casual_reason_v1_mot | 7.5050 | 1 | TBD | Reasoning/motion variant, exact architecture not publicly found. | not found | - |
| Lo-VLM | 7.5090 | 1 | TBD | VLM-family entry; exact public report not found. | not found | - |
| waymo | 7.5281 | 1 | Waymo / WOD-E2E authors | Likely NaiveEMMA/Baseline-family entry from WOD-E2E paper; 8-camera VLM trajectory baseline. | inferred from WOD-E2E paper | [WOD-E2E](https://arxiv.org/abs/2510.26125) |
| DiffusionLTF | 7.7172 | 3 | Long Nguyen, Micha Fauth, Bernhard Jaeger, Daniel Dauner, Maximilian Igl, Andreas Geiger, Kashyap Chitta | Open X-AV multi-dataset setup; Latent TransFuser/DiffusionDrive-style trajectory proposals; trained with WOD-E2E, CARLA, NAVSIM, WOD-Perception. | official report | [PDF](https://storage.googleapis.com/waymo-uploads/files/research/2025%20Technical%20Reports/2025%20WOD%20E2E%20Driving%20Challenge%20-%202nd%20Place%20-%20DiffusionLTF.pdf) |
| AutoVLA | 7.5566 | 1 | Zewei Zhou, Tianhui Cai, Seth Z. Zhao, Yun Zhang, Zhiyu Huang, Bolei Zhou, Jiaqi Ma | VLA model with trajectory/action tokenization, adaptive fast/slow CoT modes, SFT plus GRPO/RFT; reports high WOD-E2E Spotlight performance. | project/repo/arXiv | [Project](https://autovla.github.io/) |
| BBC | 7.8663 | 4 | TBD | High-ADE-quality leaderboard family; no reliable public method report found. | not found | - |
| UniPlan | 7.7795 | 4 | Lan Feng, Alexandre Alahi | DiffusionDrive-style anchored diffusion planner; WOD-E2E + nuPlan training; front-3 camera concatenation; WOD-E2E-specific anchors; multi-seed candidate selection. | official report | [PDF](https://storage.googleapis.com/waymo-uploads/files/research/2025%20Technical%20Reports/2025%20WOD%20E2E%20Driving%20Challenge%20-%201st%20Place%20-%20UniPlan.pdf) |
| dVLM-AD | 7.6331 | 1 | TBD | Diffusion VLM planning with LLaDA-V/LLaDA-8B + SigLIP2; structured reasoning trace and textual waypoints; dynamic denoising / controllable reasoning. | secondary summary | [Review](https://www.themoonlight.io/en/review/dvlm-ad-enhance-diffusion-vision-language-model-for-driving-via-controllable-reasoning) |
| Lo-VLM-RL | 7.6345 | 1 | TBD | VLM-family entry, likely RL-tuned variant; exact public report not found. | not found | - |
| TrajScorer | 7.7646 | 2 | TBD | Name suggests candidate trajectory scoring/reranking; no reliable public report found. | not found | - |
| CTL-Drive-V8 | 7.7055 | 1 | Xingnan Zhou, Ciprian Alecsandru | Likely CTL-Drive variant; VLM/QLoRA single-GPU lineage, exact V8 changes not publicly verified. | inferred from author profile | [Profile](https://obsicat.com/) |
| NTR | 8.0461 | 2 | Jiahui Li, Jiawei Sun, Kaidi Yang, Liying Liu, Zuoguan Wang | Top leaderboard method in supplied snapshot; no public architecture report found in search. | metadata only | - |
| HMVLM | 7.7367 | 1 | Daming Wang, Yuhao Song, Zijian He, Kangliang Chen, Xing Pan, Lu Deng, Weihao Gu | HaoMo VLM slow-path; selective five-view prompting, 4s ego history, multi-stage CoT: scene understanding -> decision -> trajectory; spline post-processing. | paper | [RG](https://www.researchgate.net/publication/392513976_HMVLM_Multistage_Reasoning-Enhanced_Vision-Language_Model_for_Long-Tailed_Driving_Scenarios) |
| Traj-Refine | 7.8345 | 2 | Yixin Huang | Trajectory refinement method for better trajectory prediction per supplied metadata; no public report found. | metadata only | - |
| Traj-Refine-MS-Epoch1 | 7.7387 | 2 | Yixin Huang | more_status epoch1 checkpoint variant per supplied metadata; no public report found. | metadata only | - |
| IRL-VLA | 7.8900 | 2 | Jiang Anqing | IRL-VLA dino version per supplied metadata; public report not found. | metadata only | - |
| FROST-Drive | 7.8560 | 1 | Zeyu Dong, Yimin Zhu, Yu Wu, Yu Sun | Frozen VLM vision encoder; transformer adapter for multimodal fusion; GRU decoder; custom RFS-oriented loss; argues frozen encoder generalizes better than full fine-tuning. | paper | [RG](https://www.researchgate.net/publication/399559397_FROST-Drive_Scalable_and_Efficient_End-to-End_Driving_with_a_Frozen_Vision_Encoder) |
| Traj_AR_TEST | 7.8642 | 1 | Weicheng Zheng | V-meta action-A-RL per supplied metadata; no public report found. | metadata only | - |
| qwer | 7.8761 | 1 | peter | No public method report found. | metadata only | - |
| Poutine-Base | 7.9093 | 1 | Luke Rowe, Rodrigue de Schaetzen, Roger Girgis, Christopher Pal, Liam Paull | Pre-RL Poutine variant: VLT next-token pretraining without GRPO preference tuning. | official report / arXiv | [PDF](https://storage.googleapis.com/waymo-uploads/files/research/2025%20Technical%20Reports/2025%20WOD%20E2E%20Driving%20Challenge%20-%20Special%20Mention%20-%20Poutine.pdf) |
| Poutine | 7.9860 | 1 | Luke Rowe, Rodrigue de Schaetzen, Roger Girgis, Christopher Pal, Liam Paull | 3B Qwen2.5-VL driving VLM; VLT next-token pretraining on CoVLA + WOD-E2E; 72B VLM auto-labels; GRPO preference tuning with validation rater labels. | official report / arXiv | [PDF](https://storage.googleapis.com/waymo-uploads/files/research/2025%20Technical%20Reports/2025%20WOD%20E2E%20Driving%20Challenge%20-%20Special%20Mention%20-%20Poutine.pdf) |
| RAP | 8.0430 | 1 | Lan Feng, Yang Gao, Eloi Zablocki, Quanyi Li, Wuyang Li, Sichao Liu, Matthieu Cord, Alexandre Alahi | Rasterization Augmented Planning: lightweight 3D rasterization, counterfactual recovery/cross-agent views, raster-to-real feature alignment; #1 later leaderboard entry. | paper/model card | [HF](https://huggingface.co/papers/2510.04333) |
| TTVLM | 8.0434 | 1 | Caio Azevedo, Stefano Sabatini, Sascha Hornauer, Fabien Moutarde | Qwen 3.5 2B VLM variant; pPt, noCoT, dPt0.2 per supplied metadata; public report not found. | metadata only | - |

### Top 3

| Method | Overall | Construction | Single-lane | Cut-ins | FOD | Spotlight |
|---|---:|---:|---:|---:|---:|---:|
| NTR | 8.0461 | **8.8159** | **8.6194** | 8.1160 | 8.2469 | 7.1234 |
| TTVLM | 8.0434 | 8.6441 | 8.3743 | **8.1523** | 8.2666 | 7.0927 |
| RAP | 8.0430 | 8.6939 | 8.5232 | 8.0976 | **8.2741** | **7.2041** |

### Cluster Difficulty

| Cluster | Floor | Top | Spread |
|---|---:|---:|---:|
| Construction | 4.2 | 8.8 | 4.6 |
| Single-lane | 4.0 | 8.6 | 4.6 |
| Spotlight | 4.1 | 7.2 | 3.1 |
| Others | 4.0 | 7.8 | 3.8 |
| Special Vehicles | 4.0 | 8.3 | 4.3 |

### Architecture Patterns In Top Methods

- Diffusion-based methods, especially DiffusionLTF, are consistent across clusters and strong on single-lane.
- VLM-augmented methods such as IRL-VLA, HMVLM, and dVLM-AD are strong on single-lane and pedestrian-like reasoning.
- Trajectory refinement methods appear strong on Special Vehicles.
- BBC entries combine strong RFS with very low ADE, suggesting tight trajectory prediction.
- NTR dominates Construction and Single-lane, while Spotlight remains a weak cluster even for top methods.

## Submission Implications

- Spotlight is the highest-value target because the ceiling remains materially lower than other clusters.
- Construction and Single-lane have the widest improvement spread, so they are good ablation targets.
- FOD and Special Vehicles reward semantic recognition, making them useful for evaluating any VLM/world-model component.
- ADE alone is insufficient: several rows have similar ADE but materially different RFS by cluster.
- Official prize placement differs from later leaderboard ranking; use `docs/competitive-analysis.md` when comparing methods strategically.
