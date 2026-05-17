| Ablation | RFS | Oracle | Regret | Oracle match | Grounding signal | World prior | Direct selector |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| Scalar / geometry only | 7.695 | 9.068 | 1.373 | 0.403 | none | no | no |
| InternVLA only | 7.672 | 9.225 | 1.554 | 0.392 | InternVLA | no | no |
| Cosmos + InternVLA linear fusion | 7.728 | 9.208 | 1.480 | 0.403 | Cosmos + InternVLA | no | no |
| Cosmos 64d nonlinear head | 7.845 | 9.264 | 1.419 | 0.390 | Cosmos + nonlinear head | no | yes (20, 0.60) |
| Cosmos + latent world prior | 7.816 | 9.254 | 1.438 | 0.392 | Cosmos + latent world prior | yes | no |
| Cosmos + latent world prior + direct selector | 7.848 | 9.254 | 1.407 | 0.399 | Cosmos + latent world prior | yes | yes (13, 0.62) |
