# CoRL Evidence Strength Audit

Conclusion: `strong_diagnostic`.

The evidence supports a strong diagnostic CoRL paper, but not a Best-Paper-level method claim. Blocking gates: best_paper_external_scale, positive_method_uniform_dominance, grounding_hidden_test_claim.

## Gates

| Gate | Pass | Current | Target |
| --- | ---: | ---: | --- |
| `diagnostic_internal_proxy_supported` | true | 288 | >=288 cases with at least one controlled tradeoff |
| `external_transfer_supported` | true | 10 | >=10 matched scenes with at least one external tradeoff |
| `best_paper_external_scale` | false | 10 | >=30 matched scenes |
| `positive_method_uniform_dominance` | false | 0 | >=1 external intervention that improves at least one axis without worsening another |
| `grounding_prior_signal` | true | 0.150 | >0 RFS over scalar-only selector |
| `grounding_hidden_test_claim` | false | False | true held-out or test-set grounding result |

## Internal Proxy Perturbation

Tasks: 288; perturbations: 6; tradeoff outcomes: 9; dominance outcomes: 15.

| Variant | Dominates | Tradeoff | Regresses | Unchanged |
| --- | ---: | ---: | ---: | ---: |
| `clamped_iter2` | 4 | 2 | 0 | 0 |
| `hybrid_veto_iter2` | 4 | 2 | 0 | 0 |
| `oracle_token` | 4 | 2 | 0 | 0 |
| `spotlight` | 3 | 3 | 0 | 0 |

## AlpaSim External Matrix

Common scenes: 10; tradeoff outcomes: 3; dominance outcomes: 0.

| Model | Class | Improved axes | Worsened axes |
| --- | --- | --- | --- |
| `token_dagger_iter2_clamped` | `tradeoff` | offroad, wrong_lane, progress, dist_to_gt | collision |
| `token_dagger_iter2_hybrid_clamped` | `tradeoff` | offroad, progress | collision, dist_to_gt |
| `token_dagger_srcdecay` | `tradeoff` | wrong_lane, progress | dist_to_gt |

## WOD Grounding

Best: `Cosmos 64d nonlinear head`; RFS gain over scalar-only: 0.150; head-sensitive grounding: true.
