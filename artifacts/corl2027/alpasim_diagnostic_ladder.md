# AlpaSim diagnostic ladder

Conclusion: Residual AlpaSim collision is not localized to actor visibility, token candidate availability, learned selector ranking, or direct-grid cost ranking alone.

## Rungs

- `actor_visibility`: `visibility_real_but_not_sufficient`; score=0.000. Actor dropout is real at impact, but matched actor completion preserves the same raw collision count; this rung is necessary context, not a sufficient explanation.
- `token_candidate_feasibility`: `token_safe_candidate_present_in_all_collision_scenes`; score=0.000. The committed open-loop audit finds proxy-safe token candidates before impact in every collision scene, so pure candidate absence is not the dominant explanation.
- `token_selector_ranking`: `selector_miss_secondary`; score=0.222. Selector mistakes exist, but many scenes already select actor-axis-safe tokens before collision; selector ranking alone is not sufficient.
- `direct_grid_cost_ranking`: `direct_grid_cost_not_open_loop_bottleneck`; score=0.000. The direct-grid open-loop cost selects proxy-safe actions where available, so the remaining collision cannot be assigned to direct-grid cost ranking alone.
- `controller_execution_or_proxy_mismatch`: `residual_load_bearing_boundary`; score=0.889. Closed-loop direct-grid replay removes only a small number of collisions despite open-loop proxy-safe actions. The residual boundary is controller execution, receding-horizon erosion, or proxy/simulator clearance mismatch.

## Offline localization

- `controller_execution_or_proxy_mismatch`: 0.800 (95% bootstrap CI 0.667-0.944)
- `token_selector_ranking`: 0.200 (95% bootstrap CI 0.056-0.333)
- `actor_visibility`: 0.000 (95% bootstrap CI 0.000-0.000)
- `token_candidate_feasibility`: 0.000 (95% bootstrap CI 0.000-0.000)
- `direct_grid_cost_ranking`: 0.000 (95% bootstrap CI 0.000-0.000)

## Notes

- The offline localization scores are deterministic evidence weights from committed audits, not a learned DROPO posterior.
- The ladder localizes residual collision only within the available AlpaSim artifacts; it does not prove deployment safety.
