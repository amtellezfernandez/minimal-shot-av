# nuPlan ManeuverToken Adapter Rollout

- Scene count: `1000`
- Proxy-safe rate: `0.774`
- Proxy-safe selected count: `774`
- Proxy-safe + realized safe count: `0`
- Proxy-safe + realized near/collision count: `0`
- Proxy-safe + realized missing count: `774`
- Selected token histogram: `{"evasive_left": 99, "evasive_right": 112, "lane_recover": 129, "maintain": 435, "nudge_left": 23, "nudge_right": 80, "slow_yield": 122}`

## Failure Table

| Failure rung | Count | Rate |
|---|---:|---:|
| no actor/state visibility | 0 | 0.000 |
| no safe token existed | 95 | 0.095 |
| safe token existed but selector missed it | 131 | 0.131 |
| proxy predicted safe but realized failed | 0 | 0.000 |
| metric/spec ambiguity | 774 | 0.774 |

## Selected scenes

- `2021.06.09.17.23.18_veh-38_02526_03027:4e35fbc3aa4d555d`: selected `maintain` (proxy_safe=True, safe_token_existed=True, failure_rung=metric/spec ambiguity, proxy_min_clearance=1.448 m, realized_min_clearance=None)
- `2021.06.09.17.23.18_veh-38_02526_03027:6b3acaaec3875e00`: selected `maintain` (proxy_safe=True, safe_token_existed=True, failure_rung=metric/spec ambiguity, proxy_min_clearance=1.412 m, realized_min_clearance=None)
- `2021.06.09.17.23.18_veh-38_02526_03027:0eecde897b9d5ff5`: selected `maintain` (proxy_safe=True, safe_token_existed=True, failure_rung=metric/spec ambiguity, proxy_min_clearance=1.633 m, realized_min_clearance=None)
- `2021.06.09.17.23.18_veh-38_02526_03027:8ab56791dab25aab`: selected `maintain` (proxy_safe=True, safe_token_existed=True, failure_rung=metric/spec ambiguity, proxy_min_clearance=1.501 m, realized_min_clearance=None)
- `2021.06.09.17.23.18_veh-38_02526_03027:22d429fa712352e4`: selected `maintain` (proxy_safe=True, safe_token_existed=True, failure_rung=metric/spec ambiguity, proxy_min_clearance=1.586 m, realized_min_clearance=None)
- `2021.06.09.17.23.18_veh-38_02526_03027:a5ad89515d7e517c`: selected `maintain` (proxy_safe=True, safe_token_existed=True, failure_rung=metric/spec ambiguity, proxy_min_clearance=1.536 m, realized_min_clearance=None)
- `2021.06.09.17.23.18_veh-38_02526_03027:f5fd9df7bf1f5d7e`: selected `maintain` (proxy_safe=True, safe_token_existed=True, failure_rung=metric/spec ambiguity, proxy_min_clearance=1.384 m, realized_min_clearance=None)
- `2021.06.09.17.23.18_veh-38_02526_03027:cc7fc5ea4eff5260`: selected `maintain` (proxy_safe=True, safe_token_existed=True, failure_rung=metric/spec ambiguity, proxy_min_clearance=1.553 m, realized_min_clearance=None)
- `2021.06.09.17.23.18_veh-38_02526_03027:cb3f9a1463615c48`: selected `maintain` (proxy_safe=True, safe_token_existed=True, failure_rung=metric/spec ambiguity, proxy_min_clearance=1.584 m, realized_min_clearance=None)
- `2021.06.09.17.23.18_veh-38_02526_03027:f25a411561b05534`: selected `maintain` (proxy_safe=True, safe_token_existed=True, failure_rung=metric/spec ambiguity, proxy_min_clearance=1.578 m, realized_min_clearance=None)
