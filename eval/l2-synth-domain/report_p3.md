# L2 仪器合成域适用性评估报告

**模式**: P3 img2img 零训练基线（4 锚轮协议）  
**总体判定**: **PASS**

| 挑战 | 样本数 | R_fail_synth (k/n) | Wilson 95% 上界 | R_fail_real | 空 pred | 判定 |
|------|--------|--------------------|-----------------|-------------|---------|------|
| GLI | 80 | 0.0000 (0/80) | 0.0458 | 0.0000 (0/900) | 0 | **PASS** |
| MEN | 80 | 0.0000 (0/80) | 0.0458 | 0.0000 (0/720) | 2 | **PASS** |
| METS | 80 | 0.0000 (0/80) | 0.0458 | 0.0000 (0/171) | 42 | **PASS** |
| PED | 56 | 0.0000 (0/56) | 0.0642 | 0.0000 (0/72) | 0 | **PASS** |
| SSA | 56 | 0.0000 (0/56) | 0.0642 | 0.0000 (0/42) | 0 | **PASS** |

## R_fail 细分

- **GLI**: input_fail=0 run_fail=0 hier_viol=0 (n=80)
- **MEN**: input_fail=0 run_fail=0 hier_viol=0 (n=80)
- **METS**: input_fail=0 run_fail=0 hier_viol=0 (n=80)
- **PED**: input_fail=0 run_fail=0 hier_viol=0 (n=56)
- **SSA**: input_fail=0 run_fail=0 hier_viol=0 (n=56)

## 方向说明

P3 为 img2img 零训练基线（RF 插值 strength=0.9，无 ControlNet）：每轮一个真实模态作锚、其余三模态以该锚为 src 生成，12 有序模态对全覆盖；真实锚通道直接用原始数据（重采样对齐），生成通道为 v1 DM img2img 输出。跨模态自洽性强于 P1 但弱于待训 P3 ControlNet，仅作合成域适用性前置证据。
