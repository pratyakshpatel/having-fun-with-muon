# Muon Routing Atlas: Results Summary

This note summarizes the completed experiment outputs in `runs/`. It uses final validation loss as the primary metric; lower is better. The analysis includes 41 completed runs with non-missing validation metrics.

## Main conclusion

The results support the hypothesis that Muon's useful effect is concentrated in non-Q/K hidden matrices, especially the FFN plus V/O path. Removing Q/K from Muon retains most of the gain, but it does not match the best all-hidden routing in these runs.

More precisely:

- `muon_all_hidden` is the best routing in every seed-replicated main comparison.
- `muon_no_qk` is consistently second-best and recovers most of the `muon_all_hidden` gain: 83.2% on TinyStories 12M and 84.0% on FineWeb-Edu 85M.
- `muon_mlp_only` and `muon_vo_only` individually do not recover most of the FineWeb-Edu 85M gain; each recovers about 25-26%.
- `muon_qk_only` is weak on TinyStories 12M. It improves over AdamW, but recovers only 38.4% of the all-hidden Muon gain. There is no completed FineWeb-Edu Q/K-only comparison in this run set.
- Geometry metrics support a qualitative difference between FFN and V/O updates: FFN `gate` and `up` matrices receive larger Muon update norms than attention V/O matrices.

![Final validation loss for main routing ablations](report/figures/results_final_loss_main.png)

![Fraction of all-hidden Muon gain recovered by each routing](report/figures/results_gain_recovery.png)

## Seed-replicated comparisons

Gain recovered is defined as:

```text
(AdamW final loss - routing final loss) / (AdamW final loss - Muon all-hidden final loss)
```

### TinyStories 12M

| Routing | Seeds | Final validation loss | Gain vs AdamW | Gain recovered |
|---|---:|---:|---:|---:|
| `muon_all_hidden` | 2 | 2.2672 +/- 0.0188 | 0.7850 | 100.0% |
| `muon_no_qk` | 2 | 2.3992 +/- 0.0229 | 0.6530 | 83.2% |
| `muon_attn_only` | 2 | 2.4650 +/- 0.0194 | 0.5872 | 74.8% |
| `muon_mlp_only` | 2 | 2.5710 +/- 0.0158 | 0.4812 | 61.3% |
| `muon_qk_only` | 2 | 2.7504 +/- 0.0177 | 0.3018 | 38.4% |
| `muon_vo_only` | 2 | 2.8312 +/- 0.0174 | 0.2210 | 28.1% |
| `adamw_all` | 2 | 3.0522 +/- 0.0176 | 0.0000 | 0.0% |

### FineWeb-Edu 85M

| Routing | Seeds | Final validation loss | Gain vs AdamW | Gain recovered |
|---|---:|---:|---:|---:|
| `muon_all_hidden` | 2 | 3.7848 +/- 0.0242 | 0.3610 | 100.0% |
| `muon_no_qk` | 2 | 3.8425 +/- 0.0182 | 0.3032 | 84.0% |
| `muon_mlp_only` | 2 | 4.0526 +/- 0.0030 | 0.0931 | 25.8% |
| `muon_vo_only` | 2 | 4.0547 +/- 0.0213 | 0.0910 | 25.2% |
| `adamw_all` | 2 | 4.1458 +/- 0.0222 | 0.0000 | 0.0% |

## Batch-size sensitivity

FineWeb-Edu 35M runs form an effective-batch sweep. Loss worsens as effective tokens per optimizer step increase, even though throughput slightly improves. The relative ordering remains stable: `muon_all_hidden` is best, `muon_no_qk` is second, and AdamW is worst.

![FineWeb-Edu 35M batch-size sensitivity](report/figures/results_batch_size_scaling.png)

| Effective tokens / step | AdamW | Muon all-hidden | Muon no-Q/K |
|---:|---:|---:|---:|
| 65,536 | 5.0273 | 4.4517 | 4.6173 |
| 131,072 | 5.4269 | 4.6628 | 4.8493 |
| 258,048 | 5.9085 | 5.0500 | 5.1976 |
| 512,000 | 6.3870 | 5.7640 | 5.8762 |

## Geometry

The clearest geometry signal is in update norms. On FineWeb-Edu 85M, Muon update norms are larger for FFN `gate` and `up` matrices than for attention V/O matrices. This supports the interpretation that FFN and V/O are not interchangeable under Muon, even when both contribute to the non-Q/K gain.

![FineWeb-Edu 85M Muon update norms by module](report/figures/results_update_norm_geometry.png)

Approximate mean update norms for `muon_all_hidden` on FineWeb-Edu 85M:

| Module | Mean update norm |
|---|---:|
| `attn.q_proj` | 0.0490 |
| `attn.k_proj` | 0.0489 |
| `attn.v_proj` | 0.0468 |
| `attn.o_proj` | 0.0481 |
| `mlp.down_proj` | 0.0501 |
| `mlp.gate_proj` | 0.0791 |
| `mlp.up_proj` | 0.0790 |

## Interpretation of the original claims

| Claim | Supported? | Evidence |
|---|---|---|
| Muon no-Q/K matches or beats Muon all-hidden. | No. | `muon_no_qk` is second-best but worse than `muon_all_hidden` in every comparable group. |
| Muon MLP + V/O recovers almost all Muon gain. | Mostly yes, if represented by `muon_no_qk`. | `muon_no_qk` recovers 83-84% of the all-hidden gain in the seed-replicated main comparisons. |
| Q/K-only does little or hurts. | Partially. | TinyStories Q/K-only is much weaker than all-hidden and no-Q/K, but it still beats AdamW; no FineWeb Q/K-only run is present. |
| Geometry metrics show FFN and V/O updates behave differently under Muon. | Yes. | FFN `gate/up` update norms are substantially larger than V/O update norms. |

## Caveats

- Most main comparisons have two seeds. This is enough to show a consistent ranking here, but not enough for a high-confidence statistical claim.
- FineWeb-Edu 35M entries are a batch-size sweep, not independent same-config seed replicates.
- The layerwise TinyStories 35M runs are single-seed exploratory runs. They suggest middle-layer routing can be competitive, but they should not be treated as a final ranking.
- The existing run set does not include a FineWeb-Edu Q/K-only ablation, so claims about Q/K-only on FineWeb remain untested.
