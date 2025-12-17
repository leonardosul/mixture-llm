# Research

This library implements techniques from recent research on multi-LLM collaboration.

## Key Papers

### Mixture-of-Agents (2024)

**Wang et al. "Mixture-of-Agents Enhances Large Language Model Capabilities"**
[arXiv:2406.04692](https://arxiv.org/abs/2406.04692)

The foundational MoA paper introduced the layered architecture where each layer's agents use outputs from the previous layer as auxiliary information. Key findings:

- **65.1% on AlpacaEval 2.0** using only open-source models (vs GPT-4o's 57.5%)
- LLMs exhibit "collaborativeness"—they generate better responses when given reference outputs
- Aggregator quality has **2x more impact** than proposer quality (coefficients 0.588 vs 0.281)
- Optimal configuration: 3 layers, 6 proposers

### Self-MoA (2025)

**Li et al. "Rethinking Mixture-of-Agents: Is Mixing Different Large Language Models Beneficial?"**
[arXiv:2502.00674](https://arxiv.org/abs/2502.00674)

This paper challenged the "diversity is better" assumption:

- Self-MoA outperforms standard MoA by **6.6% on AlpacaEval 2.0**
- Single top model sampled multiple times beats diverse model mixtures
- MoA performance is highly sensitive to proposer quality
- Mixing different LLMs often lowers average quality

## Key Findings

### Layer Depth

Performance improves with depth but with diminishing returns:

| Layers | AlpacaEval 2.0 | Gain |
|--------|----------------|------|
| 1 | ~44% | — |
| 2 | ~61% | +17% |
| 3 | ~65% | +4% |
| 4 | ~66% | +1% |

**Recommendation**: 3 layers is the Pareto-optimal configuration.

### Number of Proposers

More proposers help, but gains diminish:

| Proposers | AlpacaEval 2.0 |
|-----------|----------------|
| 1 | 47.8% |
| 2 | 58.8% |
| 3 | 58.0% |
| 6 | 61.3% |

**Recommendation**: 4-6 proposers for quality-critical applications.

### Model Roles

Different models excel at different roles:

| Model | As Aggregator | As Proposer |
|-------|---------------|-------------|
| GPT-4o | 65.7% | — |
| Qwen1.5-110B-Chat | 61.3% | 45.8% |
| WizardLM-8x22B | 52.9% | 56.9% |
| LLaMA-3-70B-Instruct | Good | Good |

**Key insight**: Some models are better proposers than aggregators (WizardLM), while others excel at both (Qwen, LLaMA).

### Aggregation vs Selection

MoA performs sophisticated synthesis rather than simple selection:

- BLEU score analysis shows positive correlation (0.15-0.30) between aggregator output and best proposer
- Aggregators incorporate elements from multiple proposals
- This outperforms LLM-ranker baselines by ~20 percentage points

### Diversity vs Quality Tradeoff

The Self-MoA paper revealed a nuanced picture:

- **Cross-model diversity** (mixing different LLMs) can hurt if it lowers average quality
- **In-model diversity** (sampling one model with temperature) provides sufficient variation
- Use heterogeneous MoA when models have complementary strengths
- Use Self-MoA when one model is clearly superior

## Optimal Configurations

| Objective | Configuration |
|-----------|---------------|
| Maximum quality | 3 layers, 6 diverse proposers, best aggregator |
| Cost-effective | 2 layers (MoA-Lite) |
| Single top model | Self-MoA |
| Low latency | Single layer with strong aggregator |

## Limitations

- **High Time-to-First-Token**: Model cannot produce output until all layers complete
- **Cost**: Multiple LLM calls per query
- **Complexity**: More moving parts than single-model inference

## Future Directions

Active research areas mentioned in the papers:

- **Dynamic routing**: Query-based routing to cost-performance optimal LLMs
- **Adaptive depth**: Early stopping when consensus is reached
- **Streaming**: Chunk-wise aggregation to reduce TTFT (up to 93% reduction reported)

## Citation

If you use this library in research, please cite the original papers:

```bibtex
@article{wang2024mixture,
  title={Mixture-of-Agents Enhances Large Language Model Capabilities},
  author={Wang, Junlin and Wang, Jue and Athiwaratkun, Ben and Zhang, Ce and Zou, James},
  journal={arXiv preprint arXiv:2406.04692},
  year={2024}
}

@article{li2025rethinking,
  title={Rethinking Mixture-of-Agents: Is Mixing Different Large Language Models Beneficial?},
  author={Li, Wenzhe and others},
  journal={arXiv preprint arXiv:2502.00674},
  year={2025}
}
```
