# Residual Train A/B (GPT-2 modular)

Two-arm ablation: does putting the residual causal filter **in the train loop** change what gets learned?

| Arm | Train loss | FILTER |
|-----|------------|--------|
| A | CE only | eval / end of each epoch |
| B | CE + residual READ on contrastive direction + random-direction complement penalty | same eval |

Same seeds, same `gpt2`, same modular prompts (`a + b =` mod 113), layer 6 last-token residual.

Full install and execute commands: **COMMANDS.txt**

```bash
./run_1x3090.sh             # one 24 GB 3090, 16 jobs sequential
GPUS="0 1" ./run_4x4090.sh   # two GPUs
./run_4x4090.sh             # four GPUs
# → results/ab_summary.md
```
