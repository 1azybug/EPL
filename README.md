# EPL (Enhanced Position Layout)

This repository contains links to the experiments from our EMNLP Findings 2025 accepted paper ["Position IDs Matter: An Enhanced Position Layout for Efficient Context Compression in Large Language Models"](https://arxiv.org/abs/2409.14364).

![EPL Illustration](./EPL.png)



## core code

```python
    def get_uniform_position_ids(self, x_1, x_n, voco_num=2):
        # [S]
        tot = x_n-x_1+1
        ratio = tot/voco_num
        return torch.round(
            torch.linspace(x_1 + (ratio - 1) / 2, x_n - (ratio - 1) / 2, steps=voco_num, device=self.device)
            )
```

## Experiment
Note: We conduct experiments on **different branches**

In Table 4 and Table 7 (main experiments):
- The experimental code for ICAE is available at https://github.com/1azybug/UPL/tree/main
- The experimental code for 500xCompressor is available at https://github.com/1azybug/UPL/tree/500xCompressor

In Table 5 (multimodal experiments):
- The DPL experimental code for VoCo-Llama is available at https://github.com/1azybug/VoCo-LLaMA/tree/main
- The EPL experimental code for VoCo-Llama is available at https://github.com/1azybug/VoCo-LLaMA/tree/voco-upl

In Table 6 (ablation studies):
- The experimental code for ICAE without Position IDs in the second forward pass is available at https://github.com/1azybug/UPL/tree/ablation-icae-2nd-forward-pass
- The experimental code for 500xCompressor without Position IDs in the second forward pass is available at https://github.com/1azybug/UPL/tree/ablation-500x-2nd-forward-pass

