# References

The four-page core text uses the numbered citations below. Versions were checked on September 8, 2026. These references and the figures are separate from the four-page core.

1. Shengbang Tong, David Fan, John Nguyen, et al. **Beyond Language Modeling: An Exploration of Multimodal Pretraining.** arXiv:2603.03276v1, March 3, 2026. [Paper and introduction](https://arxiv.org/html/2603.03276v1). Broad motivation from visual experience and controlled design-space studies; not evidence for our steering results.
2. Basile Terver, Tsung-Yen Yang, Jean Ponce, Adrien Bardes, and Yann LeCun. **What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?** arXiv:2512.24497v4, September 2, 2026; first submitted December 2025. [Paper](https://arxiv.org/html/2512.24497v4). Methodological references: Table 1 and Section 4 (design choices), Figures 3-7 (task-wise ablations), Table 2 (final comparison), Appendix F/Table 10 (planner settings), Appendix G.2 (aggregation), and G.3 (proxy metrics).
3. Sonia Joseph, Quentin Garrido, Randall Balestriero, Matthew Kowal, Thomas Fel, Shahab Bakhtiari, Blake Richards, and Mike Rabbat. **Interpreting Physics in Video World Models.** arXiv:2602.07050v1, February 4, 2026. [Paper](https://arxiv.org/html/2602.07050v1).
4. Miranda Muqing Miao, Subin Kim, Brandon Yang, and Lyle Ungar. **Contrastive Conceptor Activation Steering (COAST): Unlocking Vision-Language-Action Models through Hidden States.** arXiv:2605.17144v1, May 16, 2026. [Paper](https://arxiv.org/html/2605.17144v1).
5. Daniel Wurgaft, Can Rager, Matthew Kowal, et al. **Manifold Steering Reveals the Shared Geometry of Neural Network Representation and Behavior.** arXiv:2605.05115v1, May 6, 2026. [Paper](https://arxiv.org/html/2605.05115v1).
6. Guo An, Zijing Wu, Honghua Dong, et al. **Diagnosing JEPA World Models with Action-Conditioned Predictive Consistency.** arXiv:2608.12939v1, August 13, 2026. [Paper](https://arxiv.org/html/2608.12939v1).


7. Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, and Randall Balestriero. **LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels.** arXiv:2603.19312v3, June 3, 2026. [Paper](https://arxiv.org/html/2603.19312v3).
8. Lukas Kuhn, Lucas Maes, Giuseppe Serra, et al. **LeVJEPA: Efficient & Scalable Video Pretraining without the Heuristics.** arXiv:2608.27395v1, August 27, 2026. [Paper](https://arxiv.org/html/2608.27395v1).
9. Yuntian Gao and Xiangyu Xu. **Fast LeWorldModel.** arXiv:2606.26217. [Paper](https://arxiv.org/html/2606.26217).
10. Zhi Song, Ximing Xing, Zhenchao Tang, et al. **Branch-JEPA: Finite-Support Predictive Distributions for JEPA World Models.** arXiv:2607.05238v3, August 3, 2026. [Paper](https://arxiv.org/html/2607.05238v3). Earlier versions used the title MoP-JEPA; this draft cites v3.
11. Zhengxuan Wu, Aryaman Arora, Zheng Wang, et al. **ReFT: Representation Finetuning for Language Models.** arXiv:2404.03592. [Paper](https://arxiv.org/abs/2404.03592).
12. Fred Zhang and Neel Nanda. **Towards Best Practices of Activation Patching in Language Models: Metrics and Methods.** arXiv:2309.16042v2, January 17, 2024. [Paper](https://arxiv.org/abs/2309.16042v2).

## Plotting implementation source

The JEPA-WM repository at commit `13cf1d9c7e476f53c17714d2e0f1dc239a883ce0` directly imports Matplotlib and Seaborn in its [design-choice plotting script](https://github.com/facebookresearch/jepa-wms/blob/13cf1d9c7e476f53c17714d2e0f1dc239a883ce0/app/plan_common/plot/logs_plan_joint_per_design_choice.py) and [training-curve utilities](https://github.com/facebookresearch/jepa-wms/blob/13cf1d9c7e476f53c17714d2e0f1dc239a883ce0/app/plan_common/plot/logs_plan_joint_unif_utils.py). The latter explicitly exports PDF curves. This confirms the libraries used by the released plotting workflow; it does not identify the authoring tool for every conceptual diagram in the paper.
