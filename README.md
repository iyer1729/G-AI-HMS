# G-AI-HMS: Generative AI-Enabled Human Motion Simulation

**G-AI-HMS** is an adaptation of [MotionGPT](https://github.com/OpenMotionLab/MotionGPT), tailored for ergonomic motion simulation, task-specific prompt alignment, and biomechanical analysis. This repository enables high-fidelity motion generation from natural language using AI-generated prompts and evaluates alignment against MediaPipe-extracted ground truth using standard metrics (MPJPE, PA-MPJPE, DTW). This implementation relies on the original [HumanML3D dataset](https://github.com/EricGuo5513/HumanML3D), pretrained weights from the [text-to-motion model](https://github.com/EricGuo5513/text-to-motion), the raw and processed motion data in the [`data/`](./data) directory, and the developed utility scripts in the [`scripts/`](./gaihms_analysis) directory for preprocessing and analysis.

## Acknowledgment

This project is built upon the MotionGPT framework, originally developed by Jiang et al. (2024) and Chen et al. (2023). We gratefully acknowledge the authors for their foundational contributions to language-driven human motion generation and for making their code publicly available.

> Jiang, B., Chen, X., Liu, W., Yu, J., Yu, G., & Chen, T. (2024). *MotionGPT: Human Motion as a Foreign Language*. Advances in Neural Information Processing Systems, 36.

> Chen, X., Jiang, B., Liu, W., Huang, Z., Fu, B., Chen, T., & Yu, G. (2023). *Executing Your Commands via Motion Diffusion in Latent Space*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 18000–18010.
