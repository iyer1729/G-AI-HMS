![IMG_5239](https://github.com/user-attachments/assets/e38c35c0-6f93-4271-b5c6-c5cbf44d3767)![IMG_5239](https://github.com/user-attachments/assets/aa3940a2-2f05-4ade-b074-df5103cd6712)# G-AI-HMS: Generative AI-Enabled Human Motion Simulation

**G-AI-HMS** is a fork of [MotionGPT](https://github.com/OpenMotionLab/MotionGPT), adapted for ergonomic motion simulation, task-specific prompt alignment, and biomechanical evaluation. This repository enables high-fidelity motion generation from natural language using AI-generated prompts and evaluates alignment against MediaPipe-extracted ground truth using standard metrics (MPJPE, PA-MPJPE, DTW, Cosine Similarity). This implementation relies on the original [HumanML3D dataset](https://github.com/EricGuo5513/HumanML3D), pretrained weights from the [text-to-motion model](https://github.com/EricGuo5513/text-to-motion), and the developed utility scripts in the [`mGPT/data`](./mGPT/data) directory for preprocessing, analysis, and visualization.

## Acknowledgment

This project is built upon the MotionGPT framework, originally developed by Jiang et al. (2024) and Chen et al. (2023). We gratefully acknowledge the authors for their foundational contributions to language-driven human motion generation and for making their code publicly available.

> Jiang, B., Chen, X., Liu, W., Yu, J., Yu, G., & Chen, T. (2024). *MotionGPT: Human Motion as a Foreign Language*. Advances in Neural Information Processing Systems, 36.

> Chen, X., Jiang, B., Liu, W., Huang, Z., Fu, B., Chen, T., & Yu, G. (2023). *Executing Your Commands via Motion Diffusion in Latent Space*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 18000–18010.
