# Effective Variance Attention-enhanced Diffusion Model (EVADM) for Crop Field Aerial Image Super Resolution

## Overview 💥

This is the repository includes the models, methods and data developed in paper:

[**Effective variance attention-enhanced diffusion model for crop field aerial image super resolution**](https://doi.org/10.1016/j.isprsjprs.2024.08.017) that published in [**ISPRS Journal of Photogrammetry and Remote Sensing**](https://www.sciencedirect.com/journal/isprs-journal-of-photogrammetry-and-remote-sensing).
ResearchGate: [ResearchGate Article](https://www.researchgate.net/publication/383946370_Effective_variance_attention-enhanced_diffusion_model_for_crop_field_aerial_image_super_resolution)
中文简介：[基于方差注意力和隐扩散模型的无人机图像超分辨率](https://zhuanlan.zhihu.com/p/719676096)

The **Effective Variance Attention-enhanced Diffusion Model (EVADM)** is designed to enhance the resolution and quality of aerial imagery, particularly focusing on high-resolution cropland images. By leveraging emerging diffusion models (DM) and introducing the Variance-Average-Spatial Attention (VASA) mechanism, EVADM significantly improves image super-resolution (SR) tasks.
<div style="text-align: center;">
<img src="img/Fig Abst.jpg" data-align="center" width="80%">

Efficient VASA-enhanced Diffusion Model (EVADM) and the elevated image Variance after SR. 
</div>


## Main Contributions

- **Development of the CropSR Dataset**: Created a high-resolution aerial image dataset, namely CropSR, with over 321,000 samples for self-supervised SR training.
- **Introduction of Variance-Average-Spatial Attention (VASA)**: Designed a novel attention mechanism inspired by the trend of decreasing image variance with increasing flight altitude, enhancing SR model performance.
- **Efficient VASA-enhanced Diffusion Model (EVADM)**: Developed a robust model that leverages VASA to improve the quality of aerial imagery super-resolution.
- **Comprehensive Evaluation Metrics**: Introduced the Super-Resolution Relative Fidelity Index (SRFI) for a nuanced assessment of structural and perceptual similarities in SR outputs.

## Dataset

### CropSR (for training)
- **Description**: A high-resolution aerial image dataset comprising over 321,000 samples for self-supervised SR training.

### CropSR-FP/OR (for real-SR testing)
- **Description**: A combined dataset constructed from matched orthomosaic mapping (CropSR-OR) and fixed-point photographs (CropSR-FP).
- **Total Pairs**: More than 5,000 pairs.
- The test datasets can be accessed at [CropSR (for Crop Field Aerial Image Super Resolution), Mendeley Data](https://data.mendeley.com/datasets/fhvph562cn).

## Model Performance
  - Achieved a FID reduction of 14.6, and 27% boost of SRFI for ×2 real SR datasets.
  - Achieved a FID reduction of 8.0, and 6% boost of SRFI for ×4 real SR datasets.

## Generalization Ability

EVADM has demonstrated superior generalization capabilities on the open **Agriculture-Vision** dataset, highlighting its robustness across various aerial imagery tasks.

## Ablation Studies

The model's effectiveness is validated through ablation studies and feature-attention map analyses, providing insights into the mechanism of VASA and the SR process.

## Practical Applications

EVADM offers a promising approach for realistic aerial imagery super-resolution, showcasing high practicality for various downstream applications in agriculture and beyond.


## 💎 Go to /EVADM/ for demo & code

## Installation

All models were implemented using Python and the PyTorch framework and trained on an NVIDIA RTX 4090 GPU. The EVADM model is based on the LDM (Rombach et al., 2022), please refer to both [EVADM](https://github.com/HobbitArmy/EVADM/tree/main/EVADM) and [LDM](https://github.com/CompVis/latent-diffusion) setup instructions.
**Download weights to EVADM/weights/ folder from [weights](https://drive.google.com/drive/folders/1os4pK0CfyjW96FphApZpTjBziltEiTyz?usp=drive_link).**

**Go under EVADM/ and run for EVADM SR usage demo:**
```bash
python eva101_EVADM_infer.py
```

## SRFI metrics

For calculating the SRFI model, see :

```bash
eva102_SRFI_metrics.py
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

We thank all reviewers for their constructive feedback, which greatly contributed to the improvement of this project.
