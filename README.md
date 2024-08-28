# Effective Variance Attention-enhanced Diffusion Model for Crop Field Aerial Image Super Resolution (EVADM)

## Overview

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
- The test datasets can be accessed at [CropSR (for Crop Field Aerial Image Super Resolution), Mendeley Data](https://data.mendeley.com/preview/fhvph562cn?a=2477f9a3-b71e-474b-b072-150c85a2a512).

## Model Performance

- **FID Reductions**: 
  - Achieved a reduction of 14.6 for ×2 real SR datasets.
  - Achieved a reduction of 8.0 for ×4 real SR datasets.
  
- **SRFI Improvements**: 
  - 27% boost for ×2 datasets.
  - 6% boost for ×4 datasets.

## Generalization Ability

EVADM has demonstrated superior generalization capabilities on the open **Agriculture-Vision** dataset, highlighting its robustness across various aerial imagery tasks.

## Ablation Studies

The model's effectiveness is validated through ablation studies and feature-attention map analyses, providing insights into the mechanism of VASA and the SR process.

## Practical Applications

EVADM offers a promising approach for realistic aerial imagery super-resolution, showcasing high practicality for various downstream applications in agriculture and beyond.


# !! Upcoming 👇

## Installation

All models were implemented using Python and the PyTorch framework and trained on an NVIDIA RTX 4090 GPU. The diffusion model is based on the LDM (Rombach et al., 2022), while the regression- and GAN-based models were derived from MMEDIT (MMEditing, 2022). To install the necessary dependencies for the Effective Variance Attention-enhanced Diffusion Model (EVADM), please refer to the [LDM](https://github.com/CompVis/latent-diffusion) setup instructions.

```bash
pip install -r requirements.txt
```

## Usage

For testing the EVADM model, execute the following command:

```bash
python test.py --model <path_to_trained_model> --img_path <path_to_img>
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

We thank all reviewers for their constructive feedback, which greatly contributed to the improvement of this project.