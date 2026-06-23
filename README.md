MS-TGMF

MS-TGMF: Multi-Scale Temporal Gated Memory Fusion for Occlusion-Robust LiDAR-BEV Lane Detection

This repository provides the implementation of MS-TGMF, a LiDAR-BEV lane detection method built upon the official K-Lane Proj28-GFC-T3-RowRef framework.

MS-TGMF introduces three feature enhancement modules between the backbone and the RowRef lane detection head:

* TGMF: Temporal Gated Memory Fusion
* MSCE: Multi-Scale Context Enhancement
* DCRB: Directional Context Refinement Block

The proposed method aims to improve the robustness of LiDAR-BEV lane detection under challenging scenarios, including occlusion, sparse point clouds, curves, and merging/diverging lanes.

⸻

Overview

LiDAR-BEV lane detection may suffer from local discontinuity and missed lane responses when lane markings are occluded or sparsely observed. MS-TGMF enhances BEV lane features from three complementary perspectives.

First, TGMF uses adjacent historical frames to provide temporal structural compensation for the current frame. Second, MSCE captures local and larger-range spatial context using multi-branch convolution and dilated convolution. Third, DCRB refines directional structure using horizontal and vertical strip convolutions.

The overall framework keeps the original RowRef lane detection head and training objective unchanged.

⸻

Framework

The overall architecture is shown below.

⸻

Project Structure

MS-TGMF-release/
├── assets/                         # Figures and visualization examples
├── baseline/                       # K-Lane baseline framework and modified model code
│   ├── datasets/                   # Dataset loading and registration
│   ├── engine/                     # Training and evaluation engine
│   ├── models/                     # Backbone, head, point encoder, and network modules
│   ├── utils/                      # Utility functions
│   └── vis/                        # Visualization tools
├── configs/                        # Configuration files
├── docs/                           # K-Lane documentation
├── tools/                          # Additional tools
│   ├── measure_model_efficiency.py # Params/FLOPs/FPS measurement
│   ├── pickle_to_bev_png.py
│   └── vis_compare_35.py
├── train.py                        # Training script
├── validate.py                     # Evaluation script
├── visualize.py                    # Visualization script
├── requirements.txt
├── LICENSE
└── README.md

⸻

Installation

Create a conda environment:

conda create -n mstgmf python=3.8 -y
conda activate mstgmf

Install dependencies:

pip install -r requirements.txt

Please make sure that PyTorch, CUDA, and other dependencies are compatible with your hardware environment.

⸻

Dataset Preparation

This project uses the K-Lane dataset. Please download and prepare the dataset following the official K-Lane instructions.

A typical dataset structure is expected as:

data/
└── KLane/
    ├── train/
    ├── test/
    └── ...

After preparing the dataset, please modify the dataset path in the corresponding configuration file under configs/.

⸻

Training

To train the model, run:

python train.py configs/Proj28_GFC-T3_RowRef_82_73.py

The final MS-TGMF configuration is based on the Proj28-GFC-T3-RowRef framework. The proposed modules are inserted between the backbone and the RowRef lane detection head.

For ablation studies, please modify the corresponding module switches in the configuration file.

⸻

Evaluation

To evaluate a trained checkpoint, run:

python validate.py configs/Proj28_GFC-T3_RowRef_82_73.py

Please set the checkpoint path correctly in the configuration file before evaluation.

⸻

Model Efficiency

To measure model complexity, including Params, FLOPs, and FPS, run:

python tools/measure_model_efficiency.py

The measured results can be used to analyze the trade-off between detection performance and computational cost.

⸻

Main Results

The following results are obtained on the K-Lane validation setting with the reproduced Proj28-GFC-T3-RowRef baseline.

Method	            Epoch	Total	       Night	Merging	Curve	      Occ4-6
GFC-T3 Reported	       60	82.100	82.000	81.000	78.000	75.900
GFC-T3 Reproduced	       35	81.309	81.468	78.398	73.932	76.288
+TGMF	                   35	81.410	81.639	78.630	74.168	77.870
+TGMF + MSCE	       35	81.350	81.650	78.420	73.870	78.360
MS-TGMF, alpha=0.7	 35	81.497	81.626	79.167	74.693	79.048

Compared with the reproduced baseline, MS-TGMF improves performance in challenging scenarios, especially under severe occlusion. The Occ4-6 score is improved from 76.288 to 79.048.

⸻

Visualization

Some BEV visualization examples and module figures are provided in the assets/ directory.

assets/
├── architecture_v11.png
├── TGMF.drawio.pdf
├── MSCE.drawio.pdf
├── DCRB.drawio.pdf
├── vis_bev_pseudo_feat.png
└── vis_bev_pseudo_occ.png

⸻

Notes

This repository is built upon the official K-Lane baseline framework. The original documentation is kept in the docs/ directory. The proposed MS-TGMF modules are implemented within the baseline framework.

This code is intended for academic research and reproducibility.

⸻

Citation

If this work is useful for your research, please consider citing our paper after publication.

@article{mstgmf2026,
  title={MS-TGMF: Multi-Scale Temporal Gated Memory Fusion for Occlusion-Robust LiDAR-BEV Lane Detection},
  author={Sun, Shi},
  journal={TBD},
  year={2026}
}

⸻

Acknowledgement

This project is based on the official K-Lane framework. We thank the authors of K-Lane for releasing the dataset and baseline code.