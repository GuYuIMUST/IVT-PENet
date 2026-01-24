# IVT-PENet
An Information-Gated and Variance-Enhanced U-Net with Direction-Aware Positional Encoding and Channel-Spatial Transformer for Pulmonary Embolism Segmentation and Detection

![image](https://github.com/GuYuIMUST/IVT-PENet/blob/1b946317d48200d2f80a26a61233d41f4b062a48/images/figure1.png)

The full implementation will be released after the associated paper is accepted.
# Overview

**IVT-PENet** is a unified deep learning framework for pulmonary embolism detection and segmentation in CTPA images.

Built upon a 3D U-Net architecture, the model is designed to effectively handle the challenges of small emboli, low contrast, and complex three-dimensional vascular structures. By integrating enhanced feature representation and contextual modeling mechanisms, IVT-PENet achieves strong performance on the CAD-PE dataset, demonstrating reliable embolus localization and accurate volumetric segmentation.

# Key Innovations
**1.Direction-Aware Positional Encoding (DAPE)** 

    A spatially aware encoding module that:
    
        Injects learnable directional positional information along orthogonal axes
        
        Enhances spatial consistency across the encoder–decoder pathway
        
        Preserves elongated and vessel-aligned embolic structures during downsampling
        
    This enables robust modeling of spatial orientation and structural continuity.
**Information-Gated and Variance-Enhanced Multi-Scale Feature Aggregation (IG-VEASPP)** 

    A multi-scale feature aggregation module that:
    
        Applies information gating to suppress redundant feature responses
        
        Uses variance-based attention to emphasize subtle and low-contrast emboli
        
        Aggregates contextual features at multiple receptive field scales
        
    This improves discrimination of small emboli in complex vascular regions.
**Channel–Spatial Attention Transformer (CSAT)**

    An efficient transformer-based bottleneck module that:
    
        Models channel-wise dependencies to enhance semantic relevance
        
        Captures long-range spatial context in three-dimensional space
        
        Avoids the high computational cost of full 3D self-attention
        
    This strengthens global contextual understanding with minimal overhead.
    
# Main Results
Sensitivity of different network models at ε=0, 2,and5 mm
| Model | ε=0 | ε=2 | ε=5 |
|------|------------------|------------------|------------------|
| UA-2D | 0.515 | 0.596 | 0.633 |
| UA-2.5D | 0.392 | 0.477 | 0.511 |
| ASU-Mayo | 0.362 | 0.395 | 0.456 |
| FUM-Mvlab | 0.269 | 0.342 | 0.411 |
| UA-3D | 0.7 | 0.789 |  0.822|
| AANet | 0.753 | 0.859 | 0.871 |
| Dual-Hop | 0.769 | 0.847 | 0.866 |
| TSNet | 0.761 | 0.85 | 0.878 |
| **IVT-PENet (Ours)** | **0.815** | **0.895** | **0.911** |

# Data Prepare:
1.This project utilizes the open-source Vessel and Lung Masks and lung masks provided by AANet, and we sincerely appreciate the authors' contribution to the community. In this repository, the dataset is organized under the directory IVT-PENet/PEData/CAD_PE_data/vessel. The vessel masks are named following the format {ID}.nii.gz (e.g., 001.nii.gz), while the lung masks are named {ID}_lungmask.nii.gz (e.g., 001_lungmask.nii.gz). If you use these vessel or lung annotations in your research, please ensure to cite both the original AANet paper and our IVT-PENet work.

2.Download CAD-PE dataset from https://ieee-dataport.org/open-access/cad-pe Put CTPA images, e.g. 001.nrrd, in ‘IVT-PENet/PEData/CAD_PE_data/image’. Put PE labels, e.g. 001RefStd.nrrd, in ‘IVT-PENet/PEData/CAD_PE_data/label’.

3.Run nifty_preprocess.py to preprocess data. The code will use lung masks to crop the lung region ROI in the CTPA images and labels, and save them back to nifty file in ‘./PEData/processed_itk’. The image size will be smaller for faster loading in training.
# Training and Inference:
Run train_enhancement.py for training. Use the argument –unique_name to give a name. At the end of training, the code calls the inference.py automatically, and save the result in ‘IVT-PENet/pred_itk/unique_name_sth50’. You may want to train several times with different seeds. The official evaluation protocol of CAD-PE is a little bit unreasonable. If two near ground-truth PEs are detected by one connected predicted PE, only one ground-truth is counted as TP, and vice-versa. Therefore, small randomness, that causes two near PEs connected to one or one PE breaked to two, will cause relatively big difference in FROC curve.
# Evaluation:
Run evaluation.py to evaluate and plot FROC curve. Change pred_root in line 279 to the directory of your result, i.e. ‘IVT-PENet/pred_itk/unique_name_sth50’.

The guidance and the code may still have some small errors. Feel free to contact me by email or issue.
