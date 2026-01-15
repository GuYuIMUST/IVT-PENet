# IVT-PENet
An Information-Gated and Variance-Enhanced U-Net with Direction-Aware Positional Encoding and Channel-Spatial Transformer for Pulmonary Embolism Segmentation and Detection
# Requirements
pytoch==2.0.0 Newer version may need code adaptation.
simpleitk
tensorboardX
scikit-image
scikit-learn
tqdm
pandas
# Data Prepare:
1.This project utilizes the open-source Vessel and Lung Masks and lung masks provided by AANet, and we sincerely appreciate the authors' contribution to the community. In this repository, the dataset is organized under the directory IVT-PENet/PEData/CAD_PE_data/vessel. The vessel masks are named following the format {ID}.nii.gz (e.g., 001.nii.gz), while the lung masks are named {ID}_lungmask.nii.gz (e.g., 001_lungmask.nii.gz). If you use these vessel or lung annotations in your research, please ensure to cite both the original AANet paper and our IVT-PENet work.

2.Download CAD-PE dataset from https://ieee-dataport.org/open-access/cad-pe Put CTPA images, e.g. 001.nrrd, in ‘IVT-PENet/PEData/CAD_PE_data/image’. Put PE labels, e.g. 001RefStd.nrrd, in ‘IVT=PENet/PEData/CAD_PE_data/label’.

3.Run nifty_preprocess.py to preprocess data. The code will use lung masks to crop the lung region ROI in the CTPA images and labels, and save them back to nifty file in ‘./PEData/processed_itk’. The image size will be smaller for faster loading in training.
# Training and Inference:
Run train_enhancement.py for training. Use the argument –unique_name to give a name. At the end of training, the code calls the inference.py automatically, and save the result in ‘IVT-PENet/pred_itk/unique_name_sth50’. You may want to train several times with different seeds. The official evaluation protocol of CAD-PE is a little bit unreasonable. If two near ground-truth PEs are detected by one connected predicted PE, only one ground-truth is counted as TP, and vice-versa. Therefore, small randomness, that causes two near PEs connected to one or one PE breaked to two, will cause relatively big difference in FROC curve.
# Evaluation:
Run evaluation.py to evaluate and plot FROC curve. Change pred_root in line 279 to the directory of your result, i.e. ‘IVT-PENet/pred_itk/unique_name_sth50’.

The guidance and the code may still have some small errors. Feel free to contact me by email or issue.
