# GANREST-A-Hybrid-Deep-Learning-Framework-for-Enhanced-Histopathological-Image-Classification-
GANREST: Hybrid deep learning framework combining SRGAN for super-resolution with fused ResNet-50/152 architectures. Achieves 99.53% accuracy on IDC and 99.84% on BreakHis datasets for breast histopathological image classification, outperforming individual ResNets. Enhances diagnostic reliability via improved image quality and multi-scale features.
# GANREST: Hybrid Framework for Enhanced Breast Cancer Histopathological Image Classification

## Background/Objective
Breast cancer is one of the most common cancers in women worldwide. Accurate and early diagnosis through histopathological examination significantly improves survival rates. However, low-resolution images and staining variations often reduce diagnostic reliability.

This study proposes **GANREST**, a hybrid deep learning framework that combines super-resolution generative adversarial networks (SRGAN) with fused ResNet architectures to improve both image quality and classification accuracy in breast histopathological images.

## Materials and Methods
- **Framework Overview**: GANREST first applies SRGAN to generate high-resolution versions of histopathological images from the publicly available BreakHis and IDC datasets.
- **Data Handling**: Data augmentation techniques (rotation, flipping, color jittering) were employed to handle class imbalance.
- **Models Compared**:
  - Individual ResNet-50, ResNet-101, and ResNet-152 models.
  - A novel hybrid ResNet-50 and ResNet-152 architecture with multi-scale feature fusion.
- **Evaluation**: All models were evaluated using 5-fold cross-validation.

## Results
The proposed hybrid ResNet-50–ResNet-152 model within GANREST achieved:
- **99.53%** accuracy on the IDC dataset
- **99.84%** accuracy on the BreakHis dataset

Precision, recall, and F1-score all exceeding 99%. These results substantially outperform individual ResNet backbones and previously published methods.

## Conclusion
By combining image super-resolution with multi-scale feature fusion, GANREST dramatically improves the automated classification of breast histopathological images. The framework shows strong potential for real-world clinical adoption, especially in settings with limited imaging quality, and may contribute to faster and more reliable breast cancer diagnosis.
