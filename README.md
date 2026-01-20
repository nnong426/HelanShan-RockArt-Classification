# HelanShan-RockArt-Classification
Code and Data description for Rock Art Classification using ResNet.
Due to cultural heritage protection policies, the original rock art images cannot be publicly shared. However, we provide the complete source code, a metadata format example, and the dataset directory structure to ensure reproducibility.

## 📁 Dataset Structure
To run the training script, please organize your dataset directory as follows:

```text
Project_Root/
│
├── classification_image/
│   ├── MetaData.csv       <-- Label file (format shown below)
│   ├── image_001.jpg      <-- Your images
│   ├── image_002.png
│   └── ...
│
├── resnet.py          <-- Main training script
├── std_resnet.py      <-- Model definition
├── metric.py
├── seed.py
└── ...
元数据文件MetaData.csv分为两列，file和label，每张图片命名需统一，如：image_001.jpg，在MetaData.csv里对图片进行标注，如：image_001.jpg，cow。
