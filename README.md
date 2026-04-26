# HelanShan-RockArt-Classification (贺兰山岩画分类)

Notice on Code and Data Availability (代码与数据开源声明)

Due to the strict copyright regulations regarding the Helan Mountain Cultural Heritage and the protection of Intellectual Property  associated with this ongoing research project, the complete source code and raw rock art image datasets are not publicly available. 

The directory structure and metadata format are provided below solely for reference purposes to illustrate the experimental setup and methodology. Researchers requiring access to desensitized data or core scripts for academic validation purposes may contact the corresponding author via email. Access will be evaluated on a case-by-case basis.

---

鉴于贺兰山岩画文化遗产的严格文物保护规定，以及本课题后续研究的知识产权保护要求，本项目相关的完整源代码与原始图像数据集暂不在线公开分享。

下方仅提供本研究的实验目录结构与元数据格式示例，以供同行了解我们的实验设计与方法论。如有正当的学术复现或验证需求，可通过邮件联系作者申请使用。

---

## 📁 Dataset & Project Structure (参考目录结构)

To understand our experimental workflow, the project directory was organized as follows during the research:
(为了解我们的实验流程，本研究在开展时的项目目录组织如下：)

```text
Project_Root/
│
├── classification_image/
│   ├── MetaData.csv       <-- Label file (format shown below)
│   ├── image_001.jpg      <-- Raw rock art images (Not Provided)
│   ├── image_002.png      <-- Raw rock art images (Not Provided)
│   └── ...
│
├── resnet.py          <-- Main training script (Not Provided)
├── std_resnet.py      <-- Model definition (Not Provided)
├── metric.py          <-- Evaluation metrics (Not Provided)
├── seed.py            <-- Random seed configuration (Not Provided)
└── ...
元数据文件MetaData.csv分为两列，file和label，每张图片命名需统一，如：image_001.jpg，在MetaData.csv里对图片进行标注，如：image_001.jpg，cow。
