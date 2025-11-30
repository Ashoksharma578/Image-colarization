# Automatic Image Colorization using Deep Learning (U-Net)

This project automatically colorizes grayscale (black & white) images using a Deep Learning model based on U-Net architecture. The model predicts color channels from grayscale input images to generate realistic colorized outputs.

---

## Project Features

- Image colorization using U-Net architecture
- Uses LAB color space for better color prediction
- GPU (CUDA) support for faster training
- Dataset handling with corrupted image skipping
- Training checkpoint saving
- Dataset size control for fast experimentation
- Model evaluation using MSE
- Clean modular code structure

---

## Dataset

This project uses COCO 2017 Dataset.

Because the dataset is very large, it is NOT included in this repository.

Download manually from:
https://cocodataset.org

Use the training images folder:
train2017/

project_root/
images/
train/
train2017/
val/
val2017/


---

## How to Run the Project

### Step 1: Clone GitHub Repository

git clone https://github.com/Ashoksharma578/Image-colarization.git
cd Image-Colourization


python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

python check_gpu.py
python train.py

saved_models/

python test.py
Colourized_resultjjpg


**Model Architecture**
Encoder: Feature extraction
Bottleneck: Deep representation
Decoder: Reconstruction
Skip Connections: Preserve edges and details

**Loss Function**
Mean Squared Error (MSE) loss is used to match predicted color channels to true colors.

**Evaluation Metrics**
MSE (Mean Squared Error)

**Hardware Recommendation**
GPU: NVIDIA CUDA GPU recommended
CPU training possible but slow


---
