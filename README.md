# Speech Emotion Recognition using CNN-RNN Hybrid Networks

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Librosa](https://img.shields.io/badge/Librosa-Audio%20Processing-green?style=for-the-badge)

> **Try the live demo:** [Speech Emotion Recognition on Hugging Face](https://huggingface.co/spaces/sjagird1/SER)

A hybrid deep learning model combining Convolutional Neural Networks (CNNs) and dense layers to perform **Speech Emotion Recognition (SER)**. The model classifies six primary emotions -- **Anger, Disgust, Fear, Happy, Neutral, and Sad** -- from audio input using mel-spectrogram features, achieving **90.26% test accuracy**.

---

## Project Structure

```
Speech Emotion Recognition using CNN-RNN Hybrid Model/
|-- Conv2D_RNN_Model.py                                    # Full training pipeline
|-- README.md                                              # Project documentation
|-- requirements.txt                                       # Python dependencies
|-- .gitignore                                             # Git ignore rules
|-- SER POSTER.pdf                                         # Conference-style poster
|-- Speech_Emotion_Recognition_using_CNN_RNN_Hybrid_Network.pdf  # Detailed report
```

---

## Model Architecture

```
Audio Input (4s clip, 22050 Hz)
         |
         v
  Mel-Spectrogram Extraction (128 mel bins x 173 time steps)
         |
         v
+---------+----------+
| Conv2D(64, 3x3)   |  Block 1: BatchNorm -> ReLU -> MaxPool(2x2) -> Dropout(0.2)
+---------+----------+
         |
+---------+----------+
| Conv2D(128, 3x3)  |  Block 2: BatchNorm -> ReLU -> MaxPool(2x2) -> Dropout(0.2)
+---------+----------+
         |
+---------+----------+
| Conv2D(256, 3x3)  |  Block 3: BatchNorm -> ReLU -> MaxPool(2x2) -> Dropout(0.2)
+---------+----------+
         |
+---------+----------+
| Conv2D(512, 3x3)  |  Block 4: BatchNorm -> ReLU -> MaxPool(2x2) -> Dropout(0.2)
+---------+----------+
         |
      Flatten
         |
  Dense(512) -> BatchNorm -> ReLU -> Dropout(0.3)
         |
  Dense(256) -> BatchNorm -> ReLU -> Dropout(0.3)
         |
  Dense(6, Softmax)  -->  [Anger | Disgust | Fear | Happy | Neutral | Sad]
```

**Optimizer:** AdamW (lr=0.001, weight_decay=0.0001)
**Loss:** Categorical Crossentropy
**Callbacks:** EarlyStopping (patience=15), ReduceLROnPlateau (factor=0.1), ModelCheckpoint

---

## Dataset

Combined dataset of **10,898 samples** across three public corpora:

| Dataset | Samples | Type | Source |
|---------|---------|------|--------|
| RAVDESS | 1,440 | Acted speech | [Zenodo](https://zenodo.org/record/1188976) |
| TESS | 2,800 | Acted speech | [University of Toronto](https://tspace.library.utoronto.ca/handle/1807/24487) |
| CREMA-D | 7,442 | Crowd-sourced | [GitHub](https://github.com/CheyneyComputerScience/CREMA-D) |

---

## Data Preprocessing

Each audio sample undergoes the following pipeline:

1. **Resampling** to 22,050 Hz
2. **Duration normalization** -- pad or truncate to 4 seconds
3. **Mel-spectrogram extraction** -- 128 mel bins, 2048-sample FFT, 512-sample hop length
4. **Log scaling** and per-sample normalization (zero mean, unit variance)
5. **Clipping** extreme values to [-5, 5]

**Data augmentation** (applied during training):
- **Time stretching** (rate=1.2)
- **Pitch shifting** (+2 semitones)

---

## Results

| Metric | Value |
|--------|-------|
| **Training Accuracy** | 98.96% |
| **Validation Accuracy** | 90.26% |
| **Test Accuracy** | 90.26% |
| **Test Loss** | 0.3350 |

### Per-Class F1-Scores

| Emotion | F1-Score |
|---------|----------|
| Anger | 0.93 |
| Disgust | 0.89 |
| Fear | 0.88 |
| Happy | 0.91 |
| Neutral | 0.92 |
| Sad | 0.88 |

---

## Installation

```bash
git clone https://github.com/Samudyata/Speech-Emotion-Recognition-CNN-RNN.git
cd Speech-Emotion-Recognition-CNN-RNN

pip install -r requirements.txt
```

---

## Usage

1. Organize your audio dataset into folders by emotion label:
   ```
   Emotions/
   |-- Anger/
   |-- Disgust/
   |-- Fear/
   |-- Happy/
   |-- Neutral/
   |-- Sad/
   ```

2. Update the `BASE_PATH` in `Conv2D_RNN_Model.py` to point to your dataset.

3. Run training:
   ```bash
   python Conv2D_RNN_Model.py
   ```

4. The trained model is saved as `final_model_conv2d_1K_1.keras`.

---

## Future Work

- **Integrate MFCC and Chroma features** for enriched audio representation
- **Transformer-based architectures** for capturing long-range temporal dependencies
- **Cross-domain generalization** across accents, languages, and environments
- **Real-time inference** optimized for deployment in virtual assistants

---

## Authors

- **Samudyata Sudarshan Jagirdar**
- **Mahesh Divakaran Namboodiri**
- **Sayantika Paul**

---

## License

This project is intended for educational and research purposes.
