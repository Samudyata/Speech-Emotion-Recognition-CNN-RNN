# Speech Emotion Recognition using CNN-RNN Hybrid Networks

## 📘 Overview
This project implements a **hybrid deep learning model** combining Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) to perform **Speech Emotion Recognition (SER)**. The model is trained on a large corpus of emotional speech data to classify six primary emotions: **Anger, Disgust, Fear, Happy, Neutral, and Sad**.

We leverage **mel-spectrogram features** for spatial representation and LSTM layers for temporal dependency modeling. The hybrid architecture demonstrates high accuracy and robustness across diverse speakers and conditions.

---

## 🎯 Objectives
- Enhance human-computer interaction via emotion-aware speech interfaces.
- Develop a hybrid CNN-RNN model that leverages both spectral and temporal features of audio.
- Evaluate model performance on combined datasets: **RAVDESS**, **TESS**, and **CREMA-D**.

---

## 🧠 Model Architecture
- **CNN Block**: Extracts hierarchical spatial features from mel-spectrograms.
- **RNN Block**: Uses stacked LSTM layers to model temporal patterns in speech.
- **Dense Layers**: Fully connected layers leading to a softmax classifier.
- **Regularization**: Dropout, batch normalization, early stopping, and learning rate decay.

---

## 🗂️ Dataset
Combined dataset of **10,898 samples** across the following:
| Dataset  | Samples | Type     |
|----------|---------|----------|
| RAVDESS  | 1,440   | Acted    |
| TESS     | 2,800   | Acted    |
| CREMA-D  | 7,442   | Natural  |

Each sample was preprocessed using:
- Resampling to 16 kHz
- Spectral subtraction for noise reduction
- Mono conversion and normalization
- Fixed-length padding/truncation
- Mel-spectrogram extraction with logarithmic scaling and normalization

---

## 📊 Results

| Metric             | Value     |
|--------------------|-----------|
| Training Accuracy  | 98.96%    |
| Validation Accuracy| 90.26%    |
| Test Accuracy      | 90.26%    |
| Test Loss          | 0.3350    |

### Per-Class F1-Scores
| Emotion  | F1-Score |
|----------|----------|
| Anger    | 0.93     |
| Disgust  | 0.89     |
| Fear     | 0.88     |
| Happy    | 0.91     |
| Neutral  | 0.92     |
| Sad      | 0.88     |

---

## 🧪 Future Work
- **Integrate MFCC and Chroma Features**: Incorporate Mel-Frequency Cepstral Coefficients and chroma-based harmonic features for a more enriched representation of emotional cues in speech.
- **Transformer-based Architectures**: Explore transformer models for capturing long-range temporal dependencies more effectively than LSTMs.
- **Cross-Domain Generalization**: Evaluate the model across diverse accents, languages, and environmental conditions to improve real-world robustness.
- **Real-Time Inference & Deployment**: Optimize for lightweight architectures enabling deployment in real-time applications like virtual assistants or customer service bots.

---

## ✍️ Authors
- **Samudyata Sudarshan Jagirdar**  
- **Mahesh Divakaran Namboodiri**  
- **Sayantika Paul**  

---

## 🌐 Hugging Face Demo
Try the live demo on Hugging Face Spaces:  
🔗 [Speech Emotion Recognition Demo](https://huggingface.co/spaces/sjagird1/SER)

