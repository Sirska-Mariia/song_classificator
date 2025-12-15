# song_classificator

This project implements a Multimodal Deep Learning model to classify music tracks into emotional quadrants based on Russell's Circumplex Model (Happy, Angry, Sad, Calm). The approach fuses audio features (MFCCs, Chroma, etc.) with lyrical semantics (BERT embeddings) using an Attention-Based mechanism.


The project is organized into 5 notebooks:

1) EDA.ipynb
  Performs exploratory data analysis.
3) text_processing.ipynb
Performs natural language processing tasks, including text cleaning, tokenization, and lemmatization.
4) audio_processing.ipynb
Extracts and aggregates 37 acoustic features, such as MFCCs, Chroma, and Spectral Contrast. It prepares the audio data by normalizing these statistical features for model training
6) modeling.ipynb
We trained 4 ML algorithms (Logistic regression, SVN, Random forest and KNN) for text-only, audio-only and text+audio baselines. Logistic regression for text+audio model has the best results.
7) dl_modeling.ipynb
We trained deep learning models (tried text-only, audio-only models and combined model with a BERT encoder (for text) and an MLP (for audio)).




|----------|-------------|
| **`EDA.ipynb`** | Exploratory Data Analysis. Visualization of class distribution, correlation matrices, and t-SNE projections of audio/text features. |
| **`text_processing.ipynb`** | Natural Language Processing steps: cleaning, lemmatization, and sentiment analysis. Preparation of data for BERT. |
| **`audio_processing.ipynb`** | Extraction and normalization of 37 handcrafted audio features (MFCC, Spectral Contrast, RMS, etc.) using `librosa`. |
| **`modeling.ipynb`** | **Classical ML Baselines.** Training 4 algorithms (Logistic Regression, SVM, Random Forest, KNN) across three setups: *Text-only*, *Audio-only*, and *Text+Audio*. |
| **`dl_modeling.ipynb`** | **Deep Learning Model (SOTA).** Implementation of the Multimodal Classifier with Attention Fusion, MixUp regularization, and OneCycleLR scheduler. |

---

## Model Architectures

### 1. Classical Machine Learning Baselines (`modeling.ipynb`)
We evaluated traditional algorithms to establish a performance baseline.
* **Algorithms:** Logistic Regression, SVM, Random Forest, k-Nearest Neighbors (KNN).
* **Modes:** Unimodal (Text or Audio) and Multimodal (Concatenated features).

### 2. Deep Learning Approach (`dl_modeling.ipynb`)
The final model is a custom neural network designed to handle modality imbalance.

* **Audio Branch:** MLP with Residual Connections (Input 37 dim → 128 dim).
* **Text Branch:** `bert-base-uncased` (Fine-tuning last 2 layers) with projection (768 dim → 128 dim).
* **Fusion Mechanism:** Attention-Based Gated Fusion.
    * The model dynamically calculates a scalar weight $\alpha$ for each sample.
    * Formula: $F = \alpha \cdot T + (1-\alpha) \cdot A$
* **Regularization:** Dropout (0.4) and MixUp (interpolating inputs and labels to prevent overfitting).


### The "Quiet Emotion" Paradox
One of the key findings during EDA was the acoustic similarity between **Q3 (Sad/Depressed)** and **Q4 (Calm/Relaxed)**. Both classes share low energy and tempo, making them indistinguishable for audio-only models.

* **Audio-only:** Distance between Q3 and Q4 centroids was extremely low (0.6), leading to high confusion.
* **Text:** Semantic analysis revealed distinct vocabularies:
    * *Q3 Keywords:* "Gone", "Hurt", "Cry", "Pain".
    * *Q4 Keywords:* "Peace", "Angel", "Christmas", "Star".
* **Conclusion:** The multimodal approach significantly improves classification.


## Usage
1. Clone the repository:
   ```bash
   git clone [https://github.com/Sirska-Mariia/song_classificator.git](https://github.com/Sirska-Mariia/song_classificator.git)
