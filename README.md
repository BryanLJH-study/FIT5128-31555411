# Multi-Modal Depression Detection (Master Thesis S2 2024)

<<<<<<< HEAD
This repository contains the code and resources for the final version of the **Multi-Modal Depression Detection** project. This work was conducted as part of a Master's thesis in 2024, exploring interpretable machine learning models for automatic depression detection.
[Link to the thesis document](https://drive.google.com/file/d/1ey-tDiRTFwYGDblzGMOqV-zRHtNvK28D/view?usp=sharing).

---

## Quick Start Guide

### 1. Getting the Data

#### 1.1 Obtain Permission

Before accessing the datasets, ensure you have the necessary permissions:

* [DAIC-WOZ Dataset](https://dcapswoz.ict.usc.edu/)
* **Monash Behavioural Analysis Project:** Contact the relevant personnel at Monash University.

#### 1.2 Download Preprocessed Features

* [Preprocessed Features (DAIC-WOZ and MBADD)](https://drive.google.com/drive/folders/1LlTXTPLFv457x65JmmFEP-kZ7I0I9hTE?usp=sharing)

Alternatively, you can process the original data yourself by downloading the original DAIC-WOZ dataset and MBADD videos. Refer to the [feature_extraction directory README](./feature_extraction/README.md) for detailed instructions.

---

### 2. Analyzing the Data

The **analysis** directory contains exploratory notebooks primarily focused on the DAIC-WOZ dataset.

**Note:** These notebooks were part of initial exploration and are not directly relevant to the final thesis results. However, they were retained for potential future use.

---

### 3. Training, Testing, and Comparing Models

The **depression\_detection** directory includes code for model training and evaluation:

* **Modules**: Handles data reading, preprocessing, and training/testing workflows.

* **Models**: Contains final implementations for...

  * **AU-only model**
  * **MFCC-only model**
  * **AU-MFCC combined model**
  
  Each model includes:
  * `model.py`: Model architecture definition.
  * `training_pipeline.py`: Training and testing configurations.
  * `/checkpoints`: Saved model weights.
  * `/logs`: Logs for training, validation, and testing.

* **Model Comparison**: The `model_comparison.py` notebook evaluates the three models using:

  * DAIC-WOZ test split.
  * MBADD dataset.
  * Statistical significance tests for performance metrics.
=======
This is a public version of the Multi-Modal Depression Detection project, shared to provide access to the relevant parts referenced in the thesis. Its only purpose is to serve as supporting evidence of the work carried out.
>>>>>>> 37731c20c1caf40c2484f64316f5afa393887566



## Contact

For questions or inquiries, please contact:
**Bryan Lean**
\[bryanlean.jh@gmail.com]


<<<<<<< HEAD
=======
Temporary Google Drive link to download model datasets: [link](https://drive.google.com/drive/folders/1SqAoHDPgw9KiMlIZ87HmtuR4mmcfdObq?usp=sharing)
>>>>>>> 37731c20c1caf40c2484f64316f5afa393887566
