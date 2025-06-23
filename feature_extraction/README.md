# Feature Extraction for Multi-Modal Depression Detection

This directory contains the code for processing features from the DAIC-WOZ dataset and videos from the Monash Behavioural Analysis Project. The extracted features include **Facial Action Units (AUs)** and **Mel-Frequency Cepstral Coefficients (MFCCs)**.

---

## 1. Requirements

### 1.1 Python Dependencies

Install the required Python libraries using:

```bash
pip install -r requirements.txt
```

Key libraries:

* `librosa`
* `silero-vad`
* `pandas`
* `torch`

### 1.2 Additional Tools

**FFmpeg**: Required to extract audio from MBADD videos.
[Download FFmpeg](https://ffmpeg.org/download.html) and install it on your system.

---

## 2. Usage

### 2.1 DAIC-WOZ Feature Extraction

Run the following scripts to process the DAIC-WOZ dataset:

#### 2.1.1 Action Unit (AU) Features

```bash
python feature_extraction/DAIC-WOZ/participant_voiced_au.py --input_dir "DAIC-WOZ directory" --output_dir "Processed DAIC-WOZ output directory"
```

#### 2.1.2 MFCC Features

```bash
python feature_extraction/DAIC-WOZ/mfcc_extraction.py --input_dir "DAIC-WOZ directory" --output_dir "Processed DAIC-WOZ output directory"
```

---

### 2.2 MBADD Feature Extraction

#### 2.2.1 Download Raw Videos

Download the raw video files from the [VideoAnnotation Folder](https://drive.google.com/drive/folders/1BSbzVZ46BDftDJlMSCkltfxxnLWq-Seg?usp=drive_link) in the shared Behavioural Analysis Project drive (requires permission).

#### 2.2.2 Extract Action Units with OpenFace

Download and install [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace). Use the provided SLURM script to batch-process the videos:

```bash
feature_extraction/MBADD/openface_extraction.sh
```

Alternatively, download the pre-extracted OpenFace Action Unit features from the shared drive:
[FacialAnnotations/OpenFace Folder](https://drive.google.com/drive/folders/1mvLR5iNH6MDI441564Rf_VoieWzv3DMl?usp=drive_link).

#### 2.2.3 Action Unit (AU) Features

Process the extracted Action Units and video data:

```bash
python feature_extraction/MBADD/participant_voiced_au.py --video_input_dir "VideoAnnotations directory" --au_input_dir "OpenFace features directory" --output_dir "Processed MBADD output directory"
```

#### 2.2.4 MFCC Features

Extract MFCC features from the MBADD videos:

```bash
python feature_extraction/MBADD/mfcc_extraction.py --input_dir "VideoAnnotations directory" --output_dir "Processed MBADD output directory"
```

---

## 3. Note:

Each script supports additional arguments for configuring the feature extraction process (e.g., window sizes, skip sizes, and cepstral coefficients). Refer to the documentation in the respective script files for a detailed explanation of these options.
