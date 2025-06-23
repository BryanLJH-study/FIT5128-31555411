import os
import argparse
import shutil
import librosa
import numpy as np
import pandas as pd
import torch
from silero_vad import get_speech_timestamps, read_audio


def non_voiced_removal(audio:np.ndarray, sr:int):
    """
    Removes non-voiced parts using Silero VAD from a librosa array.

    Args:
        - audio (np.ndarray): Audio signal loaded with librosa (1D array).
        - sr (int): Sampling rate of the audio (must be 16kHz).

    Returns:
        - np.ndarray: Cleaned audio signal (speech segments concatenated).
    """
    # Convert to PyTorch tensor
    audio_tensor = torch.tensor(audio, dtype=torch.float32)

    # Load Silero VAD model
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', source='github')
    get_speech_timestamps = utils[0]

    # Detect speech segments
    speech_timestamps = get_speech_timestamps(audio_tensor, model, sampling_rate=sr)

    # Extract speech segments
    speech_audio = torch.cat([audio_tensor[segment['start']:segment['end']] for segment in speech_timestamps])

    return speech_audio.numpy()



def extract_mfcc_from_audio_segment(audio_segment:np.ndarray, sr:int, n_mfcc, window_length_s:float, hop_length_s:float, window_type:str, n_mels:int):
    """
    Extracts MFCC features from an audio segment.

    Args:
        - audio_segment (numpy.ndarray): Audio segment data.
        - sr (int): Sampling rate of the audio.
        - n_mfcc (int): Number of MFCC coefficients to extract.
        - window_length_s (float): The length of the window in seconds
        - hop_length_s (float): The length window intervals in seconds
        - window_type (str): Type of windowing to use


    Returns:
        - numpy.ndarray: MFCC features.
    """
    # mfccs = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=n_mfcc, hop_length=160) # Previous implementation

    # Convert to windowing arguments to frames
    window_frames = int(window_length_s * sr)
    hop_frames = int(hop_length_s * sr)

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=n_mfcc, n_fft=window_frames, hop_length=hop_frames, window=window_type, n_mels=n_mels)

    print(mfccs.shape)

    return mfccs



def process_files(input_dir, output_dir, n_mfcc=60, window_length_s=0.06, hop_length_s=0.06, window_type="hamming", n_mels=60, remove_non_voiced=True):
    """
    Processes all video files in the provided DAIC-WOZ directory,
    extracting MFCCs and saving them in the output directory, maintaining the same directory structure.
    
    Args:
        - input_dir (str): Path to the DAIC-WOZ directory.
        - output_dir (str): Path to the directory to save MFCC files.
        - n_mfcc (int): Number of MFCC coefficients to extract.
        - window_length_s (float): The length of the window in seconds
        - hop_length_s (float): The length window intervals in seconds
        - window_type (str): Type of windowing to use
    """

    # For all participants' audio files in DAIC-WOZ directory
    i = 1
    for root, dirs, files in os.walk(input_dir):
        for file_name in files:
            if file_name.endswith("_AUDIO.wav"):
                # Print progress for SLURM log file
                print(f"Processing video {i}/{189}: {root}", flush=True)

                # Get participant_id, audio file path, and transcription file path
                participant_id = file_name.split("_")[0]
                audio_path = os.path.join(root, file_name)
                transcript_file = os.path.join(root, f"{participant_id}_TRANSCRIPT.csv")
                
                if os.path.exists(transcript_file):
                    # Load the transcript file with specific delimiters
                    transcript_df = pd.read_csv(
                        transcript_file,
                        sep=r"\s+",  # One or more whitespaces as the delimiter
                        engine="python",
                        skiprows=1,  # Skip the first header row
                        header=None,
                        usecols=[0, 1, 2],  # Only load the columns we care about
                        names=['start_time', 'stop_time', 'speaker'],
                    )
                    # Remove interruptions from specific participant transcripts
                    if participant_id == "373":
                        cond = (transcript_df['start_time'] >= 5*60 + 52) & (transcript_df['stop_time'] <= 7*60)
                        transcript_df = transcript_df.drop(transcript_df[cond].index)

                    elif participant_id == "444":
                        cond = (transcript_df['start_time'] >= 4*60 + 46) & (transcript_df['stop_time'] <= 6*60 + 27)
                        transcript_df = transcript_df.drop(transcript_df[cond].index)

                    # End transcripts/audio data early where AU_CLNF data ends earlier than recording (to match both)
                    elif participant_id == "402":
                        cond = (transcript_df["start_time"] > 833.267)
                        transcript_df = transcript_df.drop(transcript_df[cond].index)

                    elif participant_id == "420":
                        cond = (transcript_df["start_time"] > 706.367)
                        transcript_df = transcript_df.drop(transcript_df[cond].index)
               

                    # Load the audio file
                    y, sr = librosa.load(audio_path, sr=None)

                    spoken_segments = []

                    # Remove non-participant segments from audio
                    for index, row in transcript_df.iterrows():
                        start_time = row['start_time']
                        stop_time = row['stop_time']
                        speaker = row["speaker"]

                        if speaker == "Participant":
                            # Extract the audio segment
                            start_sample = int(start_time * sr)
                            stop_sample = int(stop_time * sr)
                            audio_segment = y[start_sample:stop_sample]
                            spoken_segments.append(audio_segment)

                    # Remove unvoiced sections from audio if specified
                    if remove_non_voiced:
                        y_cleaned = non_voiced_removal(np.concatenate(spoken_segments), sr)
                    else:
                        y_cleaned = np.concatenate(spoken_segments)

                    # Extract MFCCs
                    try:
                        mfccs = extract_mfcc_from_audio_segment(y_cleaned, sr, n_mfcc=n_mfcc, 
                                                                window_length_s=window_length_s, hop_length_s=hop_length_s, 
                                                                window_type=window_type, n_mels=n_mels)

                        # Define the output path for the MFCC file
                        output_path = os.path.join(output_dir, f"{participant_id}_P", f"{participant_id}_MFCC.npy")

                        # Ensure the output directory exists
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)

                        # Save MFCCs as a NumPy array
                        np.save(output_path, mfccs)

                    except Exception as e:
                        print(f"Error processing {audio_path}: {e}", flush=True)


                i += 1
                
    print("All participants have been processed.", flush=True)




def copy_files(input_dir, output_dir):
    """
    Copies specific files to the output directory.
    
    Args:
        - input_dir (str): Path to the directory containing the files.
        - output_dir (str): Path to the destination directory.
    """
    files_to_copy = [
        "dev_split_Depression_AVEC2017.csv",
        "full_test_split.csv",
        "train_split_Depression_AVEC2017.csv"
    ]

    for file_name in files_to_copy:
        src_path = os.path.join(input_dir, file_name)
        dest_path = os.path.join(output_dir, file_name)
        try:
            shutil.copy2(src_path, dest_path)
            print(f"Copied {file_name} to {output_dir}", flush=True)
        except Exception as e:
            print(f"Error copying {file_name}: {e}", flush=True)




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process audio files to extract MFCC features.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory containing audio files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory to save MFCC files.")
    parser.add_argument("--n_mfcc", type=int, default=60, help="Number of MFCC coefficients to extract.")
    parser.add_argument("--window_length_s", type=float, default=0.06, help="Window length in seconds.")
    parser.add_argument("--hop_length_s", type=float, default=0.06, help="Hop length in seconds.")
    parser.add_argument("--window_type", type=str, default="hamming", help="Type of windowing to use.")
    parser.add_argument("--n_mels", type=int, default=60, help="Number of Mel bands to use.")
    parser.add_argument("--remove_non_voiced", action="store_true", help="Remove non-voiced sections from audio.")

    args = parser.parse_args()

    process_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        n_mfcc=args.n_mfcc,
        window_length_s=args.window_length_s,
        hop_length_s=args.hop_length_s,
        window_type=args.window_type,
        n_mels=args.n_mels,
        remove_non_voiced=args.remove_non_voiced
    )

    copy_files(input_dir=args.input_dir, output_dir=args.output_dir)
