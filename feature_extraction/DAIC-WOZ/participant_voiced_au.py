import os
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
        - np.ndarray: voiced timestamps.
    """
    # Convert to PyTorch tensor
    audio_tensor = torch.tensor(audio, dtype=torch.float32)

    # Load Silero VAD model
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', source='github')
    get_speech_timestamps = utils[0]

    # Detect speech segments
    speech_timestamps = get_speech_timestamps(audio_tensor, model, sampling_rate=sr)

    return speech_timestamps




def process_au_files(input_dir, output_dir):
    """
    Processes all files in the provided DAIC-WOZ directory,
    extracting MFCCs and saving them in the output directory, maintaining the same directory structure.
    
    Args:
        - input_dir (str): Path to the DAIC-WOZ directory.
        - output_dir (str): Path to the directory to save MFCC files.
    """

    # For all participants' audio files in DAIC-WOZ directory
    i = 1
    for root, dirs, files in os.walk(input_dir):
        for file_name in files:
            if file_name.endswith("_AUDIO.wav"):
                # Print progress for SLURM log file
                print(f"Processing video {i}/{189}: {root}", flush=True)
                
                # Get participant_id and audio, transcription, au file paths
                participant_id = file_name.split("_")[0]
                audio_path = os.path.join(root, file_name)
                transcript_file = os.path.join(root, f"{participant_id}_TRANSCRIPT.csv")
                au_file_path = os.path.join(root, f"{participant_id}_CLNF_AUs.txt")
                
                
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

                    if participant_id == "444":
                        cond = (transcript_df['start_time'] >= 4*60 + 46) & (transcript_df['stop_time'] <= 6*60 + 27)
                        transcript_df = transcript_df.drop(transcript_df[cond].index)

                    # End transcripts/audio data early where AU_CLNF data ends earlier than recording (to match both)
                    elif participant_id == "402":
                        cond = (transcript_df["start_time"] > 833.267)
                        transcript_df = transcript_df.drop(transcript_df[cond].index)

                    elif participant_id == "420":
                        cond = (transcript_df["start_time"] > 706.367)
                        transcript_df = transcript_df.drop(transcript_df[cond].index)


                    # Get AU data, retaining rows where timestamp falls within speaking intervals
                    au_df = pd.read_csv(au_file_path)
                    participant_speaking = transcript_df[transcript_df['speaker'] == "Participant"]
                    mask = au_df[' timestamp'].apply(
                        lambda t: any(
                            (participant_speaking['start_time'] <= t) &
                            (t <= participant_speaking['stop_time'])
                        )
                    )
                    au_df = au_df[mask]
                    au_df.reset_index(drop=True, inplace=True)


                    # Get participant voiced segments from audio
                    y, sr = librosa.load(audio_path, sr=None)

                    spoken_segments = []

                    for index, row in transcript_df.iterrows():
                        start_time = row['start_time']
                        stop_time = row['stop_time']
                        speaker = row["speaker"]

                        if speaker == "Participant":
                            start_sample = int(start_time * sr)
                            stop_sample = int(stop_time * sr)
                            audio_segment = y[start_sample:stop_sample]
                            spoken_segments.append(audio_segment)

                    speech_timestamps = non_voiced_removal(np.concatenate(spoken_segments), sr)


                    # Filter voiced sections from au_df (based on frames/index)
                    filtered_df = pd.DataFrame()
                    for dict in speech_timestamps:
                        start_idx = int((dict['start'] / sr) * 30)
                        end_idx = int((dict['end'] / sr) * 30)

                        filtered_df = pd.concat([filtered_df, au_df.loc[start_idx:end_idx-1]])

                    filtered_df.reset_index(drop=True, inplace=True)

                     # Define the output path for the MFCC file
                    output_path = os.path.join(output_dir, f"{participant_id}_P", f"{participant_id}_CLNF_AUs.txt")

                    # Ensure the output directory exists
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)

                    # Save MFCCs as a NumPy array
                    filtered_df.to_csv(output_path, index=False)


                i += 1
                
    print("All participants have been processed.", flush=True)


if __name__ == "__main__":
    source_dir = "data/DAIC-WOZ/"
    output_dir = "Temp/au_mfcc/DAIC-WOZ_Participant_Voiced/"
    process_au_files(source_dir, output_dir)
