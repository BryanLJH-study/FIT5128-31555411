import os 
import re
import argparse
import pandas as pd
import subprocess # For ffmpeg command-line calls
import librosa
import numpy as np
import torch
from silero_vad import get_speech_timestamps, read_audio



def extract_audio_from_video(video_path, audio_path, sr):
    """
    Extracts the audio from a video file using ffmpeg and saves it as a temporary .wav file.
    Args:
        - video_path (str): Path to the input video file (mp4).
        - audio_path (str): Path to save the extracted audio (wav).
    """
    try:
        command = [
            'ffmpeg', '-i', video_path, 
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # Uncompressed WAV format
            '-ar', str(sr),  
            audio_path
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    except Exception as e:
        print(f"Error extracting audio from {video_path}: {e}", flush=True)



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



def process_au_files(video_input_dir, au_input_dir, output_dir):
    """
    Processes all files in the provided Monash Behavioural Analaysis Dataset videos directory,
    extracting MFCCs and saving them in the output directory, maintaining the same directory structure.
    
    Args:
        - video_input_dir (str): Path to Monash Behavioural Analaysis Dataset videos directory
        - input_dir (str): Path to the DAIC-WOZ directory.
        - output_dir (str): Path to the directory to save MFCC files.
    """
    # Sample rate 16khz to match DAIC-WOZ
    sr = 16000 

    i = 1
    for root, _, _ in os.walk(video_input_dir):
        if root.endswith(("200(Mild)", "300(Moderate)", "400(Severe)")):
            # Print progress for SLURM log file
            print(f"Processing videos {i}/{7 * 3}: {root}", flush=True)
            video_part_paths = [os.path.join(root, file)
                                for root, _, files in os.walk(root)
                                for file in files if re.search(r"-([0-9]*[02468])\.mp4$", file)]
            
            print(f"Found {len(video_part_paths)} video parts")
            if len(video_part_paths) > 0:

                # Load participant spoken audio segments
                audio_segments = []
                au_segments = []
                for video_part_path in video_part_paths:
                    # Get relative path to preserve the directory structure
                    relative_path = os.path.relpath(video_part_path, video_input_dir)
                    participant_id, category, part, video = relative_path.split(os.sep)
                    os.makedirs(os.path.join(output_dir, participant_id, category), exist_ok=True)

                    # Load & append teh AU file
                    df = pd.read_csv(os.path.join(au_input_dir, participant_id, category, part, video.split('.')[0]+".csv"))
                    au_segments.append(df)

                    # Define the path for the temporary audio file
                    temp_audio_path = os.path.join(output_dir, participant_id, category, video+'.wav')

                    # Extract audio file from the video
                    extract_audio_from_video(video_part_path, temp_audio_path, sr)

                    # Load & append the audio data
                    y, sr = librosa.load(temp_audio_path, sr=sr)
                    audio_segments.append(y)
   
                    # Remove audio file
                    try:
                        os.remove(temp_audio_path)
                    except Exception as e:
                        print(f"Error deleting temporary audio file {temp_audio_path}: {e}", flush=True)

                au_df = pd.concat(au_segments, ignore_index=True)
                au_df = au_df.drop(columns=["face_id"])

                speech_timestamps = non_voiced_removal(np.concatenate(audio_segments), sr)

                # Filter voiced sections from au_df (based on frames/index)
                filtered_df = pd.DataFrame()
                for dict in speech_timestamps:
                    start_idx = int((dict['start'] / sr) * 120)
                    end_idx = int((dict['end'] / sr) * 120)

                    filtered_df = pd.concat([filtered_df, au_df.loc[start_idx:end_idx-1]])

                filtered_df.reset_index(drop=True, inplace=True)

                output_path = os.path.join(output_dir, participant_id, category, f"{participant_id}_{category[0:3]}.csv")

                filtered_df.to_csv(output_path, index=False)     

            i += 1
            
    print("All participants have been processed.", flush=True)




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Process video and AU files to extract participant-voiced AU segments.")
    parser.add_argument("--video_input_dir", type=str, required=True, help="Path to the videos directory.")
    parser.add_argument("--au_input_dir", type=str, required=True, help="Path to the AU annotations directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory to save processed files.")

    args = parser.parse_args()

    process_au_files(
        video_input_dir=args.video_input_dir,
        au_input_dir=args.au_input_dir,
        output_dir=args.output_dir
    )