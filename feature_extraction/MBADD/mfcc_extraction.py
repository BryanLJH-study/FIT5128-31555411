import os 
import re
import argparse
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
    Processes all video files in the provided VideoAnnotation directory.
    For each participant, for each depression level, combines spoken parts and extracts the MFCCs, then saves them in output directory.
    
    Args:
        - input_dir (str): Path to the DAIC-WOZ directory.
        - output_dir (str): Path to the directory to save MFCC files.
        - n_mfcc (int): Number of MFCC coefficients to extract.
        - window_length_s (float): The length of the window in seconds
        - hop_length_s (float): The length window intervals in seconds
        - window_type (str): Type of windowing to use
    """
    # Sample rate 16khz to match DAIC-WOZ
    sr = 16000 

    i = 1
    for root, _, _ in os.walk(input_dir):
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
                for video_part_path in video_part_paths:
                    # Get relative path to preserve the directory structure
                    relative_path = os.path.relpath(video_part_path, input_dir)
                    participant_id, category, _, video = relative_path.split(os.sep)
                    os.makedirs(os.path.join(output_dir, participant_id, category), exist_ok=True)
                    
                    # Define the path for the temporary audio file
                    temp_audio_path = os.path.join(output_dir, participant_id, category, video+'.wav')

                    # Extract audio file from the video
                    extract_audio_from_video(video_part_path, temp_audio_path, sr)

                    # Load the audio file
                    y, sr = librosa.load(temp_audio_path, sr=sr)
                    audio_segments.append(y)
   
                    # Remove audio file
                    try:
                        os.remove(temp_audio_path)
                    except Exception as e:
                        print(f"Error deleting temporary audio file {temp_audio_path}: {e}", flush=True)

                # Remove unvoiced sections from audio if specified
                if remove_non_voiced:
                    y_cleaned = non_voiced_removal(np.concatenate(audio_segments), sr)
                else:
                    y_cleaned = np.concatenate(audio_segments)

                # Extract MFCCs
                try:
                    mfccs = extract_mfcc_from_audio_segment(y_cleaned, sr, n_mfcc=n_mfcc, 
                                                            window_length_s=window_length_s, hop_length_s=hop_length_s, 
                                                            window_type=window_type, n_mels=n_mels)

                    # Define the output path for the MFCC file
                    output_path = os.path.join(output_dir, participant_id, category, f"{participant_id}_{category[0:3]}.npy")

                    # Save MFCCs as a NumPy array
                    np.save(output_path, mfccs)

                except Exception as e:
                    print(f"Error processing MFCC: {e}", flush=True)
                    

            i += 1

    print("All videos have been processed.", flush=True)
 



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process video files to extract MFCCs.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory containing video files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory to save MFCC files.")
    parser.add_argument("--n_mfcc", type=int, default=60, help="Number of MFCC coefficients to extract.")
    parser.add_argument("--window_length_s", type=float, default=0.06, help="Window length in seconds.")
    parser.add_argument("--hop_length_s", type=float, default=0.06, help="Hop length in seconds.")
    parser.add_argument("--window_type", type=str, default="hamming", help="Type of windowing to use.")
    parser.add_argument("--n_mels", type=int, default=60, help="Number of mel bands.")
    parser.add_argument("--remove_non_voiced", type=bool, default=True, help="Whether to remove non-voiced sections.")

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
