import torch
from feat import Detector
from feat.utils.io import video_to_tensor
import os
import datetime
import psutil
import av
from numpy import swapaxes
import pandas as pd
import shutil 


# Force CUDNN benchmarking off (incompatible anyway)
torch.backends.cudnn.benchmark = False


def log_message(message):
    """
    Log messages with timestamps for SLURM output.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def log_memory_usage():
    """
    Log memory usage for debugging.
    """
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    log_message(f"Current memory usage: {mem:.2f} MB")


# Modified video_to_tensor
def video_to_tensor(file_name, start_frame, num_frames):
    container = av.open(file_name)
    stream = container.streams.video[0]
    tensor = []
    frame_count = 0
    for frame in container.decode(stream):
        if frame_count >= start_frame and frame_count < start_frame + num_frames:
            frame_data = torch.from_numpy(frame.to_ndarray(format="rgb24"))
            frame_data = swapaxes(swapaxes(frame_data, 0, -1), 1, 2)
            tensor.append(frame_data)
        frame_count += 1
        if frame_count >= start_frame + num_frames:
            break
    container.close()
    return torch.stack(tensor, dim=0)



def merge_csv_files(chunk_files, output_file):
    """
    Merge multiple CSV files into a single file.
    """
    try:
        merged_data = pd.concat([pd.read_csv(chunk_file) for chunk_file in chunk_files], ignore_index=True)
        merged_data.to_csv(output_file, index=False)

        # Optionally delete chunk files after merging
        chunk_dir = os.path.dirname(chunk_files[0])  # Get the chunk directory
        for chunk_file in chunk_files:
            os.remove(chunk_file)
        
        # Remove the chunk directory
        shutil.rmtree(chunk_dir)
        log_message(f"Successfully merged and removed chunk directory: {chunk_dir}")

    except Exception as e:
        log_message(f"Error during merging chunk files: {e}")


def process_video(video_path, source_dir, output_dir, chunk_size=1792):
    """
    Process a video in chunks and merge annotations into a single CSV file.
    """
    out_name = video_path.replace(source_dir, output_dir).replace('.mp4', '.csv')
    chunk_dir = os.path.dirname(out_name) + "/chunks"
    os.makedirs(chunk_dir, exist_ok=True)  # Directory for chunk CSVs
    os.makedirs(os.path.dirname(out_name), exist_ok=True)  # Ensure output directory exists

    if not os.path.exists(out_name):  # Skip if already processed
        try:
            # Initialize the detector
            detector = Detector(device='cuda')
            log_message(f"Processing: {os.path.basename(video_path)}")
            log_memory_usage()

            # Get total frames in the video
            container = av.open(video_path)
            stream = container.streams.video[0]
            total_frames = stream.frames
            container.close()

            chunk_files = []  # List to store paths of chunk CSVs

            # Process video in chunks
            for start_frame in range(0, total_frames, chunk_size):
                video_tensor = video_to_tensor(video_path, start_frame, chunk_size)
                num_frames = video_tensor.shape[0]
                log_message(f"Processing frames {start_frame} to {start_frame + num_frames} of {total_frames}")
                log_memory_usage()

                if num_frames == 0:
                    log_message(f"Warning: {video_path} has no frames in this chunk!")
                    continue

                # Save each chunk to its own CSV
                chunk_file = os.path.join(chunk_dir, f"chunk_{start_frame}.csv")
                detector.detect(video_tensor, data_type="tensor", num_workers=4, batch_size=64, save=chunk_file, progress_bar=False)
                chunk_files.append(chunk_file)
                log_message(f"Finished processing chunk {start_frame} to {start_frame + num_frames} of {total_frames}")
                log_memory_usage()

                # Clear CUDA cache to release memory
                del video_tensor
                torch.cuda.empty_cache()
                log_message(f"Cleared memory for chunk {start_frame} to {start_frame + num_frames}")

            # Merge all chunk CSV files
            merge_csv_files(chunk_files, out_name)
            log_message(f"Merged all chunks into {out_name}")

        except RuntimeError as e:
            log_message(f"Runtime error for {video_path}: {e}")

        except Exception as e:
            log_message(f"Error processing {video_path}: {e}")


def annotate_sequential(source_dir, output_dir):
    """Annotate videos sequentially."""
    video_paths = []
    for root, dirs, files in os.walk(source_dir, topdown=False):
        for file_name in files:
            if file_name.endswith('.mp4'):
                video_path = os.path.join(root, file_name)
                video_paths.append(video_path)

    log_message(f"Found {len(video_paths)} videos to process sequentially.")

    i = 1
    for video_path in video_paths:
        log_message(f"Video {i}/{len(video_paths)}")
        process_video(video_path, source_dir, output_dir)
        i += 1


if __name__ == "__main__":
    source_dir = "/home/blea0003/VideoAnnotation"
    output_dir = "/home/blea0003/FacialAnnotations"
    annotate_sequential(source_dir, output_dir)
