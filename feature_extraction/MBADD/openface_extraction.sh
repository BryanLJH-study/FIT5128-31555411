#!/bin/bash
#SBATCH --job-name=facial_feature_extraction # Job name
#SBATCH --output=annotation_%j.out           # Output file (with job ID)
#SBATCH --error=annotation_%j.err            # Error file (with job ID)
#SBATCH --ntasks=1                           # Number of tasks
#SBATCH --cpus-per-task=4                    # Number of CPU cores
#SBATCH --mem=16G                            # Memory per node
#SBATCH --time=168:00:00                     # Max runtime (7 days)
#SBATCH --partition=cpu1,cpu2                # Partition (queue)

module load anaconda
source activate openface

# Define input and output directories
INPUT_DIR="/home/blea0003/VideoAnnotation/Type 01"
OUTPUT_DIR="/home/blea0003/OpenFaceAnnotation/Type 01"

# Start logging
echo "Job started at: $(date)" > job_log.txt
total_files=0
skipped_files=0

# Process all MP4 files
find "$INPUT_DIR" -type f -name "*.mp4" | while read -r filepath; do
    # Check if the file path contains '100(Anxiety)'
    if [[ "$filepath" == *"100(Anxiety)"* ]]; then
        echo "Skipping file in Category '100(Anxiety)': $filepath" | tee -a job_log.txt
        ((skipped_files++))
        continue
    fi

    # Increment file count
    ((total_files++))

    # Log current file
    echo "Processing file: $filepath" | tee -a job_log.txt
    start_time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "Start time: $start_time" | tee -a job_log.txt

    # Get the relative path and directory structure
    relative_path=${filepath#$INPUT_DIR/}
    subdir=$(dirname "$relative_path")

    # Ensure the output directory structure exists
    mkdir -p "$OUTPUT_DIR/$subdir"

    # Extract the video filename without the extension
    filename=$(basename -- "$filepath")
    basename="${filename%.*}"

    # Specify the output CSV file
    output_csv="$OUTPUT_DIR/$subdir/${basename}.csv"

    # Run FeatureExtraction with -of flag
    FeatureExtraction -f "$filepath" -of "$output_csv" -aus

    # Remove the corresponding (filename)_of_details.txt file
    details_file="$OUTPUT_DIR/$subdir/${basename}_of_details.txt"
    rm -f "$details_file"

    # Log completion time for the current file
    end_time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "End time: $end_time" | tee -a job_log.txt
    echo "Completed file: $filepath" | tee -a job_log.txt
    echo "------------------------------------" | tee -a job_log.txt
done

# Log total files processed and skipped
echo "Total files processed: $total_files" | tee -a job_log.txt
echo "Total files skipped: $skipped_files" | tee -a job_log.txt
echo "Job completed at: $(date)" | tee -a job_log.txt