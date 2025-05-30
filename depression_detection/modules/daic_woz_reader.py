import os
import pandas as pd


class DAIC_WOZ_READER:
    def __init__(self, directory: str, speaking_only: bool, keep_AU_cols: list, phq8_categories: dict = None, remove_unsuccessful: bool = True):
        """
        Args:
            - directory (str): Path to the DAIC-WOZ directory.
            - speaking_only (bool): Whether to include participant-speaking frames only.
            - keep_AU_cols (list): List of AU column names to retain.
            - phq8_categories (dict, optional): A dictionary where keys are category labels (or numbers)
                                                and values are tuples (lower_bound, upper_bound) for PHQ8 score ranges.
                                                Defaults to {0: (0, 4), 1: (5, 9), 2: (10, 14), 3: (15, 24)}.
            - remove_unsuccessul (bool): Whether to remove rows where AUs were not detected successfully
        """
        self.directory = directory

        # Read and combine AU files into one dataframe
        self.au_df = self.concat_au_files(directory, speaking_only=speaking_only)
        self.clean_df(keep_AU_cols=keep_AU_cols, remove_unsuccessful=remove_unsuccessful)

        # Use provided categorization if given; else default 
        if phq8_categories is None:
            phq8_categories = {0: (0, 4), 1: (5, 9), 2: (10, 14), 3: (15, 24)}

        # Read data splits
        self.split_dfs = {
            'train': self.read_split_df("train_split_Depression_AVEC2017.csv", phq8_categories=phq8_categories),
            'val': self.read_split_df("dev_split_Depression_AVEC2017.csv", phq8_categories=phq8_categories),
            'test': self.read_split_df("full_test_split.csv", phq8_categories=phq8_categories)
        }


    def concat_au_files(self, directory: str, speaking_only: bool):
        """
        Combines XXX_CLNF_AUs.txt files into a single DataFrame.
        When speaking_only is True, includes only rows where the participant is speaking
        based on the corresponding XXX_TRANSCRIPT.csv files.

        Args:
            - directory (str): Path to the DAIC-WOZ directory.
            - speaking_only (bool): Whether to include participant-speaking frames only.

        Returns:
            - pd.DataFrame: Combined DataFrame of all AU files with cleaned column names.
        """
        dfs = []

        # For all participants' AU files in DAIC-WOZ directory
        for root, dirs, files in os.walk(directory):
            for file_name in files:
                if file_name.endswith("_CLNF_AUs.txt"):
                    # Get participant ID
                    participant_id = file_name.split("_")[0]
                    print(f"Reading Participant ID: {participant_id}...", end='\r')

                    # Read AU csv & add Participant_ID column
                    au_file_path = os.path.join(root, file_name)
                    df = pd.read_csv(au_file_path)
                    df['Participant_ID'] = participant_id

                    # Remove leading/trailing spaces from column names
                    df.columns = df.columns.str.strip()

                    # Read interview trainscript
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

                        if speaking_only:
                            # Filter to include only Participant speaking rows
                            participant_speaking = transcript_df[transcript_df['speaker'] == "Participant"]

                            # Retain rows in AU data where timestamp falls within speaking intervals
                            mask = df['timestamp'].apply(
                                lambda t: any(
                                    (participant_speaking['start_time'] <= t) &
                                    (t <= participant_speaking['stop_time'])
                                )
                            )
                            df = df[mask]

                        elif participant_id in ["451", "458", "480"]:
                            # For these participants, use data from after the first transcript row to the last
                            actual_start_time = transcript_df.iloc[1]["start_time"]
                            actual_stop_time = transcript_df.iloc[-1]["stop_time"]
                            df = df[(df["timestamp"] >= actual_start_time) & (df["timestamp"] <= actual_stop_time)]


                        else:
                            # Use data from the start of "Ellie" speaking to the final spoken line
                            actual_start_time = transcript_df[transcript_df['speaker'] == "Ellie"].iloc[0]["start_time"]
                            actual_stop_time = transcript_df.iloc[-1]["stop_time"]
                            df = df[(df["timestamp"] >= actual_start_time) & (df["timestamp"] <= actual_stop_time)]

                    dfs.append(df)

        # Combine into a single DataFrame and ensure column names are stripped (in case any were missed)
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df.columns = combined_df.columns.str.strip()

        return combined_df
    

    def clean_df(self, keep_AU_cols: list, remove_unsuccessful: bool):
        """
        Cleans the AU dataframe by removing unsuccessful feature extraction rows and specific interruptions, 
        then keeps only the specified AU columns in addition to the essential columns.

        Args:
           -  keep_AU_cols (list): List of AU column names to retain.
        """
        # Remove non-successful feature extraction rows
        if remove_unsuccessful:
            unsuccessful_pct = (1 - self.au_df['success'].sum() / self.au_df.shape[0]) * 100
            print(f"Unsuccessful frames in data to be removed: {unsuccessful_pct}%")
            self.au_df = self.au_df[self.au_df["success"] == 1]

        # Remove interruptions for Participant 373 – interruption around 5:52-7:00 (in seconds)
        cond = (self.au_df["Participant_ID"] == "373") & (self.au_df["timestamp"] >= 5 * 60 + 52) & (self.au_df["timestamp"] <= 7 * 60)
        self.au_df = self.au_df.drop(self.au_df[cond].index)

        # Remove interruptions for Participant 444 – interruption around 4:46-6:27 (in seconds)
        cond = (self.au_df["Participant_ID"] == "444") & (self.au_df["timestamp"] >= 4 * 60 + 46) & (self.au_df["timestamp"] <= 6 * 60 + 27)
        self.au_df = self.au_df.drop(self.au_df[cond].index)

        # Filter df to only keep essential columns and selected AU columns.
        essential_cols = ["Participant_ID", "timestamp", "success"]
        columns_to_keep = essential_cols + keep_AU_cols
        self.au_df = self.au_df[columns_to_keep]


    def read_split_df(self, split_df_name: str, phq8_categories: dict):
        """
        Reads and processes a split dataframe. Dynamically categorizes participants based on their PHQ8 score.

        Args:
            - split_df_name (str): Filename of the split CSV.
            - phq8_categories (dict, optional): A dictionary where keys are category labels and values
                                                are tuples (lower_bound, upper_bound) defining the PHQ8 score ranges.

        Returns:
            - pd.DataFrame: Processed dataframe with a new 'Category' column.
        """
        def categorize_phq8(score):
            for category, (lower, upper) in phq8_categories.items():
                if lower <= score <= upper:
                    return int(category)
            return None  

        # Read the split dataframe
        split_df = pd.read_csv(os.path.join(self.directory, split_df_name))

        # Standardize PHQ8_Score column name (the test split uses "PHQ_Score")
        if "test" in split_df_name and "PHQ_Score" in split_df.columns:
            split_df = split_df.rename(columns={"PHQ_Score": "PHQ8_Score"})

        # Filter columns, ensure Participant_ID is a string, and add Category based on PHQ8_Score
        split_df = split_df[['Participant_ID', 'Gender', 'PHQ8_Score']]
        split_df['Participant_ID'] = split_df['Participant_ID'].astype(str)
        split_df['Category'] = split_df['PHQ8_Score'].apply(categorize_phq8)

        return split_df


    def au_df_stats(self):
        """
        Prints basic statistics of the AU dataframe.
        """
        print(f"\nUnique videos: {self.au_df.Participant_ID.nunique()}")
        print(f"Total processed frames: {self.au_df.shape[0]}")
        print(f"Avg frames per video: {self.au_df.groupby('Participant_ID').size().mean()}")
        print(f"Memory used: {self.au_df.memory_usage(deep=True).sum() / (1024 ** 3)} GB")