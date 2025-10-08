import os
import shutil

# Define the source and destination folders
source_folder = '/data/projects/dna/DNABERT/'
destination_folder = '/data/projects/dna/DNABERT/TATA_POS_JIM/'

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Loop through all files in the source folder
for filename in os.listdir(source_folder):
    # Check if the file name starts with "tata"
    if filename.startswith("tata_jim"):
        # Define the full file paths
        source_file = os.path.join(source_folder, filename)
        destination_file = os.path.join(destination_folder, filename)
        
        # Move the file to the destination folder
        shutil.move(source_file, destination_file)

print("Files moved successfully.")
