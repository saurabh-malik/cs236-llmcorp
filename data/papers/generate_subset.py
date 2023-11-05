import os
import random
import shutil

TARGET_COUNT = 20
# Set the source and destination directories
SRC_DIR = "1000-papers"
DEST_DIR = f"{TARGET_COUNT}-papers"

# Ensure the destination directory exists
if not os.path.exists(DEST_DIR):
    os.makedirs(DEST_DIR)

# Get a list of all PDF files in the source directory
pdf_files = [f for f in os.listdir(SRC_DIR) if f.endswith('.pdf')]

# Randomly select 200 PDF files
selected_files = random.sample(pdf_files, TARGET_COUNT)

# Copy the selected files to the destination directory
for file in selected_files:
    shutil.copy2(os.path.join(SRC_DIR, file), os.path.join(DEST_DIR, file))

print(f"Copied {TARGET_COUNT} random PDF files from {SRC_DIR} to {DEST_DIR}.")

