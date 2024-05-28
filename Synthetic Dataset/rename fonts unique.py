import os

folder_path = "fonts"  # Specify the folder path

# Get a list of all files in the folder
file_list = os.listdir(folder_path)

# Sort the file list alphabetically
sorted_files = sorted(file_list)

# Iterate over the sorted files and rename them with ordered numbers
for i, file_name in enumerate(sorted_files):
    file_extension = os.path.splitext(file_name)[1]  # Get the file extension
    new_name = f"{i+1000}{file_extension}"  # Generate the new name with ordered number
    new_path = os.path.join(folder_path, new_name)  # Get the new file path

    # Check if the new name already exists
    while os.path.exists(new_path):
        prefix = str(i + 1) + "_"  # Add a prefix to ensure uniqueness
        new_name = prefix + file_name
        new_path = os.path.join(folder_path, new_name)

    old_path = os.path.join(folder_path, file_name)  # Get the old file path
    os.rename(old_path, new_path)  # Rename the file

print("Files renamed successfully!")
