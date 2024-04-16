import os
import pickle
import numpy as np

def unpack_pkl_files(folder_path):
    # Get a list of all .pkl files in the folder
    pkl_files = [file for file in os.listdir(folder_path) if file.endswith('.pkl')]

    # Iterate over each .pkl file
    total_files = len(pkl_files)
    for i, pkl_file in enumerate(pkl_files):
        pkl_file_path = os.path.join(folder_path, pkl_file)

        # Create a folder with the same name as the .pkl file
        folder_name = os.path.splitext(pkl_file)[0]
        output_folder = os.path.join(folder_path, folder_name)
        os.makedirs(output_folder, exist_ok=True)

        # Load the pickle file
        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)

        # Write out each key entry as a text file
        for key, value in data.items():
            output_file_path = os.path.join(output_folder, f'{key}.txt' if not key.endswith('.txt') else key)
            # Save the numpy array as a text file
            np.savetxt(output_file_path, value)

        # Update the status bar
        # progress = (i + 1) / total_files * 100
        print(f'Progress: {0} of {1}'.format(i + 1, total_files), end='\r')

if __name__ == '__main__':
    # Prompt the user to provide the path to the .pkl file
    folder_path = input("Enter the path to the folder containing .pkl files: ")
    unpack_pkl_files(folder_path)