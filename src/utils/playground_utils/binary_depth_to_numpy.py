import os
import numpy as np

def extract_binary_files(directory, output_directory=None):
    """
    Sequentially extract binary files from a specified directory.
    
    Args:
        directory (str): Path to the directory containing binary files
        output_directory (str, optional): Path to save extracted data. 
                                          If None, uses a subdirectory in the input directory
    
    Returns:
        list: List of extracted file paths
    """
    # Ensure the directory exists
    if not os.path.isdir(directory):
        raise ValueError(f"Directory {directory} does not exist.")
    
    # Create output directory if not specified
    if output_directory is None:
        output_directory = os.path.join(directory, 'extracted_files')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Get all .bin files and sort them 
    # (assuming you want to process files in a consistent order)
    bin_files = sorted([f for f in os.listdir(directory) if f.endswith('.bin')])
    
    extracted_files = []
    
    # Process each binary file
    for file_name in bin_files:
        file_path = os.path.join(directory, file_name)
        
        try:
            # Read the binary file
            with open(file_path, 'rb') as f:
                # Read the entire file content
                data = np.fromfile(f, dtype=np.float32)
                
                # Example: Reshape data if it's a depth image (adjust as needed)
                # Assumes a specific dimension - you might need to modify this
                # based on your specific binary file structure
                try:
                    # Attempt to reshape to a square image (adjust dimensions as needed)
                    # This is a guess - you'll need to modify based on your actual data
                    side_length = int(np.sqrt(len(data)))
                    reshaped_data = data.reshape((side_length, side_length))
                except ValueError:
                    # If reshaping fails, save as a 1D array
                    reshaped_data = data
                
                # Create output file path
                output_file_path = os.path.join(output_directory, f'extracted_{file_name}.npy')
                
                # Save the extracted data
                np.save(output_file_path, reshaped_data)
                
                extracted_files.append(output_file_path)
                
                print(f"Processed: {file_name} -> {output_file_path}")
        
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
    
    return extracted_files

def main():
    # Example usage
    input_directory = '/home/abdullah/utn/phantom-human2robot/playground_sieve_sam_hamer/data/recordings/white_cloth_exp/white_nonreflective_cloth_light_on_ambient_light/depth_only_output'
    output_directory = "/home/abdullah/utn/phantom-human2robot/playground_sieve_sam_hamer/data/recordings/white_cloth_exp/white_nonreflective_cloth_light_on_ambient_light/depth_raw_npy"
    try:
        # Extract files
        extracted_files = extract_binary_files(input_directory, output_directory=output_directory)
        
        # Print summary
        print(f"\nExtraction complete. Total files processed: {len(extracted_files)}")
        print("Extracted files:", extracted_files)
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()