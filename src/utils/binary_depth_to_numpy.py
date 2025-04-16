import os
import numpy as np
from omegaconf import OmegaConf

def extract_binary_files(suffix,directory, output_directory=None, shape=None):
    """
    Sequentially extract binary files from a specified directory.
    
    Args:
        directory (str): Path to the directory containing binary files
        output_directory (str, optional): Path to save extracted data. 
                                          If None, uses a subdirectory in the input directory
    
    Returns:
        list: List of extracted file paths
    """
    # # Ensure the directory exists
    if not os.path.isdir(directory):
        raise ValueError(f"Directory {directory} does not exist.")
    
    # Create output directory if not specified
    if output_directory is None:
        output_directory = os.path.join(directory, 'extracted_files')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Get all .bin files and sort them 
    # (assuming you want to process files in a consistent order)
    bin_files = sorted([f for f in os.listdir(directory) if f.endswith(f'.{suffix}')])
    
    extracted_files = []
    
    # Process each binary file
    for file_name in bin_files:
        file_path = os.path.join(directory, file_name)
        
        try:
            # Read the binary file
            with open(file_path, 'rb') as f:
                # Read the entire file content
                data = np.fromfile(f, dtype=np.uint16)
                
                # Example: Reshape data if it's a depth image (adjust as needed)
                # Assumes a specific dimension - you might need to modify this
                # based on your specific binary file structure
                try:
                    data = data[:shape[0]*shape[1]]
                    reshaped_data = data.reshape(shape[0], shape[1])
                except ValueError:
                    # If reshaping fails, save as a 1D array
                    reshaped_data = data
                    print(f"Warning: Could not reshape {file_name}. Data saved as 1D array.")
                
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
    #load config file using OmegaConf
    parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    conf = OmegaConf.load(f"{parent_directory}/utils/conf/bin_depth.yaml")
    input_directory = conf.input_directory
    output_directory = conf.output_directory
    shape = conf.shape
    try:
        # Extract files
        extracted_files = extract_binary_files("raw",input_directory, output_directory=output_directory,shape=shape)
        
        # Print summary
        print(f"\nExtraction complete. Total files processed: {len(extracted_files)}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()