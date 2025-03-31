import numpy as np
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D
import argparse

def read_depth_image(file_path):
    """
    Read a depth image file. The appropriate reading method depends on the file format.
    This is a simplified version that attempts to handle common formats.
    """
    try:
        # Try reading as a standard image
        depth_img = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)
        
        # If it's a 16-bit image (common for depth), convert from mm to meters
        if depth_img.dtype == np.uint16:
            depth_img = depth_img.astype(np.float32) / 1000.0
        
        # display depth image statistics
        print(f"Depth image shape: {depth_img.shape}")
        print(f"Depth image dtype: {depth_img.dtype}")
        print(f"Depth image min: {np.min(depth_img)}")
        print(f"Depth image max: {np.max(depth_img)}")
        print(f"Depth image mean: {np.mean(depth_img)}")
        return depth_img
    except Exception as e:
        print(f"Error reading depth file: {e}")
        return None

def visualize_depth_2d(depth_img, colormap=cv2.COLORMAP_JET):
    """
    Create a 2D visualization of depth data using a colormap
    """
    # Normalize the depth values to 0-255 range for visualization
    if depth_img.dtype != np.uint8:
        depth_min = np.min(depth_img[depth_img > 0])  # Ignore zeros which often indicate invalid measurements
        depth_max = np.max(depth_img)
        depth_normalized = np.zeros_like(depth_img, dtype=np.uint8)
        
        # Convert valid depth values to 0-255 range
        mask = depth_img > 0
        depth_normalized[mask] = 255 * (depth_img[mask] - depth_min) / (depth_max - depth_min)
    else:
        depth_normalized = depth_img
        
    # Apply colormap
    depth_colormap = cv2.applyColorMap(depth_normalized, colormap)
    
    return depth_colormap, depth_normalized

def visualize_depth_3d(depth_img, step=10):
    """
    Create a 3D visualization of depth data as a point cloud or mesh
    Step parameter controls density of visualization (higher = faster but less dense)
    """
    print(f"Depth image shape: {depth_img.shape}")
    h, w = depth_img.shape
    
    # Create X, Y coordinate grids
    y, x = np.mgrid[0:h:step, 0:w:step]
    
    # Get Z values (depth)
    z = depth_img[::step, ::step]
    print(f"X shape: {x.shape}, Y shape: {y.shape}, Z shape: {z.shape}")
    
    # Create figure for 3D visualization
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection ='3d')

    # ax = fig.add_subplot(111, projection='3d')
    # Plot the valid points
    surf = ax.plot_surface(X=x, Y=y, Z=z, cmap='viridis', 
                          linewidth=0, antialiased=False)
    
    # Set labels
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_zlabel('Depth (meters)')
    ax.set_title('3D Depth Visualization')
    
    # Add color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    return fig

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize depth information in an image')
    parser.add_argument('depth_file', help='Path to the depth image file')
    parser.add_argument('--mode', choices=['2d', '3d', 'both'], default='both',
                        help='Visualization mode: 2D, 3D, or both')
    parser.add_argument('--colormap', type=int, default=cv2.COLORMAP_JET,
                        help='Colormap for 2D visualization (OpenCV colormap integer)')
    parser.add_argument('--step', type=int, default=10,
                        help='Step size for 3D visualization (higher = faster but less detailed)')
    parser.add_argument('--output', help='Output file path for saving results')
    
    args = parser.parse_args()
    
    # Read the depth image
    depth_img = read_depth_image(args.depth_file)
    
    if depth_img is None:
        print("Failed to read depth image.")
        return
    
    # 2D visualization
    if args.mode in ['2d', 'both']:
        depth_colormap, depth_normalized = visualize_depth_2d(depth_img, args.colormap)
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(depth_normalized, cmap='gray')
        plt.title('Depth (Grayscale)')
        plt.colorbar(label='Normalized Depth')
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB))
        plt.title('Depth (Colormap)')
        plt.colorbar(label='Normalized Depth')
        
        plt.tight_layout()
        
        if args.output and args.mode == '2d':
            plt.savefig(args.output)
        elif args.mode == '2d':
            plt.show()
    
    # 3D visualization
    if args.mode in ['3d', 'both']:
        fig = visualize_depth_3d(depth_img, args.step)
        
        if args.output and args.mode == '3d':
            fig.savefig(args.output)
        elif args.mode == '3d' or args.mode == 'both':
            plt.show()

if __name__ == "__main__":
    main()