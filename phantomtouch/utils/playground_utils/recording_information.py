from PIL import Image

def get_image_resolution(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
        return width, height

if __name__ == "__main__":
    image_path = '/home/abdullah/utn/phantom-human-videos/extracted_frames/frame_0000.jpg'  # Replace with your image path
    width, height = get_image_resolution(image_path)
    print(f"Image resolution: {width}x{height}")