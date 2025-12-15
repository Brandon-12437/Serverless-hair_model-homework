from io import BytesIO
from urllib import request
from PIL import Image

# Function to download an image from URL
def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

# Function to convert to RGB and resize
def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

if __name__ == "__main__":
    # Step 3a: Image URL
    url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
    
    # Step 3b: Download image
    img = download_image(url)
    
    # Step 3c: Resize to target size (200x200 for this model)
    target_size = (200, 200)
    img = prepare_image(img, target_size)
    
    # Step 3d: Verify
    print("Image size:", img.size)
    print("Image mode:", img.mode)
    
    # Step 3e: Save resized image (optional)
    img.save("resized_image.jpg")
