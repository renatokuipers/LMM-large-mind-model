"""
Script to generate a simple logo for the Neural Child dashboard.
"""

import os
from PIL import Image, ImageDraw, ImageFont

def create_logo(output_path, size=(200, 200), bg_color=(55, 90, 127), fg_color=(255, 255, 255)):
    """
    Create a simple logo for Neural Child.
    
    Args:
        output_path: Path where to save the logo
        size: Size of the logo image (width, height)
        bg_color: Background color in RGB
        fg_color: Foreground color in RGB
    """
    # Create a new image with a solid background
    img = Image.new('RGB', size, color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # Calculate center and dimensions
    width, height = size
    center_x, center_y = width // 2, height // 2
    
    # Draw a brain-like shape
    brain_radius = min(width, height) // 3
    
    # Draw circles representing brain lobes
    draw.ellipse((center_x - brain_radius, center_y - brain_radius, 
                  center_x + brain_radius, center_y + brain_radius), 
                 outline=fg_color, width=4)
    
    # Draw some curved lines inside to represent brain folds
    for offset in range(-brain_radius + 10, brain_radius - 10, 15):
        draw.arc((center_x - brain_radius + 10, center_y + offset,
                  center_x + brain_radius - 10, center_y + offset + 20),
                 start=0, end=180, fill=fg_color, width=2)
    
    # Draw a small heart inside the bottom part of the brain
    heart_size = brain_radius // 2
    heart_y = center_y + brain_radius // 3
    
    # Simple heart shape
    draw.ellipse((center_x - heart_size, heart_y - heart_size,
                  center_x, heart_y), 
                 fill=(231, 76, 60), outline=(231, 76, 60))
    draw.ellipse((center_x, heart_y - heart_size,
                  center_x + heart_size, heart_y), 
                 fill=(231, 76, 60), outline=(231, 76, 60))
    
    # Draw the heart point
    points = [
        (center_x - heart_size, heart_y - heart_size // 2),
        (center_x, heart_y + heart_size),
        (center_x + heart_size, heart_y - heart_size // 2)
    ]
    draw.polygon(points, fill=(231, 76, 60), outline=(231, 76, 60))
    
    # Add "NC" text below the brain
    text_y = center_y + brain_radius + 20
    draw.text((center_x, text_y), "NC", fill=fg_color, 
              anchor="mt", align="center")
    
    # Save the logo
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    
    # Also create a smaller version for the navbar
    small_size = (30, 30)
    small_img = img.resize(small_size, Image.LANCZOS)
    small_output_path = os.path.join(os.path.dirname(output_path), "logo.png")
    small_img.save(small_output_path)
    
    print(f"Logo created at {output_path}")
    print(f"Small logo created at {small_output_path}")

if __name__ == "__main__":
    output_path = os.path.join(os.path.dirname(__file__), "logo_large.png")
    create_logo(output_path) 