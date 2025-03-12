from e2b_desktop import Sandbox
import time

# With custom configuration
desktop = Sandbox(
    display=":0",  # Custom display (defaults to :0)
    resolution=(1920, 1080),  # Custom resolution
    dpi=96,  # Custom DPI
)

# Start the stream
desktop.stream.start()

# Get stream URL
url = desktop.stream.get_url()
print(url)

# Open file with default application
desktop.files.write("/home/user/index.js", "console.log('hello')") # First create the file
desktop.open("/home/user/index.js") # Then open it

# Take a screenshot and save it as "screenshot.png" locally
image = desktop.screenshot()
# Save the image to a file
with open("screenshot.png", "wb") as f:
    f.write(image)

# Run any bash command
out = desktop.commands.run("ls -la /home/user")
print(out)

desktop.wait(10000) # Wait for 10 seconds

# Stop the stream
desktop.stream.stop()