import ollama
import os

# Define the image path
image_path = 'images/handwriting.jpg'

# Check if the file exists
if not os.path.exists(image_path):
    print(f"Error: Image file '{image_path}' not found!")
else:
    response = ollama.chat(
        model='llama3.2-vision',
        messages=[{
            'role': 'user',
            'content': 'can you extract the text from this image?',
            'images': [image_path]
        }]
    )
    
    # Extract and clean up the content from the response
    content = response.message.content
    
    # Print the formatted receipt
    print("\n=== The drawing data===\n")
    for line in content.split('\n'):
        if line.strip():  # Only print non-empty lines
            print(line)