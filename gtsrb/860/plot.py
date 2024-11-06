import numpy as np
import ast
import glob
import os

# Read the original image from 'image.txt'
with open('image.txt', 'r') as file:
    data_image = ast.literal_eval(file.read())

data_image = np.array(data_image)
if data_image.max() <= 1.0:
    data_image = (data_image * 255).astype(np.uint8)
else:
    data_image = data_image.astype(np.uint8)

data_image = data_image.reshape(32, 32, 3)

# Get a list of all counterfactual files
counterfactual_files = glob.glob('counterfactual-at-pixel-*-predicted-as-6.txt')
counterfactual_files.sort()

# Read the first counterfactual image
with open(counterfactual_files[0], 'r') as file:
    data_counterfactual_first = ast.literal_eval(file.read())
data_counterfactual_first = np.array(data_counterfactual_first)
if data_counterfactual_first.max() <= 1.0:
    data_counterfactual_first = (data_counterfactual_first * 255).astype(np.uint8)
else:
    data_counterfactual_first = data_counterfactual_first.astype(np.uint8)
data_counterfactual_first = data_counterfactual_first.reshape(32, 32, 3)

# Now, compare each subsequent counterfactual to the first one
for idx, counterfactual_file in enumerate(counterfactual_files[1:], start=1):
    with open(counterfactual_file, 'r') as file:
        data_counterfactual_current = ast.literal_eval(file.read())
    data_counterfactual_current = np.array(data_counterfactual_current)
    if data_counterfactual_current.max() <= 1.0:
        data_counterfactual_current = (data_counterfactual_current * 255).astype(np.uint8)
    else:
        data_counterfactual_current = data_counterfactual_current.astype(np.uint8)
    data_counterfactual_current = data_counterfactual_current.reshape(32, 32, 3)
    
    # get size of counterfactuals for comparison, i.e. number of
    # changed channels compared to the original image
    size_counterfactual = 0
    for i in range(32):
        for j in range(32):
            if np.any(data_counterfactual_current[i, j] != data_image):
                size_counterfactual += 1
    # num pixels including all channels
    size_counterfactual2 = (data_counterfactual_current != data_image).sum()

    # Compute differences
    # a) Number of different pixels (any channel differs)
    num_different_pixels = 0
    for i in range(32):
        for j in range(32):
            if np.any(data_counterfactual_current[i, j] != data_counterfactual_first[i, j]):
                num_different_pixels += 1
    
    # b) Number of different pixel-channels
    channel_differences = (data_counterfactual_current != data_counterfactual_first)
    num_different_channels = np.sum(channel_differences)

    print(num_different_channels,size_counterfactual2)
    print(num_different_channels/size_counterfactual2)
    # print(num_different_channels, size_counterfactual)
    
    # Print the results
    pixel_index_current = os.path.basename(counterfactual_file).split('-')[3]
    pixel_index_first = os.path.basename(counterfactual_files[0]).split('-')[3]
    print(f"Comparison between counterfactual at pixel {pixel_index_first} and pixel {pixel_index_current}:")
    print(f"  a) Number of different pixels (any channel differs): {num_different_pixels} out of {32*32}")
    print(f"  b) Number of different pixel-channels: {num_different_channels} out of {32*32*3}")
    print()
exit()

# Start building the HTML content
html_content = '''
<html>
<head>
    <title>Counterfactual Images</title>
    <style>
        body { font-family: Arial, sans-serif; }
        .container { height: 800px; overflow-y: scroll; }
        .image-set { margin-bottom: 20px; }
        .image-set h2 { margin-top: 0; }
    </style>
</head>
<body>
    <h1>Counterfactual Images</h1>
    <div class="container">
'''

# Loop through each counterfactual file
for idx, counterfactual_file in enumerate(counterfactual_files):
    with open(counterfactual_file, 'r') as file:
        data_counterfactual = ast.literal_eval(file.read())

    data_counterfactual = np.array(data_counterfactual)
    if data_counterfactual.max() <= 1.0:
        data_counterfactual = (data_counterfactual * 255).astype(np.uint8)
    else:
        data_counterfactual = data_counterfactual.astype(np.uint8)

    data_counterfactual = data_counterfactual.reshape(32, 32, 3)

    difference = (data_image != data_counterfactual).any(axis=2)
    highlighted_image = data_counterfactual.copy()
    highlight_color = [255, 0, 0]
    highlighted_image[difference] = highlight_color

    pixel_index = os.path.basename(counterfactual_file).split('-')[3]

    # Create a figure and save it to a PNG in memory
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(data_image)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(data_counterfactual)
    axs[1].set_title(f'Counterfactual at Pixel {pixel_index}')
    axs[1].axis('off')

    axs[2].imshow(highlighted_image)
    axs[2].set_title('Differences Highlighted')
    axs[2].axis('off')

    plt.tight_layout()

    # Save the figure to a PNG image in memory
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    # Add the image to the HTML content
    html_content += f'''
    <div class="image-set">
        <h2>Counterfactual at Pixel {pixel_index}</h2>
        <img src="data:image/png;base64,{image_base64}" alt="Counterfactual Image">
    </div>
    '''

# Close the HTML content
html_content += '''
    </div>
</body>
</html>
'''

# Write the HTML content to a file
with open('counterfactual_images.html', 'w') as f:
    f.write(html_content)

print("HTML file 'counterfactual_images.html' has been created.")
