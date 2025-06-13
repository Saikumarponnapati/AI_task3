

# Neural Style Transfer Project

## Objective
Implement a **Neural Style Transfer** model to apply artistic styles to photographs using deep learning techniques. This notebook demonstrates how to stylize an image using a pre-trained VGG19 model.

## Features
- Load and preprocess content and style images.
- Use VGG19 for feature extraction.
- Compute content and style losses.
- Perform gradient descent to iteratively generate the stylized image.

## Subtasks
1. **Setup Environment**:
   - Install required libraries (`tensorflow`, `keras`, `Pillow`).
   - Import necessary modules and functions.

2. **Load and Preprocess Images**:
   - Content and style images are resized and converted into arrays compatible with the VGG19 model.

3. **Model Definition**:
   - Load the VGG19 model with pretrained ImageNet weights.
   - Extract intermediate layers for content and style features.

4. **Loss Computation**:
   - Content loss: Mean Squared Error between content and generated image features.
   - Style loss: Gram matrix comparison to capture style features.
   - Total variation loss to ensure spatial continuity.

5. **Optimization**:
   - Use gradient descent (Adam optimizer) to update the image iteratively.
   - Generate and display the final styled output.

## How to Run
1. Clone the repository or download the notebook.
2. Install dependencies:
   ```bash
   pip install tensorflow keras Pillow
````

3. Replace the paths of the content and style images with your own in the notebook.
4. Run the notebook to visualize the output.

## Sample Code Snippet

```python
def load_and_preprocess_image(image_path, img_size):
    img = load_img(image_path, target_size=img_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img
```

## Dependencies

* Python 3.7+
* TensorFlow
* Keras
* Pillow
* NumPy
* Matplotlib

## Output

* Styled image combining content of one image with the style of another.
* Visualization of content, style, and output images for comparison.

## Author

* *Purna Sai Kumar Reddy Ponnapati*




