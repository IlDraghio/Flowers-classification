import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
from tensorflow.keras.models import load_model                            #type: ignore
from tensorflow.keras.applications.imagenet_utils import preprocess_input #type: ignore
from PIL import Image
import base64
from io import BytesIO

EXAMPLES_DIR = "cnn_model/sample_images"
model = load_model("cnn_model/model.keras")

def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((224,224))
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

def predicted_class(request):
    uploaded_file = request.FILES["image"]
    img = Image.open(uploaded_file)
    
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    img_array = preprocess_image(img)
    preds = model.predict(img_array)
    predicted_class = np.argmax(preds, axis=1)[0]
    prediction = f"Predicted class: {predicted_class}"
    
    example_path = os.path.join(EXAMPLES_DIR, f"class_{predicted_class}.png")
    if os.path.exists(example_path):
        example_img = Image.open(example_path)
        buffer = BytesIO()
        example_img.save(buffer, format="PNG")
        buffer.seek(0)
        example_image_data = base64.b64encode(buffer.read()).decode("utf-8")
        
    return prediction,image_data,example_image_data