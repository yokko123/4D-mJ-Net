import os
import numpy as np
import cv2
import tifffile
import tensorflow as tf
import SimpleITK as sitk
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dropout
from tcn.tcn.tcn import TCN  # Ensure TCN is correctly imported

# ---------------------------- GPU CONFIGURATION ---------------------------- #
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[3], 'GPU')  # Set GPU 3 for prediction
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPU(s), {len(logical_gpus)} Logical GPU(s) set.")
    except RuntimeError as e:
        print(e)

# ---------------------------- PARAMETERS ---------------------------- #
model_path = "/home/prosjekt/IschemicStroke/Data/CTP/Published segmentation results/LT_4DmJ-Net/EXP221/TMP_MODELS_SE_v21-0.5/TCNet_3dot5D_single_encoder_DA_ADAM_VAL20_SOFTMAX_128_512x512__06.h5"
test_data_folder = "/home/stud/sazidur/bhome/preprocess_isles_amador_512_5mm"
output_folder = "/home/stud/sazidur/bhome/preprocess_isles_amador_512_5mm/output"

# ---------------------------- LOAD MODEL ---------------------------- #
# Define MonteCarloDropout layer (if needed)
class MonteCarloDropout(Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)

# ✅ Load the model with custom layers
try:
    with tf.keras.utils.custom_object_scope({'MonteCarloDropout': MonteCarloDropout, 'TCN': TCN}):
        model = load_model(model_path, compile=False)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    exit()

# -------------------- IMAGE PREPROCESSING HELPERS -------------------- #
def adjust_gamma(image, gamma=2.2):
    """Apply gamma correction."""
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def apply_histogram_equalization(image):
    """Apply histogram equalization."""
    if len(image.shape) == 2:  # Grayscale
        return cv2.equalizeHist(image)
    elif len(image.shape) == 3 and image.shape[-1] == 3:  # RGB
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    return image

# -------------------------- PREDICTION FUNCTION -------------------------- #
def predict_img(patient_id, slice_id):
    """
    Load a test image, preprocess it, predict, and save results.
    """
    slice_folder = os.path.join(test_data_folder, patient_id, slice_id)
    if not os.path.isdir(slice_folder):
        raise FileNotFoundError(f"❌ Slice folder not found: {slice_folder}")

    tiff_files = sorted([os.path.join(slice_folder, f) for f in os.listdir(slice_folder) if f.endswith('.tiff')])
    print(f"📂 Found {len(tiff_files)} TIFF files in {slice_folder}")

    if len(tiff_files) == 0:
        raise ValueError("❌ No TIFF images found!")

    images = []
    for file in tiff_files:
        img = tifffile.imread(file)
        if img is None or img.size == 0:
            raise ValueError(f"❌ Corrupted TIFF file: {file}")
        images.append(img)

    # Convert to NumPy array
    ctp_array = np.stack(images, axis=-1)  # Shape: (H, W, T)
    print(f"📊 Input shape (H, W, T): {ctp_array.shape}")

    # Ensure input is valid
    if ctp_array.size == 0:
        raise ValueError("❌ ERROR: Empty input tensor detected!")

    # Ensure correct input shape for model
    expected_shape = model.input_shape[1:]  # Model expected shape excluding batch dim
    if ctp_array.shape[0:2] != expected_shape[1:3]:
        raise ValueError(f"❌ ERROR: Model expects shape {expected_shape}, but got {ctp_array.shape}")

    # Reshape for model: (batch, time, height, width, channels)
    x_input = np.transpose(ctp_array, (2, 0, 1))  # (T, H, W)
    x_input = np.expand_dims(x_input, axis=-1)  # (T, H, W, 1)
    x_input = np.expand_dims(x_input, axis=0)  # (1, T, H, W, 1)

    # Debugging input shape
    print(f"🔹 Model expects input shape: {model.input_shape}")
    print(f"🔹 Actual input shape: {x_input.shape}")

    if x_input.shape[1] == 0:
        raise ValueError("❌ ERROR: Incorrect reshaping resulted in an empty input tensor.")

    # Run prediction
    predictions = model.predict(x_input)
    print(f"✅ Prediction completed. Shape: {predictions.shape}")

    pred_img = predictions[0]  # Remove batch dimension

    # Save prediction
    save_img(patient_id, slice_id, pred_img, output_folder)

# ------------------------ IMAGE SAVING FUNCTION ------------------------ #
def save_img(patient_id, slice_id, pred_img, output_folder, gamma=2.2, equalize=True):
    """
    Save predicted image, heatmaps, and lesion contours.
    """
    slice_output_folder = os.path.join(output_folder, patient_id, slice_id)
    os.makedirs(slice_output_folder, exist_ok=True)

    pred_img = np.uint8(pred_img * 255)

    if gamma:
        pred_img = adjust_gamma(pred_img, gamma)
    if equalize:
        pred_img = apply_histogram_equalization(pred_img)

    pred_img_path = os.path.join(slice_output_folder, "predicted_image.tiff")
    tifffile.imwrite(pred_img_path, pred_img)
    print(f"✅ Saved prediction: {pred_img_path}")

    save_contours(patient_id, slice_id, pred_img, slice_output_folder)

# ---------------------- SAVE CONTOURS FUNCTION ---------------------- #
def save_contours(patient_id, slice_id, pred_img, save_folder):
    """
    Draw and save contour images for penumbra & core.
    """
    if len(pred_img.shape) == 3 and pred_img.shape[-1] > 1:
        pred_img = np.argmax(pred_img, axis=-1).astype(np.uint8)

    pred_img = (pred_img * 255).astype(np.uint8) if pred_img.max() <= 1 else pred_img.astype(np.uint8)

    pred_img_rgb = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2RGB)

    cv2.drawContours(pred_img_rgb, [], -1, (255, 0, 0), 2)
    cv2.drawContours(pred_img_rgb, [], -1, (0, 0, 255), 2)

    contour_path = os.path.join(save_folder, "contours.tiff")
    tifffile.imwrite(contour_path, pred_img_rgb)
    print(f"✅ Saved contour image: {contour_path}")

# ------------------------------ MAIN SCRIPT ------------------------------ #
if __name__ == "__main__":
    predict_img(patient_id="0001", slice_id="8")
