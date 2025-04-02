import os
import json
import numpy as np
import cv2
import tifffile
from tensorflow.keras.models import Model
from Architectures.arch_mJNet import mJNet_3dot5D

# --------------------- Constants --------------------- #
T, H, W = 30, 512, 512
PIXELS = [0, 85, 170]
LABELS = ['background', 'penumbra', 'core']
VERBOSE = True

# --------------------- Paths --------------------- #
CONFIG_PATH = "/home/stud/sazidur/bhome/4D-mJ-Net/SAVE/EXP036.3/setting.json"
WEIGHTS_PATH = "/home/stud/sazidur/bhome/4D-mJ-Net/SAVE/EXP036.3/TMP_MODELS/mJNet_3dot5D_DA_ADAM_VAL20_SOFTMAX_128_512x512__69.h5"
INPUT_FOLDER = "/home/stud/sazidur/bhome/sus-nifti/0001/baseline_tiff/7/"
OUTPUT_FOLDER = "/home/stud/sazidur/bhome/preprocess_isles_amador_512_5mm/output/0001/baseline/7/"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --------------------- Utilities --------------------- #
def log(msg): 
    if VERBOSE: print(msg)

def get_pixel_values(): return PIXELS
def get_labels(): return LABELS
def is_TO_CATEG(): return True

def load_tiff_stack(folder, T):
    tiffs = sorted([f for f in os.listdir(folder) if f.endswith(".tiff")])
    assert len(tiffs) == T, f"Expected {T} timepoints but found {len(tiffs)}"
    stack = [tifffile.imread(os.path.join(folder, f)) for f in tiffs]
    return np.stack(stack, axis=0)[..., np.newaxis]  # (T, H, W, 1)

def save_results(img_pred, categ_img, check_img_processed, output_base, idx="12"):
    os.makedirs(output_base, exist_ok=True)
    gt_folder = os.path.join(output_base, "GT/")
    tmp_folder = os.path.join(output_base, "TMP/")
    heatmap_folder = os.path.join(output_base, "HEATMAP/")
    os.makedirs(gt_folder, exist_ok=True)
    os.makedirs(tmp_folder, exist_ok=True)
    os.makedirs(heatmap_folder, exist_ok=True)

    cv2.imwrite(os.path.join(output_base, f"{idx}.png"), img_pred)
    cv2.imwrite(os.path.join(gt_folder, f"{idx}.png"), check_img_processed)

    check_rgb = cv2.cvtColor(check_img_processed, cv2.COLOR_GRAY2RGB)

    if categ_img is not None:
        if categ_img.shape[-1] >= 3:
            penumbra_prob = cv2.convertScaleAbs(categ_img[:, :, 1] * 255)
            penumbra_colored = cv2.applyColorMap(penumbra_prob, cv2.COLORMAP_JET)
            blend_penumbra = cv2.addWeighted(check_rgb, 0.5, penumbra_colored, 0.5, 0.0)
            cv2.imwrite(os.path.join(heatmap_folder, f"{idx}_heatmap_penumbra.png"), blend_penumbra)

        core_prob = cv2.convertScaleAbs(categ_img[:, :, 2] * 255)
        core_colored = cv2.applyColorMap(core_prob, cv2.COLORMAP_JET)
        blend_core = cv2.addWeighted(check_rgb, 0.5, core_colored, 0.5, 0.0)
        cv2.imwrite(os.path.join(heatmap_folder, f"{idx}_heatmap_core.png"), blend_core)

    img_pred_rgb = cv2.cvtColor(np.uint8(img_pred), cv2.COLOR_GRAY2RGB)

    if check_img_processed is not None:
        _, penumbra_mask = cv2.threshold(check_img_processed, 85, 255, cv2.THRESH_BINARY)
        _, core_mask = cv2.threshold(check_img_processed, 170, 255, cv2.THRESH_BINARY)

        penumbra_cnt, _ = cv2.findContours(penumbra_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        core_cnt, _ = cv2.findContours(core_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img_pred_rgb = cv2.drawContours(img_pred_rgb, penumbra_cnt, -1, (255, 0, 0), 2)
        img_pred_rgb = cv2.drawContours(img_pred_rgb, core_cnt, -1, (0, 0, 255), 2)

        cv2.imwrite(os.path.join(tmp_folder, f"{idx}.png"), img_pred_rgb)

# --------------------- Load Model Config --------------------- #
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)
params = config["models"][0]["params"]
params["concatenate_input"] = params.get("concatenate_input", False)
multiInput = params.get("multiInput", {})
n_slices = params["n_slices"]

# --------------------- Load Input --------------------- #
log("📥 Loading input stack...")
img_stack = load_tiff_stack(INPUT_FOLDER, T)  # (T, H, W, 1)
input_list = [np.expand_dims(img_stack, axis=0) for _ in range(n_slices)]  # [(1, T, H, W, 1)]

# --------------------- Build and Load Model --------------------- #
log("🧠 Building model...")
model = mJNet_3dot5D(params, multiInput, usePMs=False)
model.build(input_shape=[(None, T, H, W, 1) for _ in range(n_slices)])
model.load_weights(WEIGHTS_PATH)
log("✅ Weights loaded.")

# --------------------- Predict --------------------- #
log("🔮 Predicting...")
output = model.predict(input_list, batch_size=1)[0]  # shape: (512, 512, C)

# --------------------- Process Output --------------------- #
if is_TO_CATEG():
    pred = np.argmax(output, axis=-1)  # shape: (512, 512)
    pred_img = np.zeros_like(pred, dtype=np.uint8)
    for i, val in enumerate(PIXELS):
        pred_img[pred == i] = val
else:
    pred_img = (output > 0.5).astype(np.uint8) * PIXELS[-1]

# Load GT image if available
gt_path = os.path.join(INPUT_FOLDER, "gt.png")
check_img_processed = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(gt_path) else np.zeros((H, W), dtype=np.uint8)

# --------------------- Save Outputs --------------------- #
tifffile.imwrite(os.path.join(OUTPUT_FOLDER, "prediction.tiff"), pred_img)
log("✅ Saved prediction.tiff")

if is_TO_CATEG():
    for i in range(output.shape[-1]):
        prob_map = (output[..., i] * 255).astype(np.uint8)
        tifffile.imwrite(os.path.join(OUTPUT_FOLDER, f"prob_class_{i}.tiff"), prob_map)
    log("✅ Saved probability maps.")

save_results(pred_img, categ_img=output, check_img_processed=check_img_processed, output_base=OUTPUT_FOLDER, idx="12")
log("✅ Saved contour overlay.")
log(f"🎯 Done! Unique values in prediction: {np.unique(pred_img)}")
