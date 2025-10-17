import pickle
import numpy as np
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# --- Load the Scikit-learn Model ---
try:
    with open('model.p', 'rb') as f:
        model_dict = pickle.load(f)
    model = model_dict['model']
except FileNotFoundError:
    print("Error: model.p not found.")
    exit()

# --- Define Input Structure ---
# The model input is a vector of 84 features (42 features per hand * 2 hands)
initial_type = [('float_input', FloatTensorType([None, 84]))] # None for batch size

# --- Convert to ONNX format ---
# Note: The target_opset depends on your specific library versions; 13 is usually safe.
onx = convert_sklearn(model, initial_types=initial_type, target_opset=13)

# Save the ONNX model
with open("model.onnx", "wb") as f:
    f.write(onx.SerializeToString())

print("Scikit-learn model successfully converted and saved as model.onnx")