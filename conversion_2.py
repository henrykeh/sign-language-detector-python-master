import tensorflow as tf
from onnx_tf.backend import prepare

# --- Load the ONNX model ---
onnx_model = onnx.load("model.onnx")

# --- Convert ONNX model to TensorFlow Keras model ---
tf_rep = prepare(onnx_model)

# Export the TensorFlow Keras model (SavedModel format)
tf_rep.export_graph("tf_keras_model")

print("ONNX model successfully converted and saved as tf_keras_model/")