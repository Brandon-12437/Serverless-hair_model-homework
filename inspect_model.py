import onnx

# Load the ONNX model
model = onnx.load("hair_classifier_v1.onnx")

# Print input names
print("INPUT NODES:")
for inp in model.graph.input:
    print("-", inp.name)

# Print output names
print("\nOUTPUT NODES:")
for out in model.graph.output:
    print("-", out.name)
