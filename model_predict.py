"""
A simple example of how to use the Clarifai SDK to predict the
architectural style of a building from an image.

This example uses the Architecture IDentifier model, which is a
Clarifai model that detects the architectural style of a building
from an image. The model returns a list of 3 possible
architectural styles: Modern, Romanesque, or Victorian, with a confidence score for each style.


The example takes an image URL as an argument, and uses the
Clarifai SDK to predict the architectural style of the building
in the image. The prediction is then printed to the
console.

The example also prints out the confidence score for each
architectural style. The confidence score is a value between
0 and 1 that represents how confident the model is that the
image is of a particular architectural style.

You can run this example by installing the Clarifai SDK and
running the following command:

python model_predict.py <image_url>

Replace `<image_url>` with the URL of the image you want to
predict the architectural style of.

This example assumes that you have the Clarifai SDK installed and
configured on your system. If you don't have the SDK installed,
you can install it with pip:

pip install clarifai

You can find more information about the Clarifai SDK at
https://clarifai.com/developer/docs/python

"""
from secret import CLARIFAI_PAT
from clarifai.client.model import Model
import clarifai

model_url = (
    "https://clarifai.com/aiman-ameer-malik/architecture-identifier/models/architecture-identifier"
)
filepath = "test set/victorian.jpg"
# The Predict API also accepts data through URL, Filepath & Bytes.
# Example for predict by filepath:
model_prediction = Model(model_url, pat=CLARIFAI_PAT).predict_by_filepath(filepath, input_type="image")

# Example for predict by bytes:
# model_prediction = Model(model_url, pat=CLARIFAI_PAT).predict_by_bytes(image_bytes, input_type="text")

# model_prediction = Model(url=model_url, pat=CLARIFAI_PAT).predict_by_url(
#     image_url, input_type="image"
# )

# Get the output
outputs = model_prediction.outputs[0].data.concepts
print(model_prediction.outputs[0].data)
predictions = {}
for output in outputs:
    print(output.name, ":", output.value)
    predictions[output.name] = output.value

# Extract maximum value from predicitons
max_value = max(predictions.values())

# Get the key with maximum value
max_key = max(predictions, key=predictions.get)
print("\nPredicted label : ", max_key)
print("Predicted value : ", max_value)


