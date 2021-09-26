"""
Created on Sun Sep 26 15:52:06 2021
@author: kevin machado gamboa
"""


def predict_tfl(model_interpreter, in_details, out_details, chunk):
    """
    makes predictions using a TfLite model
    @param model_interpreter: model in tflite format
    @param in_details: tflite model input details
    @param out_details: tflite model output details
    @param chunk: vector for prediction
    @return:
    """
    # Point the data to be used for testing and run the interpreter
    model_interpreter.set_tensor(in_details[0]['index'], chunk)
    model_interpreter.invoke()
    # Obtain results and map them to the classes
    predictions = model_interpreter.get_tensor(out_details[0]['index'])[0]

    return predictions
