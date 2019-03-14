# Keras to Tensorflow to IR model (NCS2)


## Goal
Convert a Keras or a Tensorflow model to IR files ready to be used with the Neural Compute Stick 2

## Prerequisites
For that you want to have OpenVino installed and python 3.5 at least. For the python requirements, see the "Requirements" section.

## File structure

## How to use

### Keras to Tensorflow conversion
If you have a Keras .h5 file, use `keras_to_tf.py` to create a Tensorflow .pb file.

        python keras_to_tf.py

that will take the Keras file situated in *Keras_model/model.h5* and create a .pb file in *TF_model/tf_model.pb*.

### Tensorflow to IR conversion
If you didn't had a .pb model before now you should have one. We'll use the model optimizer to convert the file.

        mo.py --data_type FP16 --framework tf --input_model TF_model/tf_model.pb --model_name IR_model --output_dir IR_model/ --input_shape [1,28,28,1] --input conv2d_1_input --output activation_6/Softmax

### Runing the inference on the NCS2
Now you can run the inference on the NCS2. For that use the predict_mnist.py

        python predict_mnist.py

This file load the IR model, read and convert the *data/6.jpg* and feed it for classification.

If everything goes fine, you should see something like this:

        [ INFO ] Loading network files:
        IR_model/IR_model.xml
        IR_model/IR_model.bin
        [ INFO ] Preparing input blobs
        1 1 28 28
        (28, 28)
        (1, 28, 28)
        [ INFO ] Loading model to the plugin
        [ INFO ] Starting inference (1 iterations)
        [ INFO ] Average running time of one iteration: 1.8284320831298828 ms
        [ INFO ] Processing output blob
        [[0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]]

The last line is the class vector. We have a 1 at index 6, so the image has been correctly classified.

## Requirements

        pip install -r requirements.txt



