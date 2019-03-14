mo.py --data_type FP16 --framework tf --input_model TF_model/tf_model.pb --model_name IR_model --output_dir IR_model/ --input_shape [1,28,28,1] --input conv2d_1_input --output activation_6/Softmax

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