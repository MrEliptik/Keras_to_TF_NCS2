import cv2
import sys
import os
import numpy as np
import logging as log
from time import time
from openvino.inference_engine import IENetwork, IEPlugin

im_path = 'data/6.jpg'

perf_counts = False

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    model_xml = "IR_model/IR_model.xml"
    model_bin = "IR_model/IR_model.bin"

    # Plugin initialization for specified device and load extensions library if specified
    plugin = IEPlugin(device="MYRIAD")

    # Read IR
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)

    log.info("Preparing input blobs")
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))

    # Prepare image
    n, c, h, w = net.inputs[input_blob].shape
    print(n, c, h, w)
    
    prepimg = np.ndarray(shape=(n, c, h, w))

    im = cv2.imread(im_path, 0)    # read image as grayscale
    resized_image = cv2.resize(im, (28, 28), interpolation = cv2.INTER_CUBIC)
    print(resized_image.shape)
    im_w_channel = np.ndarray(shape=(1, 28, 28))
    # Change data layout from HW to CHW
    im_w_channel[0,:,:] = resized_image
    print(im_w_channel.shape)
    
    # For 3 channel images
    #image = resized_image.transpose((2, 0, 1)) # Change data layout from HWC to CHW
    prepimg[0,:,:,:] = im_w_channel

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = plugin.load(network=net)
    del net

    # Start sync inference
    log.info("Starting inference ({} iterations)".format(1))
    infer_time = []
    t0 = time()
    res = exec_net.infer(inputs={input_blob: prepimg})
    infer_time.append((time()-t0)*1000)
    log.info("Average running time of one iteration: {} ms".format(np.average(np.asarray(infer_time))))
    # Processing output blob
    log.info("Processing output blob")
    print(res)
    res = res[out_blob]
    
    '''
    log.info("Top {} results: ".format(1))

    for i, probs in enumerate(res):
        probs = np.squeeze(probs)
        top_ind = np.argsort(probs)[-1:][::-1]
        print("Image {}\n".format(test_image))
        for id in top_ind:
            det_label = "#{}".format(id)
            print("{:.7f} label {}".format(probs[id], det_label))
        print("\n")
    '''

    del exec_net
    del plugin

if __name__ == "__main__":
    main()
