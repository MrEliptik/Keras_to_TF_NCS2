import cv2
import sys
import os
import numpy as np
import logging as log
from time import time
from openvino.inference_engine import IENetwork, IEPlugin
from ImageProcessor import ImageProcessor

processor = ImageProcessor()
test_image = 'data/photo_6.jpg'
input_image = cv2.imread(test_image)
cropped_input, cropped = processor.preprocess_image(input_image)

perf_counts = False

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    model_xml = "IR_model/ir_model.xml"
    model_bin = "IR_model/ir_model.bin"

    # Plugin initialization for specified device and load extensions library if specified
    plugin = IEPlugin(device="MYRIAD")

    # Read IR
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)

    assert len(net.inputs.keys())   == 1, "Sample supports only single input topologies"
    assert len(net.outputs)         == 1, "Sample supports only single output topologies"

    log.info("Preparing input blobs")
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    net.batch_size = len(test_image)

    # Read and pre-process input images
    n, c, h, w = net.inputs[input_blob].shape
    images = np.ndarray(shape=(n, c, h, w))

    image = cv2.imread(test_image)
    if image.shape[:-1] != (h, w):
        log.warning("Image {} is resized from {} to {}".format(test_image, image.shape[:-1], (h, w)))
        image = cv2.resize(image, (w, h))
    image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW

    log.info("Batch size is {}".format(n))

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = plugin.load(network=net)
    del net

    # Start sync inference
    log.info("Starting inference ({} iterations)".format(1))
    infer_time = []
    for i in range(1):
        t0 = time()
        res = exec_net.infer(inputs={input_blob: images})
        infer_time.append((time()-t0)*1000)
    log.info("Average running time of one iteration: {} ms".format(np.average(np.asarray(infer_time))))
    if perf_counts:
        perf_counts = exec_net.requests[0].get_perf_counts()
        log.info("Performance counters:")
        print("{:<70} {:<15} {:<15} {:<15} {:<10}".format('name', 'layer_type', 'exet_type', 'status', 'real_time, us'))
        for layer, stats in perf_counts.items():
            print("{:<70} {:<15} {:<15} {:<15} {:<10}".format(layer, stats['layer_type'], stats['exec_type'],
                                                              stats['status'], stats['real_time']))

    # Processing output blob
    log.info("Processing output blob")
    res = res[out_blob]
    log.info("Top {} results: ".format(10))

    for i, probs in enumerate(res):
        probs = np.squeeze(probs)
        top_ind = np.argsort(probs)[-10:][::-1]
        print("Image {}\n".format(test_image))
        for id in top_ind:
            det_label = "#{}".format(id)
            print("{:.7f} label {}".format(probs[id], det_label))
        print("\n")

    del exec_net
    del plugin

if __name__ == "__main__":
    main()
