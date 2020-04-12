import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

class YOLO():

    def __init__(self):

        """
        - YOLO takes an image as input. We should set the dimension of the image to a fixed number.
        - The default choice is often 416x416.
        - YOLO applies thresholding and non maxima suppression, define a value for both
        - Load the classes, model configuration (cfg file) and pretrained weights (weights file) into variables
        - If the image is 416x416, the weights must be corresponding to that image
        - Load the network with OpenCV.dnn function
        """

        self.conf_threshold = 0.5
        self.nms_threshold = 0.4
        self.inp_width = 320
        self.inp_height = 320

        with open("yolov3/coco.names", "rt") as f:

            self.classes = f.read().rstrip('\n').split('\n')

        model_configuration = "Yolov3/yolov3.cfg";
        model_weights = "Yolov3/yolov3.weights";
        self.net = cv2.dnn.readNetFromDarknet(model_configuration, model_weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

    def get_outputs_names(self):

        """
        Get the names of the output layers.
        """

        # Get the names of all the layers in the network.
        layers_names = self.net.getLayerNames()

        # Get the names of the output layers, i.e. the layers with unconnected outputs.
        return [layers_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def draw_pred(self, frame, class_id, conf, left, top, right, bottom):

        """
        Draw a bounding box around a detected object given the box coordinates.
        """

        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), thickness=5)
        label = '%.2f' % conf

        # Get the label for the class name and its confidence.
        if self.classes:

            assert(class_id < len(self.classes))
            label = f"{self.classes[class_id]}:{label}"

        # Display the label at the top of the bounding box.
        label_size, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, label_size[1])
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), thickness=3)
        return frame

    def post_process(self,frame, outs):

        """
        Take the output out of the neural network and interpret it; use the output to
        apply NMS thresholding and confidence thresholding. Also use the output to
        draw the bounding boxes using the draw_pred method.
        """

        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        class_ids = []
        confidences = []
        boxes = []

        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class
        # with the highest score.
        for out in outs:

            for detection in out:

                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.conf_threshold:

                    center_x = int(detection[0] * frame_width)
                    center_y = int(detection[1] * frame_height)
                    width = int(detection[2] * frame_width)
                    height = int(detection[3] * frame_height)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            self.conf_threshold,
            self.nms_threshold
        )

        for i in indices:

            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            output_image = self.draw_pred(
                frame, class_ids[i],
                confidences[i],
                left,
                top,
                left + width,
                top + height
            )

        return frame, boxes

    def inference(self,image):

        """
        Main loop taking an image as input and generated frames with drawn
        bounding boxes as output.
        """

        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(image, 1/255, (self.inp_width, self.inp_height), [0,0,0], 1, crop=False)

        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.get_outputs_names())

        # Remove the bounding boxes with low confidence
        final_frame, boxes = self.post_process(image, outs)

        return final_frame, boxes
