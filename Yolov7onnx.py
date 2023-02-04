# Inference for ONNX model
import cv2
import numpy as np
import onnxruntime as ort
import random

class YOLOv7:
    """
        Yolov7 onnx model detector
    """
    def __init__(self, weight,class_names = ["bottle", "milo", "redbull"], conf_thres=0.35, iou_thres=0.5, img_sz = 640): 
        """
            init parameters funtion
            input:
                weight -> string: path to onnx weights
                class_names -> list: classes name
                conf_thres -> float[0->1]: confidence threshold
                iou_thres -> float[0->1] : iou threshold
                img_sz -> int : image input size
        """
        cuda = False
        #Loading the ONNX inference session.
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(weight, providers=providers)

        # get paramerters
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.class_names = class_names
        self.img_sz = img_sz
        self.outname = [i.name for i in self.session.get_outputs()]
        self.inname = [i.name for i in self.session.get_inputs()]

        # Create a list of colors for each class where each color is a tuple of 3 integer values
        self.colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(self.class_names)}
        
        # outputs after detect object
        self.outputs = [] # batch_id, x0, y0, x1, y1, cls_id, score

    
    def letterbox(self, im, new_shape=(320, 320), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        """"
            Resize and pad image while meeting stride-multiple constraints
            input:
                im-> array[W,H,3]: input image to resize
                new_shape-> (int,int): output size
                color->(114,114,114) : padding color
            ouput:
                im-> array[W,H,3]: ouput image
                r-> float: scale ratio
                dw-> float: width padding
                dh-> float: hight padding
        """
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, r, (dw, dh)

    def detect_objects(self, image):
        """
            detect objetc function
            input:
                image -> array[W, H, 3] : input image to detect
                output-> array[W, H, 3] : output image after detect
        """
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = img.copy()
        image, ratio, dwdh = self.letterbox(image,new_shape=(self.img_sz, self.img_sz), auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)
        im = image.astype(np.float32)
        im /= 255

        # detect object
        inp = {self.inname[0]:im}
        self.outputs = self.session.run(self.outname, inp)[0]

        

        #Visualizing bounding box prediction.
        ori_images = [img.copy()]
        for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(self.outputs):
            image = ori_images[int(batch_id)]
            box = np.array([x0,y0,x1,y1])
            box -= np.array(dwdh*2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()
            cls_id = int(cls_id)
            score = round(float(score),3)
            name = self.class_names[cls_id]
            color = self.colors[name]
            name += ' '+str(score)
            cv2.rectangle(image,box[:2],box[2:],color,2)
            cv2.putText(image,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)
        
        out_img = ori_images[0]
        out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
        return out_img

if __name__ == "__main__":
    import os   

    # path to onnx model weight
    weight = "./EduBinYolov7_4_2.onnx" 

    # path to test images folder
    path = "./test/images/"

    # classes name
    class_names = ["bottle", "milo", "redbull"]

    # define detector
    yolov7_detector = YOLOv7(weight, class_names)

    # loop in test folder
    for file in os.listdir(path):
        img = cv2.imread(path + file)
        # detect image
        out_img = yolov7_detector.detect_objects(img)
        out_img = cv2.resize(out_img, (720,720))

        # show images
        cv2.imshow("out_image", out_img)
        cv2.waitKey(300)
 