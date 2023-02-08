from Yolov7onnx import YOLOv7
import os   
import cv2
if __name__ == "__main__":
   

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
        id = yolov7_detector.getIdOject(img)
        # out_img = cv2.resize(out_img, (720,720))
        print(id)
        # show images
        cv2.imshow("out_image", img)
        cv2.waitKey(0)
 