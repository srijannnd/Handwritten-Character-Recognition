import cv2
import numpy as np
from keras.models import load_model
import sys
import getopt


def main(argv):
    inp_pic = "timages/"
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print('test.py -i <inputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <inputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inp_pic += arg

    # Load the Keras CNN trained model
    model = load_model('final_try.h5')

    # Original image
    im = cv2.imread(inp_pic)
    cv2.imshow("Original Image", im)
    cv2.waitKey()

    # Read image in grayscale mode
    img = cv2.imread(inp_pic, 0)

    # Median Blur and Gaussian Blur to remove Noise
    img = cv2.medianBlur(img, 3)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Adaptive Threshold for handling lightning
    im_th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
    # cv2.imshow("Threshold Image",im_th)
    kernel = np.ones((1, 1), np.uint8)
    im_th = cv2.dilate(im_th, kernel, iterations=4)
    cv2.imshow("After", im_th)
    ##################################################################################################


    # Find contours in the image
    ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get rectangles contains each contour
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]

    # For each rectangular region, predict using cnn model
    for rect in rects:
        # Draw the rectangles
        cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
        # Make the rectangular region around the digit
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = im_th[pt1:pt1 + leng, pt2:pt2 + leng]
        # Resize the image
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)

        # Input for CNN Model
        roi = roi[np.newaxis, np.newaxis, :, :]

        # Input for Feed Forward Model
        # roi = roi.flatten()
        # roi = roi[np.newaxis]
        nbr = model.predict_classes(roi, verbose=0)
        char = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'a', 11: 'b',
                12: 'c', 13: 'd', 14: 'e', 15: 'f', 16: 'g', 17: 'h', 18: 'i', 19: 'j', 20: 'k', 21: 'l', 22: 'm',
                23: 'n', 24: 'o', 25: 'p', 26: 'q', 27: 'r', 28: 's', 29: 't', 30: 'u', 31: 'v', 32: 'w', 33: 'x',
                34: 'y', 35: 'z', 36: 'A', 37: 'B', 38: 'C', 39: 'D', 40: 'E', 41: 'F', 42: 'G', 43: 'H', 44: 'I',
                45: 'J', 46: 'K', 47: 'L', 48: 'M', 49: 'N', 50: 'O', 51: 'P', 52: 'Q', 53: 'R', 54: 'S', 55: 'T',
                56: 'U', 57: 'V', 58: 'W', 59: 'X', 60: 'Y', 61: 'Z'}
        cv2.putText(im, char[nbr[0]], (int(rect[0]), int(rect[1])), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

        cv2.imshow("Resulting Image with Predicted numbers", im)
        cv2.waitKey()

        if __name__ == "__main__":
            main(sys.argv[1:])



