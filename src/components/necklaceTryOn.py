from src.utils.exceptions import CustomException
from cvzone.PoseModule import PoseDetector
from src.utils.functions import getConfig
from src.utils.logger import logger
from PIL import Image
import numpy as np
import cvzone
import math
import cv2

class NecklaceTryOn:
    """
    A class for simulating the wearing of necklaces in images.

    This class utilizes a pose detection algorithm to accurately overlay 
    a necklace image onto a user's photo, adjusting for the user's neck 
    position and orientation.

    Attributes:
        detector (PoseDetector): An instance of the PoseDetector for identifying 
            body landmarks in images.
        config (ConfigParser): Configuration settings loaded from a specified 
            configuration file (config.ini).

    Methods:
        necklaceTryOn(image: Image.Image, jewellery: Image.Image) -> Image.Image:
            Overlays a necklace onto the user's image based on detected pose 
            landmarks and returns the resulting image.
    """
    
    def __init__(self):
        """Initialize the NecklaceTryOn class with a PoseDetector and configuration settings."""
        self.detector = PoseDetector()
        self.config = getConfig("config.ini")

    def necklaceTryOn(self, image: Image.Image, jewellery: Image.Image) -> Image.Image:
        """
        Overlay a jewelry image onto a person's image to simulate wearing the jewelry.

        Args:
            image (Image.Image): The user's image, ideally captured in a standing, upright position.
            jewellery (Image.Image): The image of the jewelry piece (e.g., necklace) to be overlaid.

        Returns:
            Image.Image: A PIL Image depicting the user wearing the specified jewelry.

        Raises:
            CustomException: If an error occurs during the image processing.
        """
        try:
            logger.info("converting images to numpy arrays")
            image = np.array(image)
            jewellery = np.array(jewellery)

            logger.info("creating a copy of original image for actual overlay")
            copyImage = image.copy()
            
            logger.info("detecting body landmarks from the input image")
            image = self.detector.findPose(image)
            lmList, _ = self.detector.findPosition(image, bboxWithHands = False, draw = False)
            pt12, pt11, pt10, pt9 = (
                lmList[12][:2],
                lmList[11][:2],
                lmList[10][:2],
                lmList[9][:2],
            )        

            logger.info("calculating the precise neck points")
            avgX1 = int(pt12[0] + (pt10[0] - pt12[0]) / 1.75)
            avgY1 = int(pt12[1] - (pt12[1] - pt10[1]) / 1.75)
            avgX2 = int(pt11[0] - (pt11[0] - pt9[0]) / 1.75)
            avgY2 = int(pt11[1] - (pt11[1] - pt9[1]) / 1.75)

            logger.info("rescaling the necklace to appropriate dimensions")
            xDist = avgX2 - avgX1
            origImgRatio = xDist / jewellery.shape[1]
            yDist = jewellery.shape[0] * origImgRatio
            jewellery = cv2.resize(
                jewellery, (int(xDist), int(yDist)), interpolation = cv2.INTER_CUBIC
            )        

            logger.info("calculating required offset to be added to the necklace image for perfect fitting")
            imageGray = cv2.cvtColor(jewellery, cv2.COLOR_BGRA2GRAY)
            for offsetOrig in range(imageGray.shape[1]):
                pixelValue = imageGray[0, :][offsetOrig]
                if (pixelValue != 255) & (pixelValue != 0):
                    break
                else:
                    continue
            offset = int(self.config.getfloat("NECKLACE TRY ON", "offsetFactor") * xDist * (offsetOrig / jewellery.shape[1]))
            yCoordinate = avgY1 - offset

            logger.info("tilting the necklace image as per the necklace points")
            angle = math.ceil(
                self.detector.findAngle(
                    p1 = (avgX2, avgY2), p2 = (avgX1, avgY1), p3 = (avgX2, avgY1)
                )[0]
            )
            if avgY2 < avgY1:
                pass
            else:
                angle = angle * -1
            jewellery = cvzone.rotateImage(jewellery, angle)

            logger.info("checking if the necklace is getting out of the frame and trimming from above if needed")
            availableSpace = copyImage.shape[0] - yCoordinate
            extra = jewellery.shape[0] - availableSpace

            logger.info("applying the calculated settings")
            if extra > 0:
                jewellery = jewellery[extra + 10 :, :]
                return self.necklaceTryOn(
                    Image.fromarray(copyImage), Image.fromarray(jewellery)
                )
            else:
                result = cvzone.overlayPNG(copyImage, jewellery, (avgX1, yCoordinate))
                result = Image.fromarray(result.astype(np.uint8))
                return result
        
        except Exception as e:
            logger.error(CustomException(e))
            print(CustomException(e))