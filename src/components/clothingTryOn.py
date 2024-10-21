from diffusers import StableDiffusionInpaintPipeline
from src.utils.exceptions import CustomException
from cvzone.PoseModule import PoseDetector
from src.utils.functions import getConfig
from src.utils.logger import logger
from PIL.ImageOps import grayscale
from PIL import Image
import numpy as np
import cvzone
import torch
import math
import cv2
import gc

class ClothingTryOn:
    """
    A class to simulate clothing try-ons by overlaying clothing images on user images 
    and generating modified outputs using inpainting techniques.

    This class utilizes a pose detection model to identify key landmarks on the user's 
    body, allowing for accurate placement and scaling of clothing images. It integrates 
    with a Stable Diffusion model for image generation, providing realistic visual 
    outputs based on specified prompts while ensuring that jewelry and accessories 
    do not interfere with the clothing representation.

    Attributes:
        detector (PoseDetector): An instance of PoseDetector for identifying body landmarks.
        config (ConfigParser): Configuration settings loaded from an external config file.
        pipeline (StableDiffusionInpaintPipeline): The Stable Diffusion inpainting model for 
            generating images based on user prompts and masks.

    Methods:
        getBinaryMask(image: Image.Image, jewellery: Image.Image) -> tuple[Image.Image]:
            Generates a binary mask indicating the presence of the clothing on the user's image.
        
        generateImage(image: Image.Image, mask: Image.Image) -> tuple[Image.Image]:
            Applies inpainting to an image using the provided binary mask, generating new images 
            based on specific color prompts while excluding jewelry and accessories.
    """

    def __init__(self):
        """Initialize the NecklaceTryOn class with a PoseDetector and configuration settings."""
        self.detector = PoseDetector()
        self.config = getConfig("config.ini")
        modelId = self.config.get("CLOTHING TRY ON", "modelId")
        device = self.config.get("CLOTHING TRY ON", "device")
        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            modelId, torch_dtype = torch.float16
        ).to(device)

    def getBinaryMask(self, image: Image.Image, jewellery: Image.Image) -> tuple[Image.Image]:
        """
        Generate a binary mask indicating the presence of the necklace on the user's image.

        This function overlays a jewelry image on the user's image and creates a binary mask, where
        the necklace is represented in white and the background in black.

        Args:
            image (Image.Image): The user's image, ideally captured in a standing, upright position.
            jewellery (Image.Image): The image of the jewelry piece (e.g., necklace) to be overlaid.

        Returns:
            tuple[Image.Image]: A tuple containing:
                - The first image as the necklace try-on output.
                - The second image as the binary mask, with the necklace shown in white and the background in black.

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
                return self.getBinaryMask(
                    Image.fromarray(copyImage), Image.fromarray(jewellery)
                )
            else:
                tryOnOutput = cvzone.overlayPNG(copyImage, jewellery, (avgX1, yCoordinate))
                tryOnOutput = Image.fromarray(tryOnOutput.astype(np.uint8))
                blackedNecklace = np.zeros(shape = copyImage.shape)
                cvzone.overlayPNG(blackedNecklace, jewellery, (avgX1, yCoordinate))
                blackedNecklace = cv2.cvtColor(blackedNecklace.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                binaryMask = blackedNecklace * ((blackedNecklace > 5) * 255)
                binaryMask[binaryMask >= 255] = 255
                binaryMask[binaryMask < 255] = 0
                binaryMask = Image.fromarray(binaryMask.astype(np.uint8))
                return (tryOnOutput, binaryMask)

        except Exception as e:
            logger.error(CustomException(e))
            print(CustomException(e))


    def generateImage(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image]:
        """
        Apply inpainting to an image using the provided binary mask.

        This function utilizes the binary mask to inpaint areas of the image, enhancing the visual output
        by generating new images based on specific color prompts while excluding jewelry and other accessories.

        Args:
            image (Image.Image): The input image where inpainting will be applied.
            mask (Image.Image): The binary mask indicating areas to be inpainted.

        Returns:
            tuple: A tuple containing three images generated based on different color prompts.

        Raises:
            CustomException: If an error occurs during the image processing.
        """
        try:
            logger.info("creating a mask where the jewellery is represented")
            jewelleryMask = Image.fromarray(np.bitwise_and(np.array(mask.convert("RGB")), np.array(image.convert("RGB"))))
            arrOrig = np.array(grayscale(mask))

            logger.info("inpainting the image using the original mask")
            image = cv2.inpaint(np.array(image), arrOrig, 15, cv2.INPAINT_TELEA)
            image = Image.fromarray(image)

            logger.info("preparing the mask for processing")
            arr = arrOrig.copy()
            maskY = np.where(arr == arr[arr != 0][0])[0][0]
            arr[maskY:, :] = 255
            newMask = Image.fromarray(arr)
            mask = newMask.copy()

            logger.info("resizing images for consistency")
            origSize = image.size
            image = image.resize((512, 512))
            mask = mask.resize((512, 512))

            logger.info("generating images for different colors")
            results = []
            for colour in ["Red", "Blue", "Green"]:
                prompt = f"{colour}, South Indian Saree, properly worn, natural setting, elegant, natural look, neckline without jewellery, simple"
                negativePrompt = ("necklaces, jewellery, jewelry, necklace, neckpiece, garland, chain, neck wear, "
                                "jewelled neck, jeweled neck, necklace on neck, jewellery on neck, accessories, "
                                "watermark, text, changed background, wider body, narrower body, bad proportions, "
                                "extra limbs, mutated hands, changed sizes, altered proportions, unnatural body proportions, "
                                "blurry, ugly")
                output = self.pipeline(
                    prompt = prompt,
                    negative_prompt = negativePrompt,
                    image = image,
                    mask_image = mask,
                    strength = 0.95,
                    guidance_score = 9,
                ).images[0]

                logger.info("resizing the output to original size")
                output = output.resize(origSize)
                tempGenerated = np.bitwise_and(
                    np.array(output),
                    np.bitwise_not(np.array(Image.fromarray(arrOrig).convert("RGB"))),
                )
                results.append(tempGenerated)

            logger.info("combining the results with the jewellery mask")
            results = [
                Image.fromarray(np.bitwise_or(x, np.array(jewelleryMask))) for x in results
            ]

            logger.info("Image generation completed successfully.")
            gc.collect()
            torch.cuda.empty_cache()
            return (results[0], results[1], results[2])

        except Exception as e:
            logger.error(CustomException(e))
            print(CustomException(e))