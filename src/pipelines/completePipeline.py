from src.components.necklaceTryOn import NecklaceTryOn
from src.components.clothingTryOn import ClothingTryOn
from PIL import Image

class Pipeline:
    """
    A pipeline for performing jewelry and clothing try-on operations.

    This class encapsulates the functionality for overlaying jewelry 
    and clothing on user images using the NecklaceTryOn and ClothingTryOn
    components.

    Attributes:
        necklaceTryOnObject (NecklaceTryOn): Instance for necklace try-on functionality.
        clothingTryOnObject (ClothingTryOn): Instance for clothing try-on functionality.
    """

    def __init__(self):
        """
        Initializes the Pipeline with instances of NecklaceTryOn and ClothingTryOn.

        This constructor sets up the necessary objects required for the 
        try-on functionalities.
        """
        self.necklaceTryOnObject = NecklaceTryOn()
        self.clothingTryOnObject = ClothingTryOn()

    def necklaceTryOn(self, image: Image.Image, jewellery: Image.Image) -> Image.Image:
        """
        Overlay a necklace image onto the user's image.

        Args:
            image (Image.Image): The user's image, ideally captured in a standing position.
            jewellery (Image.Image): The image of the necklace to be overlaid.

        Returns:
            Image.Image: A PIL Image depicting the user wearing the specified necklace.
        """
        result = self.necklaceTryOnObject.necklaceTryOn(image = image, jewellery = jewellery)
        return result
    
    def clothingTryOn(self, image: Image.Image, jewellery: Image.Image) -> Image.Image:
        """
        Simulate wearing clothing on the user's image and generate the final output.

        Args:
            image (Image.Image): The user's image, ideally captured in a standing position.
            jewellery (Image.Image): The image of the clothing item to be overlaid.

        Returns:
            Image.Image: A PIL Image depicting the user wearing the specified clothing.
        """
        tryOnOutput, mask = self.clothingTryOnObject.getBinaryMask(image = image, jewellery = jewellery)
        results = self.clothingTryOnObject.generateImage(image = tryOnOutput, mask = mask)
        return results