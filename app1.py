### FINAL COMBINED
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
import os
import gradio as gr
import numpy as np
from PIL import Image
from PIL.ImageOps import grayscale
import cv2
import torch
import gc
import math
import cvzone
from cvzone.PoseModule import PoseDetector

# images
choker_images = [Image.open(os.path.join("choker", x)) for x in os.listdir("choker")]
short_necklaces = [Image.open(os.path.join("short_necklace", x)) for x in os.listdir("short_necklace")]
long_necklaces = [Image.open(os.path.join("long_haram", x)) for x in os.listdir("long_haram")]
person_images = [Image.open(os.path.join("without_necklace", x)) for x in os.listdir("without_necklace")]

# initialising the stable diffusion model
model_id = "stabilityai/stable-diffusion-2-inpainting"
pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16
)
pipeline = pipeline.to("cuda")

# functions
def clearFunc():
    torch.cuda.empty_cache()
    gc.collect()

def necklaceTryOnPipeline(image, jewellery):
    global binaryMask
    
    image = np.array(image)
    copy_image = image.copy()
    jewellery = np.array(jewellery)

    detector = PoseDetector()

    image = detector.findPose(image)
    lmList, bBoxInfo = detector.findPosition(image, bboxWithHands=False, draw=False)

    pt12, pt11, pt10, pt9 = (
        lmList[12][:2],
        lmList[11][:2],
        lmList[10][:2],
        lmList[9][:2],
    )

    avg_x1 = int(pt12[0] + (pt10[0] - pt12[0]) / 2)
    avg_y1 = int(pt12[1] - (pt12[1] - pt10[1]) / 2)

    avg_x2 = int(pt11[0] - (pt11[0] - pt9[0]) / 2)
    avg_y2 = int(pt11[1] - (pt11[1] - pt9[1]) / 2)

    image_gray = cv2.cvtColor(jewellery, cv2.COLOR_BGRA2GRAY)

    if avg_y2 < avg_y1:
        angle = math.ceil(
            detector.findAngle(
                p1=(avg_x2, avg_y2), p2=(avg_x1, avg_y1), p3=(avg_x2, avg_y1)
            )[0]
        )
    else:
        angle = math.ceil(
            detector.findAngle(
                p1=(avg_x2, avg_y2), p2=(avg_x1, avg_y1), p3=(avg_x2, avg_y1)
            )[0]
        )
        angle = angle * -1

    xdist = avg_x2 - avg_x1
    origImgRatio = xdist / jewellery.shape[1]
    ydist = jewellery.shape[0] * origImgRatio

    for offset_orig in range(image_gray.shape[1]):
        pixel_value = image_gray[0, :][offset_orig]
        if (pixel_value != 255) & (pixel_value != 0):
            break
        else:
            continue

    offset = int(0.8 * xdist * (offset_orig / jewellery.shape[1]))
    jewellery = cv2.resize(
        jewellery, (int(xdist), int(ydist)), interpolation=cv2.INTER_CUBIC
    )
    jewellery = cvzone.rotateImage(jewellery, angle)
    y_coordinate = avg_y1 - offset
    available_space = copy_image.shape[0] - y_coordinate
    extra = jewellery.shape[0] - available_space
    if extra > 0:
        jewellery = jewellery[extra + 10 :, :]
        return necklaceTryOnPipeline(
            Image.fromarray(copy_image), Image.fromarray(jewellery)
        )
    else:
        result = cvzone.overlayPNG(copy_image, jewellery, (avg_x1, y_coordinate))
        # masking
        blackedNecklace = np.zeros(shape = copy_image.shape)
        # overlay
        cvzone.overlayPNG(blackedNecklace, jewellery, (avg_x1, y_coordinate))
        blackedNecklace = cv2.cvtColor(blackedNecklace.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        binaryMask = blackedNecklace * ((blackedNecklace > 5) * 255)
        binaryMask[binaryMask >= 255] = 255
        binaryMask[binaryMask < 255] = 0
        return Image.fromarray(result.astype(np.uint8)), Image.fromarray(binaryMask.astype(np.uint8))

# SD Model
def sd_inpaint(image, mask):
#    image = Image.fromarray(image)
#    mask = Image.fromarray(mask)

    jewellery_mask = Image.fromarray(
        np.bitwise_and(np.array(mask), np.array(image))
    )
    arr_orig = np.array(grayscale(mask))

    image = cv2.inpaint(np.array(image), arr_orig, 15, cv2.INPAINT_TELEA)
    image = Image.fromarray(image)

    arr = arr_orig.copy()
    mask_y = np.where(arr == arr[arr != 0][0])[0][0]
    arr[mask_y:, :] = 255

    new = Image.fromarray(arr)

    mask = new.copy()

    orig_size = image.size

    image = image.resize((512, 512))
    mask = mask.resize((512, 512))

    results = []
    for colour in ["Red", "Blue", "Green"]:
        prompt = f"{colour}, South Indian Saree, properly worn, natural setting, elegant, natural look, neckline without jewellery, simple"
        negative_prompt = "necklaces, jewellery, jewelry, necklace, neckpiece, garland, chain, neck wear, jewelled neck, jeweled neck, necklace on neck, jewellery on neck, accessories, watermark, text, changed background, wider body, narrower body, bad proportions, extra limbs, mutated hands, changed sizes, altered proportions, unnatural body proportions, blury, ugly"

        output = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            strength=0.95,
            guidance_score=9,
            # generator = torch.Generator("cuda").manual_seed(42)
        ).images[0]

        output = output.resize(orig_size)
        temp_generated = np.bitwise_and(
            np.array(output),
            np.bitwise_not(np.array(Image.fromarray(arr_orig).convert("RGB"))),
        )
        results.append(temp_generated)

    results = [
        Image.fromarray(np.bitwise_or(x, np.array(jewellery_mask))) for x in results
    ]
    clearFunc()
    return results[0], results[1], results[2]

# interface

with gr.Blocks() as interface:
  with gr.Row():
    inputImage = gr.Image(label = "Input Image", type = "pil", image_mode = "RGB", interactive = True)
    selectedNecklace = gr.Image(label = "Selected Necklace", type = "pil", image_mode = "RGBA", visible = False)
    hiddenMask = gr.Image(visible = False, type = "pil")
  with gr.Row():
    gr.Examples(examples = person_images, inputs=[inputImage], label="Models")
  with gr.Row():
    gr.Examples(examples = choker_images, inputs = [selectedNecklace], label = "Chokers")
  with gr.Row():
    gr.Examples(examples = short_necklaces, inputs = [selectedNecklace], label = "Short Necklaces")
  with gr.Row():
    gr.Examples(examples = long_necklaces, inputs = [selectedNecklace], label = "Long Necklaces")
  with gr.Row():
    outputOne = gr.Image(label = "Output 1", interactive = False)
    outputTwo = gr.Image(label = "Output 2", interactive = False)
    outputThree = gr.Image(label = "Output 3", interactive = False)
  with gr.Row():
    submit = gr.Button("Enter")

  selectedNecklace.change(fn = necklaceTryOnPipeline, inputs = [inputImage, selectedNecklace], outputs = [inputImage, hiddenMask])
  submit.click(fn = sd_inpaint, inputs = [inputImage, hiddenMask], outputs = [outputOne, outputTwo, outputThree])


interface.launch(debug = True)