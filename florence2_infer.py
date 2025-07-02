
'''
below export needed when running in dgx
'''

# export HF_MODULES_CACHE=/mnt/
# export HF_TOKENIZERS_CACHE=/mnt/
# export HF_DATASETS_CACHE=/mnt/
# export TRANSFORMERS_CACHE=/mnt/
# export TORCH_HOME=/mnt/

import os
# print("Calling")
from transformers import AutoProcessor, AutoModelForCausalLM
# print("Called")
from tqdm import tqdm
import torch
import time
import matplotlib.pyplot as plt  
import matplotlib.patches as patches  
from PIL import Image, ImageDraw
import re
from threading import Thread

def divide_list_into_n_parts(input_list, n):
    elements_per_part = len(input_list) // n
    remainder = len(input_list) % n

    start = 0
    parts = []

    for i in range(n):
        end = start + elements_per_part + (1 if i < remainder else 0)

        parts.append(input_list[start:end])

        start = end

    return parts

def plot_bbox_with_tag(image_path, data, out_folder):
    fig, ax = plt.subplots()

    # Display the image
    #ax.imshow(image_path)

    for bbox, label in zip(data['bboxes'], data['labels']):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='black', alpha=0.5))

    # Remove the axis ticks and labels
    ax.axis('off')

    # os.makedirs(out_folder, exist_ok=True)
    # output_path = os.path.join(out_folder, os.path.basename(image_path))
    fig.savefig(f"{out_folder}/{image_path}")

def plot_bbox(image_path, bbox, out_folder):
    """
    Plots bounding boxes directly on the original image and saves the output.

    Parameters:
    - image_path (str): Path to the input image.
    - bbox (list of tuples): List of bounding boxes in the format (x1, y1, x2, y2).
    - out_folder (str): Path to the folder to save the output image.
    """
    image = Image.open(image_path).convert("RGB")

    draw = ImageDraw.Draw(image)

    for box in bbox:
        x1, y1, x2, y2 = box

        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

    os.makedirs(out_folder, exist_ok=True)

    output_path = os.path.join(out_folder, os.path.basename(image_path))
    image.save(output_path)
    # print(f"Image saved to {output_path}")


start = time.time()
# Load Florence 2 Large Model
def load_model():
    #Normal florence2
    model_id = 'microsoft/Florence-2-large'
    #model_id = 'microsoft/Florence-2-base'

    # finetuned florence2
    # model_id = '/raid/training_data/florence2/model_checkpoints/epoch_50'
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                trust_remote_code=True).eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    return model, processor

def infer_florence(text_prompt, model, processor, image, text_input=None):
    if text_input is None:
        prompt = text_prompt
    else:
        prompt = text_prompt + text_input
    # print(prompt)
    inputs = processor(text=prompt, images=image, return_tensors='pt').to('cuda')
    generated_ids = model.generate(
        input_ids = inputs["input_ids"],
        pixel_values = inputs["pixel_values"],
        max_new_tokens = 4096,
        early_stopping = False,
        do_sample = False,
        num_beams = 3
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task = text_prompt,
        image_size = (image.width, image.height)
    )
    return parsed_answer

# Perform object detection on a single image
def perform_object_detection(image_name, image_path, prompt, model, processor, text_input, out_folder):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")

    # Perform inference
    outputs = infer_florence(prompt, model, processor, image, text_input)
    # Parse results
    bbox = outputs["<OD>"]["bboxes"]
    # labels = outputs["<OD>"]["labels"]
    # print(bbox)

    # for i,box in enumerate(bbox):
    #     cropped_img = image.crop(box)
    #     out_img_path = os.path.join(out_folder, f"{i}_{image_name}")
    #     cropped_img.save(out_img_path)

    # plot_bbox_with_tag(image_name, outputs["<OD>"], out_folder)

    plot_bbox(image_path, bbox, out_folder)
    # return plot_im


def process_folder(folder_path, prompt, device, text_input, out_folder):
    img_folder = "/raid/training_data/florence2/ph_shelf"
    model, processor = load_model()
    model.to(device)

    for image_name in tqdm(folder_path):
        image_path = os.path.join(img_folder, image_name)
        if os.path.isfile(image_path) and image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            # print(f"Processing: {image_name}")
            im_time = time.time()
            try:
                perform_object_detection(image_name, image_path, prompt, model, processor, text_input, out_folder)
            except Exception as e:
                print(e)
                continue
            done_im_time = time.time() - im_time
        # print("Time for 1 image", done_im_time)
    


# Main function
def main():
    threads = []
    num = 2
    folder_path = "/raid/training_data/florence2/ph_shelf"  
    prompt = "<OD>"  
    text_input = None
    out_folder = "/raid/training_data/florence2/florence_ph_shelf"
    device = torch.device("cuda")
    if not os.path.exists(out_folder):
        os.makedirs(out_folder, exist_ok=True)

    data = divide_list_into_n_parts(os.listdir(folder_path), num)
    print("Loading Florence 2 Large model...")

    thread1 = Thread(target=process_folder, args=(data[0], prompt, device, text_input, out_folder,))
    thread2 = Thread(target=process_folder, args=(data[1], prompt, device, text_input, out_folder,))
    # thread3 = Thread(target=process_folder, args=(data[2], prompt, device, text_input, out_folder,))
    # thread4 = Thread(target=process_folder, args=(data[3], prompt, device, text_input, out_folder,))

    threads.append(thread1)
    threads.append(thread2)
    # threads.append(thread3)
    # threads.append(thread4)
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
    # process_folder(folder_path, prompt, model, processor, text_input, out_folder)
    print(time.time() - start)
    print(f"Object detection completed. Results saved to {out_folder}")


if __name__ == "__main__":
    main()
