import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info # Make sure this utility is available

def main():
    # --- Configuration for Image Folder and Output ---
    # !!! IMPORTANT: Define your image folder path here !!!
    # This path should be accessible from the compute node where the bsub job runs.
    # Examples:
    # 1. Absolute path on a shared file system:
    # IMAGE_FOLDER_PATH = "/path/to/your/shared_storage/my_images_for_qwen"
    # 2. Relative path (assuming images are in a subfolder where the script is run):
    IMAGE_FOLDER_PATH = "./images_to_process" 
    # If the bsub job changes directories, make sure this path is correct relative to the new CWD.

    OUTPUT_FILE_NAME = "output_7b_dynamic.txt" # Name for the output file
    MODEL_PROMPT = "Describe the main subjects and the overall scene in these images."

    # --- Load Model and Processor ---
    print("Loading model and processor...")
    # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )

    # default processor
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    print("Model and processor loaded.")

    # --- Prepare Image Paths from Folder ---
    image_paths = []
    if not os.path.isdir(IMAGE_FOLDER_PATH):
        print(f"Error: The specified image folder '{IMAGE_FOLDER_PATH}' does not exist or is not accessible.")
        print("Please ensure the folder exists and is accessible from the compute node.")
        return

    # List common image file extensions (you can expand this list)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

    for filename in os.listdir(IMAGE_FOLDER_PATH):
        if filename.lower().endswith(valid_extensions):
            full_path = os.path.join(IMAGE_FOLDER_PATH, filename)
            image_paths.append(full_path)

    if not image_paths:
        print(f"No valid image files found in '{IMAGE_FOLDER_PATH}'. Exiting.")
        return

    # Sort image paths for consistent processing order (optional but good practice)
    image_paths.sort()
    print(f"Found {len(image_paths)} images in '{IMAGE_FOLDER_PATH}'.")

    # --- Construct Messages for Model ---
    messages_content = []
    for img_path in image_paths:
        messages_content.append({"type": "image", "image": img_path})
    messages_content.append({"type": "text", "text": MODEL_PROMPT})

    messages = [
        {
            "role": "user",
            "content": messages_content,
        }
    ]
    print("Messages constructed for the model.")

    # --- Preparation for Inference ---
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    # Ensure inputs are moved to the correct device
    print("Preparing inputs for model...")
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda") # Move to GPU directly here
    print("Inputs ready.")

    # --- Inference: Generation of the output ---
    print("Generating response from the model...")
    generated_ids = model.generate(**inputs, max_new_tokens=512) # Increased max_new_tokens for potentially longer descriptions
    
    # Handle cases where input_ids length might differ if padding added tokens
    input_ids_len = inputs.input_ids.shape[1]
    generated_ids_trimmed = [
        out_ids[input_ids_len:] for out_ids in generated_ids
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("Response generated.")
    
    # --- Save and Print Output ---
    with open(OUTPUT_FILE_NAME, "w", encoding="utf-8") as f:
        for line in output_text:
            f.write(line + "\n")
    print(f"Saved output to {OUTPUT_FILE_NAME}")
    
    # Optional: Also print to console
    for i, text_out in enumerate(output_text):
        print(f"\n--- Generated Output {i+1} ---")
        print(text_out)


if __name__ == "__main__":
    main()