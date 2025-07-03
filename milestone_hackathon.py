import os
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info # Assuming this utility is available

# Setare automată pe CUDA dacă e disponibil
device = "cuda" if torch.cuda.is_available() else "cpu"

# Încarcă modelul cu flash_attention_2 (dacă e suportat)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype="auto",
    # attn_implementation="flash_attention_2", # Uncomment if supported and causes issues
    device_map="auto",
)
# Procesorul (folosește tokenizer, image processor etc.)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# Resize funcțional (512x512)
def resize_image(path, size=(512, 512)):
    img = Image.open(path).convert("RGB")
    return img.resize(size)

# === Procesează multiple videoclipuri ===
base_frames_dir = "./frames" # The main directory containing video subfolders
output_master_file = "all_videos_anomaly_report.txt" # New: Single output file name

# Get a list of all video subdirectories
video_dirs = [
    os.path.join(base_frames_dir, d)
    for d in os.listdir(base_frames_dir)
    if os.path.isdir(os.path.join(base_frames_dir, d))
]

if not video_dirs:
    print(f"No video directories found in {base_frames_dir}. Please ensure your frames are organized as frames/video_name/frame.jpg")

# Open the master output file once in write mode to clear it (or 'a' if you want to append to existing)
# We'll use 'w' to start fresh each time the script runs.
with open(output_master_file, "w", encoding="utf-8") as f_master:
    f_master.write("--- Anomaly Report for All Videos ---\n\n")

for video_dir in video_dirs:
    video_name = os.path.basename(video_dir)
    print(f"\nProcessing frames for video: {video_name}")

    # Open the master file in append mode within the loop for writing
    # This ensures that even if an error occurs for a later video, previous results are saved.
    with open(output_master_file, "a", encoding="utf-8") as f_master:
        f_master.write(f"\n--- Video: {video_name} ---\n")

    try:
        # === 1. Încarcă și procesează imaginile din curentul director video ===
        image_files = sorted([
            os.path.join(video_dir, f)
            for f in os.listdir(video_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])[::3]  # fiecare al 3-lea frame

        if not image_files:
            print(f"No image files found in {video_dir}. Skipping this video.")
            with open(output_master_file, "a", encoding="utf-8") as f_master:
                f_master.write("  No image files found. Skipped.\n")
            continue

        # prompt = "Analyze these images of a public space and try to decide if there is an anomaly. I want a short description of the scene and people involved. Be careful \
        #     on the bags that people carry cuz if they are carrying a bag that is not theirs, it can be a suspicious activity. I want a clear decision like 'STATUS: NORMAL' or 'STATUS: ANOMALY' \
        #         and a detailed description of the anomaly if it is an anomaly. "

        prompt =     "Based on these images of a public space, perform the following steps:\n \
    1. **Scene Overview:** Provide a concise description of the overall scene, including the environment, general activities, and a count of people involved.\n\
    2. **Anomaly Check (Bags and General Suspicion):** Carefully look for any unusual, suspicious, or potentially problematic activities. **Pay very close attention to how people are interacting with bags.** Specifically, consider if someone appears to be taking a bag that clearly does not belong to them, or if there's any sudden snatching, dropping, or unusual exchange of bags. Also look for aggressive behavior, vandalism, or suspicious unattended items.\n\
    3. **Decision & Details:** Conclude with a clear status line. If any anomaly is detected, respond with 'STATUS: ANOMALY' immediately followed by a detailed description of the anomaly, including who is involved, their specific actions (especially regarding bags), and why it's considered unusual or concerning. If no anomalies are detected, respond with 'STATUS: NORMAL' and briefly state that the scene appears typical."


        # Using the anomaly prompt
        message_content = [{"type": "image", "image": resize_image(p)} for p in image_files]
        message_content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": message_content}]

        # === 2. Pregătește datele pentru model ===
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # === 3. Generează output ===
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512)

        output_text = processor.batch_decode(
            generated_ids[:, inputs['input_ids'].shape[1]:], # Corrected: Access 'input_ids' from the dict
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        # === 4. Scrie rezultatul în fișierul master ===
        with open(output_master_file, "a", encoding="utf-8") as f_master:
            f_master.write(output_text[0] + "\n")

        print(f"✅ Output for {video_name} generated and appended to {output_master_file}:")
        print(output_text[0])

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            error_message = f"❌ CUDA Out of Memory for video: {video_name}. Skipping to next video."
            print(error_message)
            with open(output_master_file, "a", encoding="utf-8") as f_master:
                f_master.write(error_message + "\n")
        else:
            error_message = f"An unexpected error occurred while processing {video_name}: {e}"
            print(error_message)
            with open(output_master_file, "a", encoding="utf-8") as f_master:
                f_master.write(error_message + "\n")
    finally:
        # Aggressive memory cleanup
        if device == "cuda":
            if 'message_content' in locals(): del message_content
            if 'messages' in locals(): del messages
            if 'text' in locals(): del text
            if 'image_inputs' in locals(): del image_inputs
            if 'video_inputs' in locals(): del video_inputs
            if 'inputs' in locals(): del inputs
            if 'generated_ids' in locals(): del generated_ids
            if 'output_text' in locals(): del output_text # Only delete if it's there
            torch.cuda.empty_cache()
            print(f"  GPU memory cleared for {video_name}.")

print(f"\nAll video processing complete. Full report saved to {output_master_file}")