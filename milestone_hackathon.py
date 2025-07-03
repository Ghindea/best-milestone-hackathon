import os
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Setare automată pe CUDA dacă e disponibil
device = "cuda" if torch.cuda.is_available() else "cpu"

# Încarcă modelul cu flash_attention_2 (dacă e suportat)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype="auto",
    # attn_implementation="flash_attention_2",
    device_map="auto",
)

# Procesorul (folosește tokenizer, image processor etc.)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# === 1. Încarcă și procesează imaginile din photos/ ===
image_dir = "./photos"

# Preia toate fișierele imagine din folder, sortează, și selectează 1 din 3
image_files = sorted([
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])[::3]  # fiecare al 3-lea frame

# Resize funcțional (512x512)
def resize_image(path, size=(512, 512)):
    img = Image.open(path).convert("RGB")
    return img.resize(size)

# Creează mesajul cu toate imaginile redimensionate și o singură întrebare
message_content = [{"type": "image", "image": resize_image(p)} for p in image_files]
message_content.append({"type": "text", "text": "Describe the scene in the images as a coherent story, with details about the people and the environment."})

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

inputs = inputs.to(device)

# === 3. Generează output ===
generated_ids = model.generate(**inputs, max_new_tokens=512)

# Curăță output-ul
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)

# === 4. Scrie rezultatul în fișier ===
with open("output.txt", "w", encoding="utf-8") as f:
    for line in output_text:
        f.write(line + "\n")

print("✅ Output generat și salvat în output.txt:")
print(output_text[0])
