import cv2
import os

# Carpeta de origen con los vídeos
source_folder = "/dtu/blackhole/00/best/data/training_videos"
# Solo procesar estos dos vídeos
video_files = [
    "2018-03-13.17-20-14.17-21-19.school.G421.r13.avi",
    "2018-03-05.13-15-01.13-20-01.bus.G331.r13.avi"
]

# Carpeta de salida para los frames
main_output_folder = "frames"
os.makedirs(main_output_folder, exist_ok=True)

for video_file in video_files:
    video_path = os.path.join(source_folder, video_file)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"No se pudo abrir {video_file}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(f"No se pudo obtener FPS de {video_file}, usando 30 por defecto")
        fps = 30  # Valor por defecto si no se puede leer

    frame_interval = int(fps * 1)  # Captura cada 1 segundo

    frame_number = 0
    saved_frame_count = 0

    # Crear subcarpeta para este vídeo
    video_name = os.path.splitext(video_file)[0]
    video_output_folder = os.path.join(main_output_folder, video_name)
    os.makedirs(video_output_folder, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_number % frame_interval == 0:
            output_name = f"frame_{frame_number}.jpg"
            output_path = os.path.join(video_output_folder, output_name)
            cv2.imwrite(output_path, frame)
            saved_frame_count += 1
        frame_number += 1

    cap.release()
    print(f"{saved_frame_count} capturas guardadas de {video_file} en {video_output_folder}")