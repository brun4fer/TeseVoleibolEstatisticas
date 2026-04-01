import cv2
import os
import shutil

VIDEO_PATH = r"C:\Users\Utilizador\Desktop\Mestrado\Tese\VideosJogos\VideoAcademica.mp4"
OUTPUT_DIR = "frames"
MAX_IMAGES = 1000

START_TIME = "00:29:02"

def time_to_seconds(t):
    h, m, s = map(int, t.split(":"))
    return h * 3600 + m * 60 + s

# 🔥 1. Apagar pasta antiga
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 🔥 2. Abrir vídeo
cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
start_frame = int(time_to_seconds(START_TIME) * fps)

# 🔥 3. Ir diretamente para o minuto certo
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret or saved_count >= MAX_IMAGES:
        break

    # 🔥 guardar 1 a cada 3 frames (melhor variedade)
    if frame_count % 3 == 0:
        filename = os.path.join(OUTPUT_DIR, f"frame_{saved_count}.jpg")
        cv2.imwrite(filename, frame)
        saved_count += 1

    frame_count += 1

cap.release()

print(f"Frames guardados: {saved_count}")