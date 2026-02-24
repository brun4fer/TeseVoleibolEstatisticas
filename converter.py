import os
import subprocess
from pathlib import Path
from static_ffmpeg import add_paths

# Esta linha adiciona automaticamente o FFmpeg ao caminho do script
add_paths()

def convert_ts_to_mp4(input_dir):
    folder = Path(input_dir)
    ts_files = list(folder.glob("*.ts"))
    
    if not ts_files:
        print("Nenhum ficheiro .ts encontrado.")
        return

    for ts_file in ts_files:
        output_file = ts_file.with_suffix(".mp4")
        print(f"A converter: {ts_file.name} -> {output_file.name}...")
        
        # Usamos -c copy para ser instantâneo e manter a qualidade original
        command = [
            "ffmpeg", "-i", str(ts_file),
            "-c", "copy", "-y",
            str(output_file)
        ]
        
        try:
            # Correr o comando
            subprocess.run(command, check=True)
            print(f"✅ Sucesso!")
        except Exception as e:
            print(f"❌ Erro: {e}")

if __name__ == "__main__":
    videos_path = r"C:\Users\Utilizador\Desktop\Mestrado\Tese\VideosJogos"
    convert_ts_to_mp4(videos_path)