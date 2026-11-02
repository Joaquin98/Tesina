# Tesina

Modelos M0, M1, M2 y M3 para clasificar jugadas de rugby.

Requisitos: Python 3.13, Git y Git LFS. Los comandos son para Windows (PowerShell).

## 1. Descargar

```powershell
git lfs install
git clone https://github.com/Joaquin98/Tesina.git
cd Tesina
git lfs pull
```

Git LFS descarga los videos y los pesos de los modelos.

## 2. Instalar las dependencias

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Ejecutar los siguientes comandos desde la carpeta `Tesina`.
Los ejemplos usan `M0`. Para otro modelo, reemplazar `M0` por `M1`, `M2` o `M3` en el comando y en las rutas.

## 3. Entrenar

```powershell
.\.venv\Scripts\python.exe tools/modelos.py train --family M0 --output models/M0/runs/entrenamiento_1
```

Usa los conjuntos de entrenamiento y validación de `data/clips_dataset`.
Genera y reutiliza la caché local, que no se sube a GitHub.
Para repetir el entrenamiento, usar otro nombre de carpeta de salida.

## 4. Predecir

Con los pesos incluidos, sin entrenar antes:

```powershell
.\.venv\Scripts\python.exe tools/modelos.py predict --family M0 --video data/videos/clips_224/C0881.mp4
```

Con el modelo guardado en el paso 3:

```powershell
.\.venv\Scripts\python.exe tools/modelos.py predict --family M0 --video data/videos/clips_224/C0881.mp4 --run models/M0/runs/entrenamiento_1
```

Para analizar otro video, cambiar la ruta indicada en `--video`.
Las predicciones se muestran en la terminal.
