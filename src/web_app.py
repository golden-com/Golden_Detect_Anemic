# src/web_app.py
from flask import Flask, request, jsonify, render_template_string
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Cargar el modelo (SIN MODIFICAR)
try:
    model = load_model('models/final_model.h5')
    print("✅ Modelo cargado correctamente")
except Exception as e:
    print("❌ Error al cargar el modelo:", e)
    model = None


# ============================================================
# VALIDACIÓN DE CALIDAD DE IMAGEN (nueva etapa, sección 11)
# NO detecta conjuntiva, NO decide el resultado.
# Solo comprueba brillo y nitidez mínimos con OpenCV.
# ============================================================
def validar_calidad_imagen(ruta_imagen):
    img = cv2.imread(ruta_imagen)
    if img is None:
        return False, "No se pudo leer la imagen."

    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Brillo promedio (0-255) ---
    brillo = float(np.mean(gris))
    if brillo < 40:
        return False, "La imagen está demasiado oscura. Mejora la iluminación e intenta de nuevo."
    if brillo > 235:
        return False, "La imagen tiene demasiado brillo o reflejo. Evita luz directa muy fuerte."

    # --- Nitidez: varianza del Laplaciano ---
    # Valores bajos = imagen borrosa
    nitidez = float(cv2.Laplacian(gris, cv2.CV_64F).var())
    if nitidez < 60:
        return False, "La fotografía parece borrosa. Mantén la cámara firme y enfocada."

    return True, "OK"


# 🔹 Ruta principal: cámara guiada + validación + resultado
@app.route('/')
def home():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🩺 Golden Detect Anemic</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; text-align: center; margin: 0; padding: 20px; background: #f7f9fc; }
            .container { max-width: 600px; margin: 0 auto; padding: 20px; background: white; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; margin-bottom: 4px; }
            .subtitle { color: #7f8c8d; font-size: 0.9em; margin-top: 0; }
            p { color: #555; line-height: 1.5; }

            .tips { text-align: left; background: #f0f8ff; border-radius: 8px; padding: 12px 16px; font-size: 0.85em; color: #34495e; margin: 12px 0; }
            .tips li { margin-bottom: 4px; }

            .camera-wrap { position: relative; width: 100%; max-width: 420px; margin: 15px auto; border-radius: 12px; overflow: hidden; background: #000; }
            video, #captureCanvas, #cropCanvas { width: 100%; display: block; }
            #cropCanvas, #captureCanvas { display: none; }

            .guide-overlay { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }
            .guide-oval {
                position: absolute;
                top: 30%; left: 20%; width: 60%; height: 35%;
                border: 3px dashed #ffffffcc;
                border-radius: 50% / 60%;
                box-shadow: 0 0 0 2000px rgba(0,0,0,0.35);
            }
            .guide-text {
                position: absolute; bottom: 6%; left: 50%; transform: translateX(-50%);
                color: white; background: rgba(0,0,0,0.55); padding: 6px 12px; border-radius: 20px;
                font-size: 0.8em; white-space: nowrap;
            }

            button { padding: 12px 24px; font-size: 16px; background: #3498db; color: white; border: none; border-radius: 6px; cursor: pointer; margin: 8px 4px; }
            button:hover { background: #2980b9; }
            button:disabled { background: #b0c4d4; cursor: not-allowed; }
            #startBtn { background: #27ae60; }
            #startBtn:hover { background: #219150; }

            #preview { max-width: 100%; max-height: 220px; margin: 15px 0; border-radius: 8px; display: none; }

            .result { margin: 20px 0; padding: 15px; border-radius: 8px; font-weight: bold; font-size: 1.1em; text-align: left; }
            .anemia { background-color: #ffebee; color: #c62828; }
            .posible { background-color: #fff8e1; color: #e65100; }
            .normal { background-color: #e8f5e8; color: #2e7d32; }
            .error { background-color: #fff3e0; color: #ef6c00; }
            .disclaimer { font-size: 0.75em; color: #888; margin-top: 8px; }

            .steps { display: flex; justify-content: space-between; margin: 15px 0; font-size: 0.75em; color: #95a5a6; }
            .steps span.active { color: #3498db; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🩺 Golden Detect Anemic</h1>
            <p class="subtitle">Evaluación preliminar de anemia mediante análisis de imagen</p>

            <ul class="tips">
                <li>💡 Usa buena iluminación, evita luz directa muy fuerte</li>
                <li>🖐️ Baja suavemente el párpado inferior con un dedo</li>
                <li>📷 Mantén la cámara firme y enfocada</li>
                <li>🚫 Evita reflejos y fotos borrosas</li>
            </ul>

            <div class="camera-wrap">
                <video id="video" autoplay playsinline></video>
                <div class="guide-overlay">
                    <div class="guide-oval"></div>
                    <div class="guide-text">Ubica la conjuntiva palpebral inferior aquí</div>
                </div>
                <canvas id="captureCanvas"></canvas>
                <canvas id="cropCanvas"></canvas>
            </div>

            <img id="preview" src="" alt="Vista previa recortada">

            <div>
                <button id="startBtn" onclick="iniciarCamara()">📷 Activar cámara</button>
                <button id="captureBtn" onclick="capturar()" disabled>🔍 Capturar y analizar</button>
                <button id="retryBtn" onclick="reiniciar()" style="display:none;">🔄 Realizar otro análisis</button>
            </div>

            <div id="result"></div>
        </div>

        <script>
            const video = document.getElementById('video');
            const captureCanvas = document.getElementById('captureCanvas');
            const cropCanvas = document.getElementById('cropCanvas');
            const preview = document.getElementById('preview');
            let stream = null;

            // Geometría FIJA del óvalo guía (coincide con el CSS .guide-oval)
            const GUIA = { xPct: 0.20, yPct: 0.30, wPct: 0.60, hPct: 0.35 };

            async function iniciarCamara() {
                try {
                    stream = await navigator.mediaDevices.getUserMedia({
                        video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 960 } }
                    });
                    video.srcObject = stream;
                    document.getElementById('startBtn').style.display = 'none';
                    document.getElementById('captureBtn').disabled = false;
                } catch (e) {
                    showResult('⚠️ No se pudo acceder a la cámara. Revisa los permisos del navegador.', 'error');
                }
            }

            function capturar() {
                // 1. Volcar el frame completo de video al canvas
                captureCanvas.width = video.videoWidth;
                captureCanvas.height = video.videoHeight;
                const ctx = captureCanvas.getContext('2d');
                ctx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

                // 2. Recortar SOLO la región del óvalo guía (geometría fija, sin IA)
                const cropX = GUIA.xPct * captureCanvas.width;
                const cropY = GUIA.yPct * captureCanvas.height;
                const cropW = GUIA.wPct * captureCanvas.width;
                const cropH = GUIA.hPct * captureCanvas.height;

                cropCanvas.width = cropW;
                cropCanvas.height = cropH;
                const cctx = cropCanvas.getContext('2d');
                cctx.drawImage(captureCanvas, cropX, cropY, cropW, cropH, 0, 0, cropW, cropH);

                preview.src = cropCanvas.toDataURL('image/jpeg', 0.92);
                preview.style.display = 'block';

                detenerCamara();
                enviarParaAnalisis();
            }

            function detenerCamara() {
                if (stream) {
                    stream.getTracks().forEach(t => t.stop());
                }
                document.getElementById('captureBtn').style.display = 'none';
            }

            async function enviarParaAnalisis() {
                const resultDiv = document.getElementById('result');
                resultDiv.innerHTML = "🔍 Verificando calidad de la imagen...";
                resultDiv.className = "result";

                cropCanvas.toBlob(async (blob) => {
                    const formData = new FormData();
                    formData.append('image', blob, 'captura.jpg');

                    resultDiv.innerHTML = "🧠 Analizando mediante inteligencia artificial...";

                    try {
                        const res = await fetch('/predict', { method: 'POST', body: formData });
                        const data = await res.json();

                        if (data.error) {
                            showResult("⚠️ " + data.error, "error");
                            document.getElementById('retryBtn').style.display = 'inline-block';
                        } else {
                            let cls = 'normal';
                            if (data.result.includes('POSIBLE')) cls = 'posible';
                            else if (data.result.includes('ANEMIA DETECTADA')) cls = 'anemia';

                            showResult(`
                                <div><strong>Resultado del análisis:</strong> ${data.result}</div>
                                <div><strong>Confianza del modelo:</strong> ${data.confidence}%</div>
                                <div><strong>Método:</strong> Análisis de imagen mediante inteligencia artificial</div>
                                <div class="disclaimer">Este resultado es únicamente una evaluación preliminar y no reemplaza una evaluación médica ni un análisis de sangre.</div>
                            `, cls);
                            document.getElementById('retryBtn').style.display = 'inline-block';
                        }
                    } catch (e) {
                        showResult("⚠️ Error de conexión. Intenta de nuevo.", "error");
                        document.getElementById('retryBtn').style.display = 'inline-block';
                    }
                }, 'image/jpeg', 0.92);
            }

            function reiniciar() {
                document.getElementById('result').innerHTML = '';
                document.getElementById('result').className = '';
                preview.style.display = 'none';
                document.getElementById('retryBtn').style.display = 'none';
                document.getElementById('captureBtn').style.display = 'inline-block';
                document.getElementById('startBtn').style.display = 'inline-block';
                document.getElementById('captureBtn').disabled = true;
            }

            function showResult(html, cls) {
                const div = document.getElementById('result');
                div.innerHTML = html;
                div.className = 'result ' + cls;
            }
        </script>
    </body>
    </html>
    """
    return render_template_string(html)


# 🔹 Ruta para predicción (con validación de calidad previa)
@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Modelo no disponible. Entrena primero con train_model.py'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No se subió ningún archivo'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ninguna imagen'}), 400

    temp_path = 'temp_image.jpg'
    file.save(temp_path)

    try:
        # --- Etapa nueva: validación de calidad (sección 11) ---
        ok, mensaje = validar_calidad_imagen(temp_path)
        if not ok:
            os.remove(temp_path)
            return jsonify({'error': mensaje}), 400

        # --- Preprocesamiento y predicción (SIN CAMBIOS respecto al modelo) ---
        img = image.load_img(temp_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]
        os.remove(temp_path)

        if prediction > 0.7:
            result = '🔴 ANEMIA DETECTADA'
        elif prediction > 0.5:
            result = '🟡 POSIBLE ANEMIA'
        else:
            result = '🟢 SIN ANEMIA'

        confidence = (prediction if prediction > 0.5 else 1 - prediction) * 100

        return jsonify({
            'result': result,
            'confidence': round(confidence, 2)
        })

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': f'Error procesando la imagen: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True)            .result { margin: 20px; padding: 15px; border-radius: 8px; font-weight: bold; font-size: 1.1em; }
            .anemia { background-color: #ffebee; color: #c62828; }
            .normal { background-color: #e8f5e8; color: #2e7d32; }
            .error { background-color: #fff3e0; color: #ef6c00; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🩺 Golden Detect Anemic</h1>
            <p>Sube una foto clara del ojo inferior para detectar signos de anemia.</p>
            <p><small>💡 Consejo: Ilumina bien el ojo y levanta el párpado inferior.</small></p>

            <div class="upload-area" onclick="document.getElementById('image').click()">
                <p>📷 Haz clic para seleccionar una imagen</p>
                <input type="file" id="image" accept="image/*" style="display: none;" onchange="previewImage(event)">
            </div>

            <img id="preview" src="" alt="Vista previa">

            <button onclick="analyze()">🔍 Analizar Imagen</button>

            <div id="result"></div>
        </div>

        <script>
            function previewImage(event) {
                const file = event.target.files[0];
                if (!file) return;
                const reader = new FileReader();
                reader.onload = function(e) {
                    const img = document.getElementById('preview');
                    img.src = e.target.result;
                    img.style.display = 'block';
                };
                reader.readAsDataURL(file);
            }

            async function analyze() {
                const fileInput = document.getElementById('image');
                const file = fileInput.files[0];
                if (!file) {
                    showResult('⚠️ Por favor, sube una imagen', 'error');
                    return;
                }

                const formData = new FormData();
                formData.append('image', file);

                const resultDiv = document.getElementById('result');
                resultDiv.innerHTML = "🔄 Analizando...";
                resultDiv.className = "result";

                try {
                    const res = await fetch('/predict', { method: 'POST', body: formData });
                    const data = await res.json();

                    if (data.error) {
                        showResult("❌ " + data.error, "error");
                    } else {
                        const cls = data.result.includes('Anemia') ? 'anemia' : 'normal';
                        showResult(`<strong>${data.result}</strong><br>Confianza: ${data.confidence}%`, cls);
                    }
                } catch (e) {
                    showResult("⚠️ Error de conexión. Intenta de nuevo.", "error");
                }
            }

            function showResult(msg, cls) {
                const div = document.getElementById('result');
                div.innerHTML = msg;
                div.className = 'result ' + cls;
            }
        </script>
    </body>
    </html>
    """
    return render_template_string(html)

# 🔹 Ruta para predicción
@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Modelo no disponible. Entrena primero con train_model.py'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No se subió ningún archivo'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ninguna imagen'}), 400

    # Guardar temporalmente
    temp_path = 'temp_image.jpg'
    file.save(temp_path)

    # Preparar imagen para predicción (224x224)
    try:
        img = image.load_img(temp_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predicción
        prediction = model.predict(img_array)[0][0]
        os.remove(temp_path)

        # Clasificación con niveles
        if prediction > 0.7:
            result = '🔴 ANEMIA DETECTADA'
        elif prediction > 0.5:
            result = '🟡 POSIBLE ANEMIA'
        else:
            result = '🟢 SIN ANEMIA'

        confidence = (prediction if prediction > 0.5 else 1 - prediction) * 100

        return jsonify({
            'result': result,
            'confidence': round(confidence, 2)
        })

    except Exception as e:
        os.remove(temp_path)
        return jsonify({'error': f'Error procesando la imagen: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True)
