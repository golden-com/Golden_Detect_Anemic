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
        <title>Golden Detect Anemic</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Work+Sans:wght@400;500;600&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg: #EEF1EE;
                --surface: #FFFFFF;
                --ink: #1C2321;
                --ink-soft: #52605B;
                --accent: #9C3B2E;
                --accent-soft: #F1DAD4;
                --teal: #1F6F6B;
                --teal-soft: #DCEEEC;
                --amber: #96690F;
                --amber-soft: #F3E6C9;
                --border: #D9DCD5;
            }
            * { box-sizing: border-box; }
            body {
                font-family: 'Work Sans', sans-serif;
                margin: 0;
                background: var(--bg);
                color: var(--ink);
                line-height: 1.5;
            }
            .shell { max-width: 980px; margin: 0 auto; padding: 48px 24px 80px; }

            header.top { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 56px; border-bottom: 1px solid var(--border); padding-bottom: 20px; }
            .brand { font-family: 'Fraunces', serif; font-size: 1.3rem; font-weight: 600; letter-spacing: -0.01em; }
            .brand span { color: var(--accent); }
            .tagline { font-size: 0.85rem; color: var(--ink-soft); }

            .hero { display: grid; grid-template-columns: 1fr 1fr; gap: 56px; align-items: start; }
            @media (max-width: 800px) { .hero { grid-template-columns: 1fr; gap: 40px; } .shell { padding: 32px 18px 60px; } }

            h1 { font-family: 'Fraunces', serif; font-weight: 500; font-size: clamp(1.8rem, 4vw, 2.5rem); line-height: 1.15; margin: 0 0 18px; letter-spacing: -0.01em; }
            .lede { color: var(--ink-soft); font-size: 1rem; max-width: 46ch; margin: 0 0 28px; }

            .tips { list-style: none; padding: 0; margin: 0 0 32px; display: flex; flex-direction: column; gap: 10px; }
            .tips li { display: flex; gap: 10px; font-size: 0.88rem; color: var(--ink-soft); align-items: flex-start; }
            .tips li::before { content: ''; width: 5px; height: 5px; border-radius: 50%; background: var(--accent); margin-top: 8px; flex-shrink: 0; }

            .disclaimer-box { border-left: 2px solid var(--border); padding-left: 14px; font-size: 0.8rem; color: var(--ink-soft); }

            /* Panel de cámara */
            .camera-card { background: var(--ink); border-radius: 4px; overflow: hidden; }
            .camera-wrap { position: relative; width: 100%; aspect-ratio: 4/3; background: #0d1210; }
            video, #captureCanvas, #cropCanvas { width: 100%; height: 100%; object-fit: cover; display: block; }
            #cropCanvas, #captureCanvas { display: none; }

            .guide-overlay { position: absolute; inset: 0; pointer-events: none; }
            .guide-oval {
                position: absolute;
                top: 30%; left: 20%; width: 60%; height: 35%;
                border: 1.5px solid #F5F0E8cc;
                border-radius: 50% / 60%;
                box-shadow: 0 0 0 1200px rgba(10,14,12,0.55);
            }
            .guide-text {
                position: absolute; bottom: 5%; left: 50%; transform: translateX(-50%);
                color: #F5F0E8; font-size: 0.78rem; text-align: center; white-space: nowrap;
                letter-spacing: 0.01em;
            }

            .camera-controls { padding: 16px 18px; display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }

            button {
                font-family: 'Work Sans', sans-serif;
                font-size: 0.88rem;
                font-weight: 500;
                padding: 11px 20px;
                border: 1px solid transparent;
                border-radius: 3px;
                cursor: pointer;
                transition: background 0.15s ease, border-color 0.15s ease;
            }
            #startBtn { background: var(--accent); color: #FBF6F2; }
            #startBtn:hover { background: #832F24; }
            #captureBtn { background: transparent; color: #F5F0E8; border-color: #F5F0E840; }
            #captureBtn:hover:not(:disabled) { border-color: #F5F0E8; }
            #captureBtn:disabled { color: #F5F0E850; cursor: not-allowed; }
            #retryBtn { background: transparent; color: var(--ink); border-color: var(--border); }
            #retryBtn:hover { border-color: var(--ink-soft); }

            #preview { display: none; }

            .result { margin-top: 18px; padding: 18px 20px; border-radius: 3px; font-size: 0.95rem; }
            .result:empty { display: none; }
            .result-label { font-size: 0.75rem; color: var(--ink-soft); margin-bottom: 2px; }
            .result-value { font-family: 'Fraunces', serif; font-size: 1.3rem; font-weight: 500; margin-bottom: 10px; }
            .anemia { background: var(--accent-soft); }
            .anemia .result-value { color: var(--accent); }
            .posible { background: var(--amber-soft); }
            .posible .result-value { color: var(--amber); }
            .normal { background: var(--teal-soft); }
            .normal .result-value { color: var(--teal); }
            .error { background: #F5F0E8; color: var(--ink-soft); border: 1px solid var(--border); }
            .disclaimer { font-size: 0.78rem; color: var(--ink-soft); margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(0,0,0,0.08); }

            /* Cómo funciona */
            .how { margin-top: 72px; padding-top: 36px; border-top: 1px solid var(--border); }
            .how h2 { font-family: 'Fraunces', serif; font-weight: 500; font-size: 1.4rem; margin-bottom: 28px; }
            .how-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 20px; }
            @media (max-width: 800px) { .how-grid { grid-template-columns: 1fr 1fr; } }
            .how-step { border-top: 2px solid var(--accent); padding-top: 12px; }
            .how-step .num { font-family: 'Fraunces', serif; font-size: 1rem; color: var(--accent); margin-bottom: 6px; }
            .how-step h3 { font-size: 0.9rem; margin: 0 0 6px; }
            .how-step p { font-size: 0.82rem; color: var(--ink-soft); margin: 0; }
        </style>
    </head>
    <body>
        <div class="shell">
            <header class="top">
                <div class="brand">Golden Detect <span>Anemic</span></div>
                <div class="tagline">Evaluación preliminar por imagen</div>
            </header>

            <div class="hero">
                <div>
                    <h1>Una fotografía de la conjuntiva, analizada por inteligencia artificial.</h1>
                    <p class="lede">Golden Detect Anemic ofrece una orientación preliminar sobre posibles signos de anemia a partir de una imagen de la conjuntiva palpebral inferior.</p>

                    <ul class="tips">
                        <li>Busca buena iluminación natural, sin luz directa muy fuerte</li>
                        <li>Baja suavemente el párpado inferior con un dedo</li>
                        <li>Mantén la cámara firme y enfocada</li>
                        <li>Evita reflejos y fotografías borrosas</li>
                    </ul>

                    <p class="disclaimer-box">Esta herramienta ofrece una evaluación preliminar. No reemplaza un diagnóstico médico ni un análisis de sangre.</p>
                </div>

                <div>
                    <div class="camera-card">
                        <div class="camera-wrap">
                            <video id="video" autoplay playsinline></video>
                            <div class="guide-overlay">
                                <div class="guide-oval"></div>
                                <div class="guide-text">Ubica la conjuntiva palpebral inferior aquí</div>
                            </div>
                            <canvas id="captureCanvas"></canvas>
                            <canvas id="cropCanvas"></canvas>
                        </div>
                        <div class="camera-controls">
                            <button id="startBtn" onclick="iniciarCamara()">Activar cámara</button>
                            <button id="captureBtn" onclick="capturar()" disabled>Capturar y analizar</button>
                            <button id="retryBtn" onclick="reiniciar()" style="display:none;">Realizar otro análisis</button>
                        </div>
                    </div>
                    <img id="preview" src="" alt="Vista previa recortada">
                    <div id="result" class="result"></div>
                </div>
            </div>

            <div class="how">
                <h2>Cómo funciona</h2>
                <div class="how-grid">
                    <div class="how-step"><div class="num">01</div><h3>Captura</h3><p>Se ubica la conjuntiva dentro del marco guía.</p></div>
                    <div class="how-step"><div class="num">02</div><h3>Verificación</h3><p>Se comprueba brillo y nitidez mínimos.</p></div>
                    <div class="how-step"><div class="num">03</div><h3>Procesamiento</h3><p>La imagen se ajusta para el modelo.</p></div>
                    <div class="how-step"><div class="num">04</div><h3>Inteligencia artificial</h3><p>El modelo analiza la imagen.</p></div>
                    <div class="how-step"><div class="num">05</div><h3>Resultado</h3><p>Se muestra la evaluación preliminar.</p></div>
                </div>
            </div>
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
                    showResult('No se pudo acceder a la cámara. Revisa los permisos del navegador.', 'error');
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
                resultDiv.innerHTML = "Verificando calidad de la imagen...";
                resultDiv.className = "result";

                cropCanvas.toBlob(async (blob) => {
                    const formData = new FormData();
                    formData.append('image', blob, 'captura.jpg');

                    resultDiv.innerHTML = "Analizando mediante inteligencia artificial...";

                    try {
                        const res = await fetch('/predict', { method: 'POST', body: formData });
                        const data = await res.json();

                        if (data.error) {
                            showResult(data.error, "error");
                            document.getElementById('retryBtn').style.display = 'inline-block';
                        } else {
                            let cls = 'normal';
                            let etiqueta = data.result;
                            if (data.result.includes('POSIBLE')) cls = 'posible';
                            else if (data.result.includes('ANEMIA DETECTADA')) cls = 'anemia';
                            etiqueta = etiqueta.replace(/^[^A-Za-zÁÉÍÓÚáéíóúÑñ]+/, '').trim();

                            showResult(`
                                <div class="result-label">Resultado del análisis</div>
                                <div class="result-value">${etiqueta}</div>
                                <div>Confianza del modelo: ${data.confidence}%</div>
                                <div>Método: análisis de imagen mediante inteligencia artificial</div>
                                <div class="disclaimer">Esta es una evaluación preliminar y no reemplaza un diagnóstico médico ni un análisis de sangre.</div>
                            `, cls);
                            document.getElementById('retryBtn').style.display = 'inline-block';
                        }
                    } catch (e) {
                        showResult("Error de conexión. Intenta de nuevo.", "error");
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
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
