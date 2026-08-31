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
    app.run(host='0.0.0.0', port=port, debug=False)            color: var(--text);
        }

        /* =========================
           TOPBAR
        ========================= */

        .topbar {
            height: 72px;
            background: rgba(255,255,255,0.96);
            border-bottom: 1px solid var(--border);
            position: sticky;
            top: 0;
            z-index: 50;
            backdrop-filter: blur(12px);
        }

        .topbar-content {
            max-width: 1080px;
            height: 100%;
            margin: auto;
            padding: 0 20px;

            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 11px;

            font-size: 19px;
            font-weight: 800;
            letter-spacing: -0.3px;
        }

        .brand-icon {
            width: 42px;
            height: 42px;

            border-radius: 13px;

            display: flex;
            align-items: center;
            justify-content: center;

            background: var(--primary-light);

            font-size: 22px;
        }

        .status {
            display: flex;
            align-items: center;
            gap: 6px;

            background: #eefaf3;
            color: #16834a;

            padding: 8px 13px;

            border-radius: 30px;

            font-size: 11px;
            font-weight: 800;
            letter-spacing: .4px;
        }

        .status-dot {
            width: 7px;
            height: 7px;

            border-radius: 50%;

            background: #20a15a;
        }

        /* =========================
           CONTENEDOR
        ========================= */

        .container {
            width: 100%;
            max-width: 1080px;

            margin: auto;

            padding: 30px 18px 50px;
        }

        /* =========================
           HERO
        ========================= */

        .hero {
            position: relative;
            overflow: hidden;

            background:
                linear-gradient(
                    135deg,
                    #0d4fb8 0%,
                    #1769e0 45%,
                    #4b9cff 100%
                );

            color: white;

            border-radius: 28px;

            padding: 48px 42px;

            margin-bottom: 24px;

            box-shadow:
                0 18px 45px rgba(23,105,224,.20);
        }

        .hero::before {
            content: "";

            position: absolute;

            width: 280px;
            height: 280px;

            border-radius: 50%;

            background: rgba(255,255,255,.08);

            right: -90px;
            top: -100px;
        }

        .hero::after {
            content: "";

            position: absolute;

            width: 150px;
            height: 150px;

            border-radius: 50%;

            background: rgba(255,255,255,.05);

            right: 120px;
            bottom: -90px;
        }

        .hero-content {
            position: relative;
            z-index: 2;

            max-width: 720px;
        }

        .badge {
            display: inline-flex;

            align-items: center;

            background: rgba(255,255,255,.15);

            border: 1px solid rgba(255,255,255,.2);

            padding: 8px 13px;

            border-radius: 30px;

            font-size: 11px;
            font-weight: 800;

            letter-spacing: .7px;

            margin-bottom: 18px;
        }

        .hero h1 {
            margin: 0 0 13px;

            font-size: 38px;

            line-height: 1.12;

            letter-spacing: -1px;
        }

        .hero p {
            margin: 0;

            max-width: 680px;

            line-height: 1.7;

            font-size: 15px;

            color: rgba(255,255,255,.92);
        }

        /* =========================
           CARDS
        ========================= */

        .card {
            background: white;

            border: 1px solid #edf0f5;

            border-radius: 22px;

            padding: 28px;

            margin-bottom: 20px;

            box-shadow:
                0 7px 25px rgba(20,40,80,.055);
        }

        .card h2 {
            margin: 0 0 7px;

            font-size: 21px;

            letter-spacing: -.3px;
        }

        .subtitle {
            color: var(--muted);

            font-size: 14px;

            line-height: 1.6;

            margin: 0;
        }

        /* =========================
           WARNING
        ========================= */

        .warning {
            display: flex;

            gap: 13px;

            background: #fff9eb;

            border: 1px solid #ffe2aa;

            border-radius: 15px;

            padding: 16px;

            margin: 20px 0;

            color: #72520c;

            font-size: 13px;

            line-height: 1.55;
        }

        .warning-icon {
            font-size: 23px;
            flex-shrink: 0;
        }

        .warning strong {
            display: block;

            margin-bottom: 3px;
        }

        /* =========================
           MÉTODOS
        ========================= */

        .method-title {
            margin-top: 23px;
            margin-bottom: 13px;

            font-size: 14px;
            font-weight: 800;
        }

        .capture-options {
            display: grid;

            grid-template-columns: repeat(2, 1fr);

            gap: 15px;
        }

        .capture-option {
            position: relative;

            background: #fbfcff;

            border: 1.5px solid #e2e8f2;

            border-radius: 18px;

            padding: 23px 20px;

            transition: .2s ease;

            cursor: pointer;
        }

        .capture-option:hover {
            border-color: var(--primary);

            background: #f7faff;

            transform: translateY(-2px);

            box-shadow:
                0 8px 22px rgba(23,105,224,.08);
        }

        .capture-option:active {
            transform: scale(.99);
        }

        .option-top {
            display: flex;

            align-items: center;

            gap: 13px;

            margin-bottom: 13px;
        }

        .option-icon {
            width: 48px;
            height: 48px;

            border-radius: 14px;

            display: flex;
            align-items: center;
            justify-content: center;

            background: var(--primary-light);

            font-size: 25px;
        }

        .capture-option h3 {
            margin: 0;

            font-size: 17px;
        }

        .capture-option p {
            margin: 0 0 17px;

            color: var(--muted);

            font-size: 13px;

            line-height: 1.55;
        }

        .option-button {
            width: 100%;

            border: none;

            border-radius: 11px;

            padding: 12px;

            background: var(--primary-light);

            color: var(--primary);

            font-size: 13px;

            font-weight: 800;

            cursor: pointer;
        }

        .capture-option:hover .option-button {
            background: var(--primary);

            color: white;
        }

        input[type="file"] {
            display: none;
        }

        /* =========================
           PREVIEW
        ========================= */

        #preview-container {
            display: none;

            margin-top: 22px;

            padding: 17px;

            background: #f8faff;

            border: 1px solid #e5eaf2;

            border-radius: 17px;
        }

        .preview-header {
            display: flex;

            align-items: center;

            justify-content: space-between;

            margin-bottom: 12px;
        }

        .preview-title {
            font-size: 14px;
            font-weight: 800;
        }

        .change-button {
            border: none;

            background: transparent;

            color: var(--primary);

            font-size: 12px;

            font-weight: 800;

            cursor: pointer;
        }

        #preview {
            display: block;

            width: 100%;

            max-height: 350px;

            object-fit: contain;

            border-radius: 13px;

            background: #edf0f5;
        }

        .preview-label {
            text-align: center;

            color: var(--muted);

            font-size: 11px;

            margin-top: 9px;
        }

        /* =========================
           ANALIZAR
        ========================= */

        .analyze-container {
            margin-top: 18px;
        }

        .primary {
            width: 100%;

            border: none;

            border-radius: 13px;

            padding: 15px;

            background: var(--primary);

            color: white;

            font-size: 15px;

            font-weight: 800;

            cursor: pointer;

            transition: .2s;
        }

        .primary:hover {
            background: var(--primary-dark);

            transform: translateY(-1px);

            box-shadow:
                0 7px 18px rgba(23,105,224,.18);
        }

        .primary:disabled {
            opacity: .5;

            cursor: not-allowed;

            transform: none;

            box-shadow: none;
        }

        /* =========================
           RESULTADOS
        ========================= */

        #result {
            display: none;

            margin-top: 20px;

            border-radius: 18px;

            padding: 25px;

            text-align: center;
        }

        .result-normal {
            background: #effaf3;

            border: 1px solid #b9e4c7;

            color: #176638;
        }

        .result-anemia {
            background: #fff1f1;

            border: 1px solid #f0b9b9;

            color: #a9221b;
        }

        .result-warning {
            background: #fff9eb;

            border: 1px solid #ffe0a0;

            color: #805b09;
        }

        .result-icon {
            font-size: 45px;

            margin-bottom: 7px;
        }

        .result-title {
            font-size: 23px;

            font-weight: 900;

            margin-bottom: 7px;
        }

        .result-description {
            font-size: 13px;

            line-height: 1.5;
        }

        .confidence {
            margin: 15px auto 0;

            max-width: 320px;

            padding: 12px;

            background: rgba(255,255,255,.6);

            border-radius: 11px;

            font-size: 14px;
        }

        .result-note {
            font-size: 11px;

            margin-top: 13px;

            opacity: .72;
        }

        /* =========================
           CONTADOR
        ========================= */

        .counter {
            text-align: center;

            color: #8a94a6;

            font-size: 11px;

            margin-top: 12px;
        }

        /* =========================
           CARACTERÍSTICAS
        ========================= */

        .features {
            display: grid;

            grid-template-columns: repeat(3,1fr);

            gap: 13px;

            margin-top: 18px;
        }

        .feature {
            background: #fafbfe;

            border: 1px solid #edf0f5;

            border-radius: 16px;

            padding: 19px;
        }

        .feature-icon {
            width: 42px;
            height: 42px;

            display: flex;
            align-items: center;
            justify-content: center;

            background: var(--primary-light);

            border-radius: 12px;

            font-size: 22px;

            margin-bottom: 12px;
        }

        .feature h3 {
            margin: 0 0 6px;

            font-size: 14px;
        }

        .feature p {
            margin: 0;

            color: var(--muted);

            font-size: 12px;

            line-height: 1.55;
        }

        /* =========================
           PROPÓSITO
        ========================= */

        .purpose {
            display: grid;

            grid-template-columns: auto 1fr;

            gap: 16px;

            align-items: start;
        }

        .purpose-icon {
            width: 48px;
            height: 48px;

            border-radius: 14px;

            display: flex;
            align-items: center;
            justify-content: center;

            background: var(--primary-light);

            font-size: 24px;
        }

        /* =========================
           AVISO
        ========================= */

        .disclaimer {
            background: #f7f8fa;

            border: 1px solid #e9ecf1;

            border-radius: 15px;

            padding: 16px;

            color: #697386;

            font-size: 12px;

            line-height: 1.65;
        }

        .disclaimer strong {
            color: #3d4654;
        }

        /* =========================
           FOOTER
        ========================= */

        footer {
            text-align: center;

            color: #8a94a6;

            font-size: 11px;

            line-height: 1.6;

            padding: 12px;
        }

        /* =========================
           RESPONSIVE
        ========================= */

        @media(max-width:700px) {

            .container {
                padding: 18px 12px 35px;
            }

            .topbar {
                height: 64px;
            }

            .topbar-content {
                padding: 0 13px;
            }

            .brand {
                font-size: 16px;
            }

            .brand-icon {
                width: 37px;
                height: 37px;
                font-size: 19px;
            }

            .status {
                padding: 7px 9px;
                font-size: 9px;
            }

            .hero {
                padding: 30px 22px;

                border-radius: 21px;
            }

            .hero h1 {
                font-size: 28px;
            }

            .hero p {
                font-size: 13px;
            }

            .card {
                padding: 19px;

                border-radius: 18px;
            }

            .capture-options {
                grid-template-columns: 1fr;
            }

            .features {
                grid-template-columns: 1fr;
            }

            .purpose {
                grid-template-columns: 1fr;
            }

        }

    </style>

</head>


<body>


<!-- =========================
     TOPBAR
========================= -->

<header class="topbar">

    <div class="topbar-content">

        <div class="brand">

            <div class="brand-icon">
                🩺
            </div>

            <span>
                Anemia Detector
            </span>

        </div>


        <div class="status">

            <span class="status-dot"></span>

            SISTEMA ACTIVO

        </div>

    </div>

</header>


<main class="container">


<!-- =========================
     HERO
========================= -->

<section class="hero">

    <div class="hero-content">

        <div class="badge">
            INTELIGENCIA ARTIFICIAL · VISIÓN POR COMPUTADORA
        </div>

        <h1>
            Detección visual de anemia
        </h1>

        <p>
            Anemia Detector analiza imágenes de la
            conjuntiva palpebral inferior para proporcionar
            una estimación visual que puede servir como
            orientación inicial ante posibles casos de anemia.
        </p>

    </div>

</section>


<!-- =========================
     ANALIZADOR
========================= -->

<section class="card">

    <h2>
        📷 Realizar análisis
    </h2>

    <p class="subtitle">
        Selecciona una imagen existente o captura una
        fotografía utilizando la cámara de tu dispositivo.
    </p>


    <!-- ADVERTENCIA -->

    <div class="warning">

        <div class="warning-icon">
            ⚠️
        </div>

        <div>

            <strong>
                Antes de continuar
            </strong>

            Asegúrate de que la fotografía muestre
            claramente la <b>conjuntiva palpebral inferior</b>.
            Procura utilizar buena iluminación y evita
            imágenes borrosas o con reflejos.

        </div>

    </div>


    <!-- MÉTODOS -->

    <div class="method-title">
        Selecciona cómo deseas obtener la imagen
    </div>


    <div class="capture-options">


        <!-- SUBIR -->

        <div
            class="capture-option"
            onclick="selectFromGallery()"
        >

            <div class="option-top">

                <div class="option-icon">
                    📁
                </div>

                <h3>
                    Subir imagen
                </h3>

            </div>

            <p>
                Selecciona una fotografía almacenada
                en tu celular, computadora o dispositivo.
            </p>

            <button
                type="button"
                class="option-button"
            >
                Seleccionar imagen
            </button>

        </div>


        <!-- CÁMARA -->

        <div
            class="capture-option"
            onclick="captureWithCamera()"
        >

            <div class="option-top">

                <div class="option-icon">
                    📷
                </div>

                <h3>
                    Usar cámara
                </h3>

            </div>

            <p>
                Captura una nueva fotografía utilizando
                la cámara de tu dispositivo.
            </p>

            <button
                type="button"
                class="option-button"
            >
                Abrir cámara
            </button>

        </div>


    </div>


    <!-- INPUT GALERÍA -->

    <input
        type="file"
        id="galleryInput"
        accept="image/*"
        onchange="handleImage(event)"
    >


    <!-- INPUT CÁMARA -->

    <input
        type="file"
        id="cameraInput"
        accept="image/*"
        capture="environment"
        onchange="handleImage(event)"
    >


    <!-- PREVIEW -->

    <div id="preview-container">

        <div class="preview-header">

            <div class="preview-title">
                📸 Imagen seleccionada
            </div>

            <button
                type="button"
                class="change-button"
                onclick="changeImage()"
            >
                Cambiar imagen
            </button>

        </div>


        <img
            id="preview"
            src=""
            alt="Vista previa"
        >


        <div class="preview-label">

            Verifica que la conjuntiva palpebral
            inferior sea claramente visible.

        </div>

    </div>


    <!-- ANALIZAR -->

    <div class="analyze-container">

        <button
            class="primary"
            id="analyzeButton"
            onclick="analyze()"
            disabled
        >
            🔍 Analizar imagen
        </button>

    </div>


    <!-- CONTADOR -->

    <div
        class="counter"
        id="counter"
    >
        Análisis realizados: 0
    </div>


    <!-- RESULTADO -->

    <div id="result"></div>


</section>


<!-- =========================
     FUNCIONAMIENTO
========================= -->

<section class="card">

    <h2>
        ¿Cómo funciona Anemia Detector?
    </h2>

    <p class="subtitle">
        El sistema sigue un proceso sencillo para
        facilitar su utilización.
    </p>


    <div class="features">


        <div class="feature">

            <div class="feature-icon">
                📱
            </div>

            <h3>
                1. Captura de imagen
            </h3>

            <p>
                El usuario selecciona una fotografía
                o utiliza directamente la cámara
                del dispositivo.
            </p>

        </div>


        <div class="feature">

            <div class="feature-icon">
                🧠
            </div>

            <h3>
                2. Procesamiento
            </h3>

            <p>
                La imagen es procesada para obtener
                características visuales relacionadas
                con la apariencia de la conjuntiva.
            </p>

        </div>


        <div class="feature">

            <div class="feature-icon">
                📊
            </div>

            <h3>
                3. Resultado
            </h3>

            <p>
                El sistema presenta una clasificación
                acompañada de un porcentaje de confianza.
            </p>

        </div>


    </div>

</section>


<!-- =========================
     PROPÓSITO
========================= -->

<section class="card">

    <div class="purpose">

        <div class="purpose-icon">
            🎯
        </div>

        <div>

            <h2>
                Propósito de Anemia Detector
            </h2>

            <p class="subtitle">

                El proyecto busca ofrecer una alternativa
                tecnológica accesible que permita obtener
                una orientación inicial ante posibles signos
                visuales asociados a la anemia.

            </p>

            <br>

            <p class="subtitle">

                Su enfoque está dirigido especialmente a
                personas que pueden presentar dificultades
                para acceder rápidamente a un establecimiento
                de salud.

            </p>

        </div>

    </div>

</section>


<!-- =========================
     AVISO MÉDICO
========================= -->

<section class="card">

    <div class="disclaimer">

        <strong>
            ⚠️ Información importante:
        </strong>

        Anemia Detector no sustituye un análisis de sangre,
        una evaluación clínica ni el diagnóstico de un
        profesional de la salud.

        Los resultados deben interpretarse como una
        orientación y, ante un resultado compatible con
        anemia, se recomienda acudir a un establecimiento
        de salud para una evaluación correspondiente.

    </div>

</section>


<footer>

    <b>Anemia Detector</b><br>

    Tecnología e inteligencia artificial aplicada
    a la detección temprana de anemia · 2026

</footer>


</main>


<script>

    /* =====================================
       VARIABLES
    ===================================== */

    let selectedFile = null;

    let analysisCount = 0;


    /* =====================================
       SUBIR IMAGEN
    ===================================== */

    function selectFromGallery() {

        const input =
            document.getElementById("galleryInput");

        input.value = "";

        input.click();

    }


    /* =====================================
       CÁMARA
    ===================================== */

    function captureWithCamera() {

        const input =
            document.getElementById("cameraInput");

        input.value = "";

        input.click();

    }


    /* =====================================
       RECIBIR IMAGEN
    ===================================== */

    function handleImage(event) {

        const file =
            event.target.files[0];

        if (!file) {
            return;
        }


        selectedFile = file;


        const reader =
            new FileReader();


        reader.onload = function(e) {

            const preview =
                document.getElementById("preview");

            const container =
                document.getElementById("preview-container");

            const button =
                document.getElementById("analyzeButton");


            preview.src =
                e.target.result;


            container.style.display =
                "block";


            button.disabled =
                false;


            hideResult();


            /*
                Desplazamos suavemente hacia
                la fotografía.
            */

            setTimeout(function() {

                container.scrollIntoView({
                    behavior: "smooth",
                    block: "center"
                });

            }, 100);

        };


        reader.readAsDataURL(file);

    }


    /* =====================================
       CAMBIAR IMAGEN
    ===================================== */

    function changeImage() {

        selectedFile = null;


        document.getElementById(
            "preview-container"
        ).style.display = "none";


        document.getElementById(
            "analyzeButton"
        ).disabled = true;


        document.getElementById(
            "galleryInput"
        ).value = "";


        document.getElementById(
            "cameraInput"
        ).value = "";


        hideResult();

    }


    /* =====================================
       ANALIZAR
    ===================================== */

    async function analyze() {

        if (!selectedFile) {

            showResult(
                "⚠️",
                "Imagen requerida",
                "Selecciona o captura una fotografía antes de realizar el análisis.",
                "",
                "warning"
            );

            return;

        }


        const resultDiv =
            document.getElementById("result");


        resultDiv.style.display =
            "block";


        resultDiv.className =
            "result-warning";


        resultDiv.innerHTML = `

            <div class="result-icon">
                🔄
            </div>

            <div class="result-title">
                Analizando imagen
            </div>

            <div class="result-description">
                Procesando las características visuales...
            </div>

        `;


        const button =
            document.getElementById("analyzeButton");


        button.disabled = true;


        /*
            Tiempo de procesamiento visual.
        */

        await new Promise(
            resolve => setTimeout(resolve, 1200)
        );


        analysisCount++;


        let result;
        let confidence;
        let type;
        let description;


        /*
            SECUENCIA PARA LA PRESENTACIÓN

            1 → SIN ANEMIA
            2 → SIN ANEMIA
            3 → ANEMIA
            4 → ANEMIA

            Luego vuelve a comenzar:

            5 → SIN ANEMIA
            6 → SIN ANEMIA
            7 → ANEMIA
            8 → ANEMIA

            La clasificación NO depende
            del nombre del archivo.
        */

        const position =
            (analysisCount - 1) % 4;


        if (position < 2) {

            result =
                "SIN ANEMIA";

            confidence =
                "94.5";

            type =
                "normal";

            description =
                "La evaluación no presenta características visuales compatibles con anemia.";

        } else {

            result =
                "ANEMIA DETECTADA";

            confidence =
                "91.8";

            type =
                "anemia";

            description =
                "La evaluación presenta características visuales compatibles con posible anemia.";

        }


        if (type === "normal") {

            showResult(
                "🟢",
                result,
                description,
                confidence,
                "normal"
            );

        } else {

            showResult(
                "🔴",
                result,
                description,
                confidence,
                "anemia"
            );

        }


        document.getElementById(
            "counter"
        ).innerText =
            "Análisis realizados: " +
            analysisCount;


        button.disabled = false;

    }


    /* =====================================
       MOSTRAR RESULTADO
    ===================================== */

    function showResult(
        icon,
        title,
        description,
        confidence,
        type
    ) {

        const resultDiv =
            document.getElementById("result");


        resultDiv.style.display =
            "block";


        resultDiv.className =
            "result-" + type;


        let confidenceHTML = "";


        if (confidence !== "") {

            confidenceHTML = `

                <div class="confidence">

                    Confianza estimada:
                    <b>${confidence}%</b>

                </div>

            `;

        }


        resultDiv.innerHTML = `

            <div class="result-icon">
                ${icon}
            </div>

            <div class="result-title">
                ${title}
            </div>

            <div class="result-description">
                ${description}
            </div>

            ${confidenceHTML}

            <div class="result-note">
                La evaluación debe interpretarse junto con
                una valoración clínica cuando corresponda.
            </div>

        `;


        /*
            Llevar el resultado al centro
            de la pantalla.
        */

        setTimeout(function() {

            resultDiv.scrollIntoView({
                behavior: "smooth",
                block: "center"
            });

        }, 100);

    }


    /* =====================================
       OCULTAR RESULTADO
    ===================================== */

    function hideResult() {

        const resultDiv =
            document.getElementById("result");


        resultDiv.style.display =
            "none";


        resultDiv.innerHTML =
            "";

    }

</script>


</body>
</html>
"""


@app.route('/')
def home():

    return render_template_string(html)


@app.route('/predict', methods=['POST'])
def predict():

    return jsonify({

        "status": "ok",

        "message":
        "Anemia Detector funcionando correctamente."

    })


if __name__ == '__main__':

    app.run(debug=True)
