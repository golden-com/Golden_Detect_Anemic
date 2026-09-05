from flask import Flask, request, jsonify, render_template_string
import os
import requests
import base64
import numpy as np
import uuid
import time
import urllib.parse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

tf.config.set_visible_devices([], 'GPU')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

try:
    model = load_model('models/final_model.h5')
    print("Modelo cargado correctamente")
except Exception as e:
    print("Error al cargar el modelo:", e)
    model = None

# ============================================================
# Sesiones temporales para el flujo QR (celular -> computadora)
# En memoria: suficiente para una demo con un solo proceso (gunicorn -w 1)
# ============================================================
SESIONES = {}
SESION_TTL_SEGUNDOS = 15 * 60


def limpiar_sesiones_viejas():
    ahora = time.time()
    vencidas = [sid for sid, s in SESIONES.items() if ahora - s['creada'] > SESION_TTL_SEGUNDOS]
    for sid in vencidas:
        SESIONES.pop(sid, None)


# ============================================================
# Imagenes de referencia (few-shot) para mejorar la verificacion
# de que la foto sea realmente una conjuntiva palpebral valida.
# Se cargan UNA VEZ al iniciar el servidor y nunca se muestran
# al publico: solo se envian a Gemini como ejemplos internos.
# ============================================================
REF_IMAGENES_URLS = [
    "https://raw.githubusercontent.com/golden-com/Golden_Detect_Anemic/main/data/test/anemia/c (1).jpg",
    "https://raw.githubusercontent.com/golden-com/Golden_Detect_Anemic/main/data/test/anemia/c (15).jpg",
]

REFERENCIAS_B64 = []


def cargar_imagenes_referencia():
    for url in REF_IMAGENES_URLS:
        try:
            url_codificada = urllib.parse.quote(url, safe=':/')
            resp = requests.get(url_codificada, timeout=10)
            resp.raise_for_status()
            REFERENCIAS_B64.append(base64.b64encode(resp.content).decode('utf-8'))
            print(f"[referencia] cargada correctamente: {url}")
        except Exception as e:
            print(f"[referencia] No se pudo cargar {url}: {e}")


cargar_imagenes_referencia()


def validar_imagen_es_ojo(ruta_imagen):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return True, "OK"

    try:
        with open(ruta_imagen, "rb") as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

        partes = []

        if REFERENCIAS_B64:
            partes.append({
                "text": "Las siguientes 1 o 2 imagenes son EJEMPLOS de referencia: fotografias validas de una conjuntiva palpebral inferior humana, bien enfocadas y encuadradas, donde el ojo ocupa una parte significativa de la imagen."
            })
            for ref_b64 in REFERENCIAS_B64:
                partes.append({
                    "inline_data": {"mime_type": "image/jpeg", "data": ref_b64}
                })

        partes.append({
            "text": (
                "Ahora evalua la SIGUIENTE fotografia (la ultima imagen adjunta), comparandola con los "
                "ejemplos anteriores. Responde UNICAMENTE 'SI' si muestra claramente un ojo humano con la "
                "conjuntiva palpebral inferior visible, ocupando una parte significativa del encuadre "
                "(similar a los ejemplos). Responde 'NO' si no es un ojo humano, si el ojo o la conjuntiva "
                "ocupan una porcion muy pequeña de la foto, si esta muy borrosa o mal iluminada, o si es "
                "una imagen no relacionada (objetos, paisajes, rostros completos sin enfoque en el ojo, etc)."
            )
        })
        partes.append({
            "inline_data": {"mime_type": "image/jpeg", "data": image_data}
        })

        payload = {"contents": [{"parts": partes}]}

        response = requests.post(url, json=payload, timeout=12)
        response.raise_for_status()
        data = response.json()
        texto = data["candidates"][0]["content"]["parts"][0]["text"].strip().upper()

        print(f"[Validacion ojo] Respuesta de Gemini: {texto[:80]}")

        if texto.startswith("NO"):
            return False, "LA IMAGEN BRINDADA NO CONTIENE LA CONJUNTIVA PALPEBRAL REQUERIDA. Por favor, toma una foto clara del ojo donde se vea la parte interna del parpado inferior (conjuntiva), ocupando gran parte de la imagen. La foto debe estar enfocada y con buena iluminacion."
        else:
            return True, "OK"

    except Exception as e:
        print(f"[Validacion ojo] Error: {e}")
        return True, "OK"


def consultar_gemini(ruta_imagen):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        with open(ruta_imagen, "rb") as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

        payload = {
            "contents": [{
                "parts": [
                    {
                        "text": "Eres un asistente de apoyo medico. Analiza esta imagen de un ojo humano, enfocandote especificamente en la conjuntiva palpebral inferior. Indica el nivel de probabilidad de anemia basado en la palidez observada. Responde UNICAMENTE con una de estas tres frases exactas: 'ALTA PROBABILIDAD', 'LEVE PROBABILIDAD', o 'BAJA PROBABILIDAD'. No uses comillas, no agregues explicaciones, ni saludos, ni otro texto."
                    },
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_data
                        }
                    }
                ]
            }]
        }

        response = requests.post(url, json=payload, timeout=8)
        response.raise_for_status()
        data = response.json()
        texto = data["candidates"][0]["content"]["parts"][0]["text"].strip().upper()

        if "ALTA" in texto:
            return "ALTA"
        elif "BAJA" in texto:
            return "BAJA"
        else:
            return "LEVE"

    except Exception as e:
        print(f"[Gemini] Error: {e}")
        return None


def analizar_imagen(temp_path):
    """
    Logica central de analisis (SIN CAMBIOS respecto a la version anterior).
    Reutilizada por /predict y por el flujo QR, para no duplicar la logica.
    Devuelve (respuesta_dict, codigo_http)
    """
    ok_ojo, mensaje_ojo = validar_imagen_es_ojo(temp_path)
    if not ok_ojo:
        return {'error': mensaje_ojo}, 400

    img = image.load_img(temp_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = float(model.predict(img_array)[0][0])

    if prediction > 0.7:
        result = "ALTA PROBABILIDAD DE ANEMIA"
    elif prediction > 0.5:
        gemini_opinion = consultar_gemini(temp_path)

        if gemini_opinion == "ALTA":
            result = "LEVE PROBABILIDAD DE ANEMIA (Se recomienda consulta medica)"
        elif gemini_opinion == "BAJA":
            result = "LEVE PROBABILIDAD DE ANEMIA"
        else:
            result = "LEVE PROBABILIDAD DE ANEMIA"
    else:
        result = "BAJA PROBABILIDAD DE ANEMIA"

    if prediction > 0.5:
        confidence = prediction * 100
    else:
        confidence = (1 - prediction) * 100

    return {'result': result, 'confidence': round(confidence, 2)}, 200


PAGINA_PRINCIPAL = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Golden Detect Anemic</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Work+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #EEF1EE; --panel: #FFFFFF; --panel-2: #F4F6F3; --text: #1C2321;
            --text-soft: #52605B; --text-faint: #7C877F; --accent: #9C3B2E;
            --accent-strong: #832F24; --accent-soft: rgba(156, 59, 46, 0.10);
            --teal: #1F6F6B; --teal-soft: rgba(31, 111, 107, 0.10);
            --amber: #96690F; --amber-soft: rgba(150, 105, 15, 0.10);
            --border: #D9DCD5;
        }
        * { box-sizing: border-box; }
        html, body { margin: 0; padding: 0; }
        body {
            font-family: 'Work Sans', sans-serif; background: var(--bg); color: var(--text);
            min-height: 100vh; overflow-x: hidden; line-height: 1.55;
        }
        body.no-scroll { overflow: hidden; }
        body::before {
            content: ''; position: fixed; inset: 0;
            background-image: linear-gradient(rgba(210,84,63,0.035) 1px, transparent 1px), linear-gradient(90deg, rgba(210,84,63,0.035) 1px, transparent 1px);
            background-size: 56px 56px; pointer-events: none; z-index: 0;
        }
        body::after {
            content: ''; position: fixed; top: -15%; left: 50%; transform: translateX(-50%);
            width: 900px; height: 700px;
            background: radial-gradient(circle, rgba(210,84,63,0.10) 0%, transparent 70%);
            pointer-events: none; z-index: 0;
        }
        .navbar {
            position: relative; z-index: 5; display: flex; align-items: center; justify-content: space-between;
            padding: 20px 40px;
        }
        .brand { display: flex; align-items: center; gap: 10px; }
        .brand img { width: 42px; height: 42px; object-fit: contain; }
        .brand-name { font-family: 'Fraunces', serif; font-weight: 600; font-size: 1.05rem; letter-spacing: -0.01em; }
        .brand-name span { color: var(--accent); }
        .nav-cta {
            font-family: 'Work Sans', sans-serif; font-weight: 600; font-size: 0.85rem;
            padding: 10px 22px; background: var(--accent); color: #FBF6F2;
            border: none; border-radius: 24px; cursor: pointer; transition: background .2s ease, transform .2s ease;
        }
        .nav-cta:hover { background: var(--accent-strong); transform: translateY(-1px); }
        .hero {
            position: relative; z-index: 1; display: flex; flex-direction: column; align-items: center; text-align: center;
            padding: 40px 20px 70px; min-height: calc(100vh - 90px); justify-content: center;
        }
        .hero-icon img { max-height: 72px; max-width: 140px; object-fit: contain; margin-bottom: 26px; }
        .hero h1 {
            font-family: 'Fraunces', serif; font-weight: 500; letter-spacing: -0.01em;
            font-size: clamp(2rem, 4.5vw, 3rem); line-height: 1.15; max-width: 680px; margin: 0 0 20px;
        }
        .hero-subtitle { font-size: 1rem; color: var(--text-soft); margin: 0 0 14px; }
        .hero-description { font-size: 0.92rem; color: var(--text-faint); max-width: 520px; margin: 0 0 36px; }
        .hero-actions { display: flex; align-items: center; gap: 22px; margin-bottom: 40px; flex-wrap: wrap; justify-content: center; }
        .btn-primary {
            font-family: 'Work Sans', sans-serif; font-weight: 600; font-size: 0.95rem;
            padding: 14px 32px; background: var(--accent); color: #FBF6F2; border: none;
            border-radius: 30px; cursor: pointer; transition: background .2s ease, transform .2s ease, box-shadow .2s ease;
        }
        .btn-primary:hover { background: var(--accent-strong); transform: translateY(-2px); box-shadow: 0 10px 30px rgba(210,84,63,0.28); }
        .btn-text { color: var(--text-soft); text-decoration: none; font-size: 0.9rem; background: none; border: none; cursor: pointer; font-family: 'Work Sans', sans-serif; }
        .btn-text:hover { color: var(--text); }
        .trust-badges { display: flex; align-items: center; gap: 26px; flex-wrap: wrap; justify-content: center; margin-bottom: 44px; }
        .trust-badge { display: flex; align-items: center; gap: 8px; font-size: 0.78rem; color: var(--text-faint); }
        .trust-badge svg { width: 14px; height: 14px; color: var(--teal); }
        .credit { text-align: center; }
        .credit-eureka { display: inline-block; font-family: 'Fraunces', serif; font-size: 0.85rem; color: var(--accent); border: 1px solid var(--border); padding: 6px 16px; border-radius: 20px; margin-bottom: 6px; }
        .credit-inst { font-size: 0.75rem; color: var(--text-faint); }
        .overlay {
            position: fixed; inset: 0; z-index: 50; background: rgba(10, 8, 6, 0.72);
            display: none; align-items: center; justify-content: center; padding: 24px;
        }
        .overlay.open { display: flex; }
        .modal {
            position: relative; background: var(--panel); border: 1px solid var(--border);
            border-radius: 18px; box-shadow: 0 30px 80px rgba(0,0,0,0.5);
            width: 100%; max-height: 90vh; overflow-y: auto;
        }
        .modal-close {
            position: absolute; top: 16px; right: 16px; width: 32px; height: 32px; border-radius: 50%;
            background: rgba(28,35,33,0.05); border: 1px solid var(--border);
            color: var(--text-soft); font-size: 1rem; cursor: pointer;
            display: flex; align-items: center; justify-content: center; z-index: 2;
        }
        .modal-close:hover { color: var(--text); background: rgba(28,35,33,0.10); }
        .modal-back {
            background: none; border: none; color: var(--text-soft); font-size: 0.82rem;
            cursor: pointer; padding: 0; margin-bottom: 18px; font-family: 'Work Sans', sans-serif;
        }
        .modal-back:hover { color: var(--text); }
        #modalWarning .modal { max-width: 480px; padding: 44px 40px; text-align: center; }
        .warning-icon { margin-bottom: 22px; }
        .warning-icon svg { width: 52px; height: 52px; }
        #modalWarning h2 { font-family: 'Fraunces', serif; font-weight: 500; font-size: 1.4rem; margin: 0 0 22px; line-height: 1.3; }
        .duration-badge {
            display: inline-flex; align-items: center; gap: 7px;
            padding: 9px 18px; border: 1px solid rgba(63,175,166,0.35); border-radius: 30px;
            font-size: 0.78rem; color: var(--teal); background: var(--teal-soft); margin-bottom: 24px;
        }
        .duration-badge svg { width: 14px; height: 14px; }
        .modal-text { font-size: 0.88rem; color: var(--text-soft); margin: 0 0 14px; }
        .modal-text strong { color: var(--text); }
        #modalHow .modal { max-width: 1000px; padding: 48px 44px; }
        @media (max-width: 600px) {
            #modalWarning .modal, #modalHow .modal, #modalDetector .modal, #modalQR .modal { padding: 28px 20px; }
        }
        #modalHow h2 { font-family: 'Fraunces', serif; font-weight: 500; font-size: 1.6rem; text-align: center; margin: 0 0 8px; }
        #modalHow .modal-sub { text-align: center; color: var(--text-faint); font-size: 0.88rem; margin: 0 0 36px; }
        .steps-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 16px; margin-bottom: 36px; }
        @media (max-width: 900px) { .steps-grid { grid-template-columns: repeat(2, 1fr); } }
        @media (max-width: 480px) { .steps-grid { grid-template-columns: 1fr; } }
        .step-card { background: var(--panel-2); border: 1px solid var(--border); border-radius: 14px; padding: 22px 18px; text-align: left; }
        .step-icon { width: 38px; height: 38px; border-radius: 10px; background: var(--accent-soft); display: flex; align-items: center; justify-content: center; margin-bottom: 14px; }
        .step-icon svg { width: 18px; height: 18px; color: var(--accent); }
        .step-card h3 { font-size: 0.85rem; font-weight: 600; margin: 0 0 8px; }
        .step-card p { font-size: 0.78rem; color: var(--text-faint); margin: 0; line-height: 1.6; }
        #modalHow .cta-row { text-align: center; }
        #modalDetector .modal { max-width: 640px; padding: 44px 40px; }
        #modalDetector h2 { font-family: 'Fraunces', serif; font-weight: 500; font-size: 1.4rem; text-align: center; margin: 0 0 6px; }
        #modalDetector .modal-sub { text-align: center; color: var(--text-faint); font-size: 0.85rem; margin: 0 0 30px; }
        .option-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
        @media (max-width: 640px) { .option-grid { grid-template-columns: 1fr; } }
        .option-card {
            background: var(--panel-2); border: 1px solid var(--border); border-radius: 14px;
            padding: 26px 20px; text-align: left; cursor: pointer;
            font-family: 'Work Sans', sans-serif; color: var(--text);
            transition: border-color .2s ease, background .2s ease;
        }
        .option-card:hover { border-color: var(--accent); background: rgba(210,84,63,0.06); }
        .option-icon { width: 40px; height: 40px; border-radius: 10px; background: var(--accent-soft); display: flex; align-items: center; justify-content: center; margin-bottom: 16px; }
        .option-icon svg { width: 20px; height: 20px; color: var(--accent); }
        .option-card h3 { font-size: 0.95rem; font-weight: 600; margin: 0 0 6px; }
        .option-card p { font-size: 0.78rem; color: var(--text-faint); margin: 0; line-height: 1.5; }
        .upload-box {
            border: 1.5px dashed var(--border); border-radius: 12px; padding: 30px 20px;
            text-align: center; cursor: pointer; transition: border-color .2s ease;
        }
        .upload-box:hover { border-color: var(--accent); }
        .upload-box p { font-size: 0.85rem; color: var(--text-soft); margin: 0 0 4px; }
        .upload-box span { font-size: 0.75rem; color: var(--text-faint); }
        .mini-disclaimer { font-size: 0.75rem; color: var(--text-faint); border-left: 2px solid var(--border); padding-left: 12px; margin: 18px 0; }
        #previewUpload { max-width: 100%; max-height: 220px; border-radius: 10px; margin: 16px auto 0; display: none; }
        #modalCamera .modal { max-width: 900px; width: 92vw; padding: 40px 36px 36px; }
        #modalCamera h2 { font-family: 'Fraunces', serif; font-weight: 500; font-size: 1.4rem; text-align: center; margin: 0 0 6px; }
        #modalCamera .modal-sub { text-align: center; color: var(--text-faint); font-size: 0.85rem; margin: 0 0 26px; }
        .detector-grid { display: grid; grid-template-columns: 0.85fr 1.15fr; gap: 32px; align-items: start; }
        @media (max-width: 760px) { .detector-grid { grid-template-columns: 1fr; } }
        .tips { list-style: none; padding: 0; margin: 0 0 22px; display: flex; flex-direction: column; gap: 9px; }
        .tips li { display: flex; gap: 9px; font-size: 0.82rem; color: var(--text-soft); align-items: flex-start; }
        .tips li::before { content: ''; width: 5px; height: 5px; border-radius: 50%; background: var(--accent); margin-top: 7px; flex-shrink: 0; }
        .camera-card { background: #0D0B09; border-radius: 12px; overflow: hidden; border: 1px solid var(--border); }
        .camera-wrap { position: relative; width: 100%; aspect-ratio: 4/3; background: #0a0908; }
        video, #captureCanvas, #cropCanvas { width: 100%; height: 100%; object-fit: cover; display: block; }
        #cropCanvas, #captureCanvas { display: none; }
        .guide-overlay { position: absolute; inset: 0; pointer-events: none; }
        .guide-oval {
            position: absolute; top: 30%; left: 20%; width: 60%; height: 35%;
            border: 1.5px solid rgba(245,240,232,0.8); border-radius: 50% / 60%;
            box-shadow: 0 0 0 1200px rgba(8,6,5,0.55);
        }
        .guide-text {
            position: absolute; bottom: 5%; left: 50%; transform: translateX(-50%);
            color: var(--text); font-size: 0.76rem; text-align: center; white-space: nowrap; letter-spacing: 0.01em;
        }
        .camera-controls { padding: 14px 16px; display: flex; gap: 10px; flex-wrap: wrap; }
        button.ctrl {
            font-family: 'Work Sans', sans-serif; font-size: 0.84rem; font-weight: 500;
            padding: 10px 18px; border: 1px solid transparent; border-radius: 24px; cursor: pointer;
            transition: background .15s ease, border-color .15s ease;
        }
        .ctrl.solid { background: var(--accent); color: #FBF6F2; }
        .ctrl.solid:hover { background: var(--accent-strong); }
        .ctrl.outline { background: transparent; color: var(--text); border-color: rgba(245,240,232,0.25); }
        .ctrl.outline:hover:not(:disabled) { border-color: var(--text); }
        .ctrl.outline:disabled { color: rgba(245,240,232,0.3); cursor: not-allowed; }
        .ctrl.ghost { background: transparent; color: var(--text-soft); border-color: var(--border); }
        .ctrl.ghost:hover { border-color: var(--text-soft); }
        .result { margin-top: 16px; padding: 18px 20px; border-radius: 10px; font-size: 0.9rem; display: none; }
        .result.show { display: block; }
        .result-label { font-size: 0.72rem; color: var(--text-faint); margin-bottom: 3px; }
        .result-value { font-family: 'Fraunces', serif; font-size: 1.25rem; font-weight: 500; margin-bottom: 10px; }
        .confidence-row { font-size: 0.82rem; color: var(--text-soft); margin-bottom: 4px; display: flex; justify-content: space-between; }
        .confidence-bar { height: 8px; background: rgba(0,0,0,0.08); border-radius: 5px; overflow: hidden; margin-bottom: 10px; }
        .confidence-fill { height: 100%; border-radius: 5px; transition: width .5s ease; }
        .result.anemia { background: var(--accent-soft); }
        .result.anemia .result-value { color: var(--accent-strong); }
        .result.anemia .confidence-fill { background: var(--accent-strong); }
        .result.posible { background: var(--amber-soft); }
        .result.posible .result-value { color: var(--amber); }
        .result.posible .confidence-fill { background: var(--amber); }
        .result.normal { background: var(--teal-soft); }
        .result.normal .result-value { color: var(--teal); }
        .result.normal .confidence-fill { background: var(--teal); }
        .result.error { background: var(--panel-2); border: 1px solid var(--border); color: var(--text-soft); }
        .result .disclaimer { font-size: 0.74rem; color: var(--text-faint); margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--border); }

        /* Modal QR */
        #modalQR .modal { max-width: 460px; padding: 40px 36px; text-align: center; }
        .qr-box { background: #FFFFFF; border: 1px solid var(--border); border-radius: 14px; padding: 20px; display: inline-block; margin: 10px 0 18px; }
        #qrCanvasWrap { width: 200px; height: 200px; display: flex; align-items: center; justify-content: center; }
        .qr-estado { font-size: 0.85rem; color: var(--text-soft); margin-bottom: 6px; }
        .qr-spinner {
            width: 16px; height: 16px; border: 2px solid var(--border); border-top-color: var(--accent);
            border-radius: 50%; display: inline-block; margin-right: 8px; vertical-align: middle;
            animation: girar 0.8s linear infinite;
        }
        @keyframes girar { to { transform: rotate(360deg); } }
        .qr-link { font-size: 0.72rem; color: var(--text-faint); word-break: break-all; margin-top: 10px; }
    </style>
</head>
<body>
    <nav class="navbar">
        <div class="brand">
            <img src="{{ url_for('static', filename='logo.png') }}" alt="Logo Golden Detect Anemic">
            <div class="brand-name">Golden Detect <span>Anemic</span></div>
        </div>
        <button class="nav-cta" onclick="openModal('modalWarning')">Iniciar analisis</button>
    </nav>

    <section class="hero">
        <div class="hero-icon">
            <img src="{{ url_for('static', filename='logo.png') }}" alt="Logo Golden Detect Anemic">
        </div>
        <h1>Evaluacion preliminar de anemia mediante inteligencia artificial</h1>
        <p class="hero-subtitle">Una fotografia de la conjuntiva palpebral inferior, analizada en segundos</p>
        <p class="hero-description">
            Golden Detect Anemic utiliza un modelo de inteligencia artificial entrenado para
            ofrecer una orientacion preliminar sobre posibles signos de anemia. No reemplaza
            un examen medico ni un analisis de sangre.
        </p>
        <div class="hero-actions">
            <button class="btn-primary" onclick="openModal('modalWarning')">Iniciar mi analisis</button>
            <button class="btn-text" onclick="openModal('modalHow')">Descubre como funciona &darr;</button>
        </div>
        <div class="trust-badges">
            <div class="trust-badge">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
                Evaluacion preliminar, no diagnostica
            </div>
            <div class="trust-badge">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                Resultado en segundos
            </div>
            <div class="trust-badge">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 12l2 2 4-4"/><circle cx="12" cy="12" r="10"/></svg>
                Sin costo, uso educativo
            </div>
        </div>
        <div class="credit">
            <div class="credit-eureka">Proyecto de solucion tecnologica</div>
            <div class="credit-inst">Institucion Educativa "Victor Manuel Maurtua" &middot; Parcona, Ica</div>
        </div>
    </section>

    <div class="overlay" id="modalWarning">
        <div class="modal">
            <div class="warning-icon">
                <svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M32 8L4 56H60L32 8Z" fill="#D9A54B" opacity="0.9"/>
                    <path d="M32 8L4 56H60L32 8Z" stroke="#D9A54B" stroke-width="2" fill="none"/>
                    <text x="32" y="46" text-anchor="middle" font-size="28" font-weight="800" fill="#17120F" font-family="Work Sans, sans-serif">!</text>
                </svg>
            </div>
            <h2>Advertencia medica previa</h2>
            <div class="duration-badge">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                Captura y analisis: menos de un minuto
            </div>
            <p class="modal-text">Golden Detect Anemic es una herramienta <strong>unicamente informativa.</strong></p>
            <p class="modal-text"><strong>Los resultados generados por la IA NO CONSTITUYEN UN DIAGNOSTICO MEDICO.</strong></p>
            <p class="modal-text">En caso de duda o sospecha de anemia, <strong>consulte a un profesional de la salud.</strong></p>
            <button class="btn-primary" onclick="closeModal('modalWarning'); openModal('modalHow');">Entiendo, continuar</button>
        </div>
    </div>

    <div class="overlay" id="modalHow">
        <div class="modal">
            <button class="modal-close" onclick="closeModal('modalHow')">&times;</button>
            <h2>Como funciona el analisis</h2>
            <p class="modal-sub">Cinco pasos, sin cita previa, disponible cuando lo necesites.</p>
            <div class="steps-grid">
                <div class="step-card">
                    <div class="step-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="6" width="18" height="14" rx="2"/><circle cx="12" cy="13" r="3"/></svg></div>
                    <h3>1. Captura</h3>
                    <p>Subes una foto, usas la camara con guia, o escaneas un QR con tu celular.</p>
                </div>
                <div class="step-card">
                    <div class="step-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 12l2 2 4-4"/><circle cx="12" cy="12" r="10"/></svg></div>
                    <h3>2. Verificacion</h3>
                    <p>El modelo comprueba que sea un ojo con conjuntiva visible.</p>
                </div>
                <div class="step-card">
                    <div class="step-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="4" y="4" width="16" height="16" rx="2"/></svg></div>
                    <h3>3. Procesamiento</h3>
                    <p>La imagen se ajusta al formato que usa el modelo.</p>
                </div>
                <div class="step-card">
                    <div class="step-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M12 2v4M12 18v4M2 12h4M18 12h4"/></svg></div>
                    <h3>4. Inteligencia artificial</h3>
                    <p>El modelo entrenado analiza la imagen recibida.</p>
                </div>
                <div class="step-card">
                    <div class="step-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 3v18h18"/><path d="M7 14l4-4 4 4 5-5"/></svg></div>
                    <h3>5. Resultado</h3>
                    <p>Se muestra la evaluacion preliminar y su confianza.</p>
                </div>
            </div>
            <div class="cta-row">
                <button class="btn-primary" onclick="closeModal('modalHow'); resetDetectorView(); openModal('modalDetector');">Iniciar mi analisis ahora</button>
            </div>
        </div>
    </div>

    <div class="overlay" id="modalDetector">
        <div class="modal">
            <button class="modal-close" onclick="closeModal('modalDetector')">&times;</button>
            <div id="opcionesSeccion">
                <h2>Analisis por imagen</h2>
                <p class="modal-sub">Elige como quieres proporcionar la fotografia.</p>
                <div class="option-grid">
                    <button class="option-card" onclick="mostrarSubida()">
                        <div class="option-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg></div>
                        <h3>Subir archivo</h3>
                        <p>Selecciona una foto ya tomada desde tu dispositivo.</p>
                    </button>
                    <button class="option-card" onclick="abrirCamara()">
                        <div class="option-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="6" width="18" height="14" rx="2"/><circle cx="12" cy="13" r="3"/></svg></div>
                        <h3>Tomar foto</h3>
                        <p>Usa la camara de este dispositivo con guia visual.</p>
                    </button>
                    <button class="option-card" onclick="abrirQR()">
                        <div class="option-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><path d="M14 14h3v3h-3zM18 14h3v3M14 18h3v3M21 21h-3"/></svg></div>
                        <h3>Usar mi celular</h3>
                        <p>Escanea un QR y toma la foto desde tu telefono.</p>
                    </button>
                </div>
            </div>
            <div id="uploadSeccion" style="display:none;">
                <button class="modal-back" onclick="volverOpciones()">&larr; Volver</button>
                <h2>Subir archivo</h2>
                <p class="modal-sub">Elige una foto donde se vea con claridad la conjuntiva palpebral inferior.</p>
                <ul class="tips">
                    <li>Usa una foto con buena iluminacion natural</li>
                    <li>Evita imagenes borrosas o con reflejos</li>
                    <li>La conjuntiva debe verse claramente, sin recortes</li>
                </ul>
                <div class="upload-box" onclick="document.getElementById('fileInput').click()">
                    <p>Haz clic para seleccionar una imagen</p>
                    <span>JPG o PNG</span>
                </div>
                <input type="file" id="fileInput" accept="image/*" onchange="onFileSelected(event)" hidden>
                <img id="previewUpload" src="" alt="Vista previa">
                <p class="mini-disclaimer">Esta es una evaluacion preliminar y no reemplaza un diagnostico medico ni un analisis de sangre.</p>
                <div id="resultUpload" class="result"></div>
            </div>
        </div>
    </div>

    <div class="overlay" id="modalCamera">
        <div class="modal">
            <button class="modal-close" onclick="cerrarCamara()">&times;</button>
            <button class="modal-back" onclick="cerrarCamara()">&larr; Volver</button>
            <h2>Captura guiada</h2>
            <p class="modal-sub">Activa la camara y ubica la conjuntiva dentro del marco guia.</p>
            <div class="detector-grid">
                <div>
                    <ul class="tips">
                        <li>Busca buena iluminacion natural, sin luz directa muy fuerte</li>
                        <li>Baja suavemente el parpado inferior con un dedo</li>
                        <li>Manten la camara firme y enfocada</li>
                        <li>Evita reflejos y fotografias borrosas</li>
                    </ul>
                    <p class="mini-disclaimer">Esta es una evaluacion preliminar y no reemplaza un diagnostico medico ni un analisis de sangre.</p>
                </div>
                <div>
                    <div class="camera-card">
                        <div class="camera-wrap">
                            <video id="video" autoplay playsinline></video>
                            <div class="guide-overlay">
                                <div class="guide-oval"></div>
                                <div class="guide-text">Ubica la conjuntiva palpebral inferior aqui</div>
                            </div>
                            <canvas id="captureCanvas"></canvas>
                            <canvas id="cropCanvas"></canvas>
                        </div>
                        <div class="camera-controls">
                            <button class="ctrl solid" id="startBtn" onclick="iniciarCamara()">Activar camara</button>
                            <button class="ctrl outline" id="captureBtn" onclick="capturar()" disabled>Capturar y analizar</button>
                            <button class="ctrl ghost" id="retryBtn" onclick="reiniciarCamara()" style="display:none;">Realizar otro analisis</button>
                        </div>
                    </div>
                    <div id="resultCamera" class="result"></div>
                </div>
            </div>
        </div>
    </div>

    <div class="overlay" id="modalQR">
        <div class="modal">
            <button class="modal-close" onclick="cerrarQR()">&times;</button>
            <h2>Usa tu celular</h2>
            <p class="modal-sub">Escanea este codigo con tu telefono para tomar la foto desde ahi.</p>
            <div class="qr-box"><div id="qrCanvasWrap"></div></div>
            <div class="qr-estado" id="qrEstadoTexto"><span class="qr-spinner"></span>Esperando captura desde el celular...</div>
            <div class="qr-link" id="qrLinkTexto"></div>
            <div id="resultQR" class="result"></div>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/qrcodejs/1.0.0/qrcode.min.js"></script>
    <script>
        function openModal(id) {
            document.getElementById(id).classList.add('open');
            document.body.classList.add('no-scroll');
        }
        function closeModal(id) {
            document.getElementById(id).classList.remove('open');
            if (!document.querySelector('.overlay.open')) {
                document.body.classList.remove('no-scroll');
            }
        }
        function resetDetectorView() {
            document.getElementById('opcionesSeccion').style.display = 'block';
            document.getElementById('uploadSeccion').style.display = 'none';
            const resultUpload = document.getElementById('resultUpload');
            resultUpload.innerHTML = '';
            resultUpload.className = 'result';
            document.getElementById('previewUpload').style.display = 'none';
            document.getElementById('fileInput').value = '';
        }
        function mostrarSubida() {
            document.getElementById('opcionesSeccion').style.display = 'none';
            document.getElementById('uploadSeccion').style.display = 'block';
        }
        function volverOpciones() { resetDetectorView(); }
        function abrirCamara() {
            closeModal('modalDetector');
            reiniciarCamara();
            openModal('modalCamera');
        }
        function cerrarCamara() {
            detenerCamara();
            closeModal('modalCamera');
            resetDetectorView();
            openModal('modalDetector');
        }
        function onFileSelected(event) {
            const file = event.target.files[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = document.getElementById('previewUpload');
                img.src = e.target.result;
                img.style.display = 'block';
            };
            reader.readAsDataURL(file);
            enviarImagen(file, 'resultUpload');
        }
        const video = document.getElementById('video');
        const captureCanvas = document.getElementById('captureCanvas');
        const cropCanvas = document.getElementById('cropCanvas');
        let stream = null;
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
                showResult('No se pudo acceder a la camara. Revisa los permisos del navegador.', 'error', 'resultCamera');
            }
        }
        function capturar() {
            captureCanvas.width = video.videoWidth;
            captureCanvas.height = video.videoHeight;
            const ctx = captureCanvas.getContext('2d');
            ctx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
            const cropX = GUIA.xPct * captureCanvas.width;
            const cropY = GUIA.yPct * captureCanvas.height;
            const cropW = GUIA.wPct * captureCanvas.width;
            const cropH = GUIA.hPct * captureCanvas.height;
            cropCanvas.width = cropW;
            cropCanvas.height = cropH;
            const cctx = cropCanvas.getContext('2d');
            cctx.drawImage(captureCanvas, cropX, cropY, cropW, cropH, 0, 0, cropW, cropH);
            detenerCamara();
            cropCanvas.toBlob((blob) => enviarImagen(blob, 'resultCamera'), 'image/jpeg', 0.92);
            document.getElementById('retryBtn').style.display = 'inline-block';
        }
        function detenerCamara() {
            if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
            document.getElementById('captureBtn').style.display = 'none';
        }
        function reiniciarCamara() {
            const resultDiv = document.getElementById('resultCamera');
            resultDiv.innerHTML = '';
            resultDiv.className = 'result';
            document.getElementById('retryBtn').style.display = 'none';
            document.getElementById('captureBtn').style.display = 'inline-block';
            document.getElementById('startBtn').style.display = 'inline-block';
            document.getElementById('captureBtn').disabled = true;
        }

        function construirBarraConfianza(confidence, cls) {
            const pct = Math.max(0, Math.min(100, confidence));
            return `
                <div class="confidence-row"><span>Confianza del modelo</span><span>${pct}%</span></div>
                <div class="confidence-bar"><div class="confidence-fill" style="width:${pct}%"></div></div>
            `;
        }

        async function enviarImagen(blob, resultId) {
            const resultDiv = document.getElementById(resultId);
            resultDiv.className = 'result show';
            resultDiv.innerHTML = "Verificando que la imagen sea valida...";
            const formData = new FormData();
            formData.append('image', blob, 'imagen.jpg');
            resultDiv.innerHTML = "Analizando mediante inteligencia artificial...";
            try {
                const res = await fetch('/predict', { method: 'POST', body: formData });
                const data = await res.json();
                if (data.error) {
                    showResult(data.error, "error", resultId);
                } else {
                    mostrarResultadoCompleto(data, resultId);
                }
            } catch (e) {
                showResult("Error de conexion. Intenta de nuevo.", "error", resultId);
            }
        }

        function mostrarResultadoCompleto(data, resultId) {
            let cls = 'normal';
            let etiqueta = data.result;
            if (data.result.includes('ALTA')) cls = 'anemia';
            else if (data.result.includes('LEVE')) cls = 'posible';

            showResult(`
                <div class="result-label">Resultado del analisis</div>
                <div class="result-value">${etiqueta}</div>
                ${construirBarraConfianza(data.confidence, cls)}
                <div class="disclaimer">Esta es una evaluacion preliminar y no reemplaza un diagnostico medico ni un analisis de sangre.</div>
            `, cls, resultId);
        }

        function showResult(html, cls, resultId) {
            const div = document.getElementById(resultId);
            div.innerHTML = html;
            div.className = 'result show ' + cls;
        }

        /* ---- Flujo QR (celular -> computadora) ---- */
        let qrSesionId = null;
        let qrIntervalo = null;

        async function abrirQR() {
            closeModal('modalDetector');
            document.getElementById('qrCanvasWrap').innerHTML = '';
            document.getElementById('resultQR').innerHTML = '';
            document.getElementById('resultQR').className = 'result';
            document.getElementById('qrEstadoTexto').innerHTML = '<span class="qr-spinner"></span>Generando codigo...';
            document.getElementById('qrEstadoTexto').style.display = 'block';
            document.getElementById('qrLinkTexto').textContent = '';
            openModal('modalQR');

            try {
                const res = await fetch('/api/sesion/nueva', { method: 'POST' });
                const data = await res.json();
                qrSesionId = data.sesion_id;

                const urlMovil = `${window.location.origin}/m/${qrSesionId}`;
                new QRCode(document.getElementById('qrCanvasWrap'), {
                    text: urlMovil,
                    width: 180,
                    height: 180
                });
                document.getElementById('qrLinkTexto').textContent = urlMovil;
                document.getElementById('qrEstadoTexto').innerHTML = '<span class="qr-spinner"></span>Esperando captura desde el celular...';

                qrIntervalo = setInterval(consultarEstadoSesion, 2000);
            } catch (e) {
                document.getElementById('qrEstadoTexto').textContent = 'No se pudo generar el codigo. Intenta de nuevo.';
            }
        }

        async function consultarEstadoSesion() {
            if (!qrSesionId) return;
            try {
                const res = await fetch(`/api/sesion/${qrSesionId}/estado`);
                const data = await res.json();

                if (data.estado === 'listo') {
                    clearInterval(qrIntervalo);
                    document.getElementById('qrEstadoTexto').style.display = 'none';

                    if (data.imagen) {
                        document.getElementById('qrCanvasWrap').innerHTML =
                            `<img src="data:image/jpeg;base64,${data.imagen}" alt="Foto capturada" style="width:100%; height:100%; object-fit:cover; border-radius:8px;">`;
                    }

                    if (data.error) {
                        showResult(data.error, 'error', 'resultQR');
                    } else {
                        mostrarResultadoCompleto(data, 'resultQR');
                    }
                }
            } catch (e) { /* se reintenta en el siguiente ciclo */ }
        }

        function cerrarQR() {
            if (qrIntervalo) clearInterval(qrIntervalo);
            qrSesionId = null;
            closeModal('modalQR');
            openModal('modalDetector');
        }
    </script>
</body>
</html>
"""


PAGINA_MOVIL = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Golden Detect Anemic - Captura</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500&family=Work+Sans:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #EEF1EE; --text: #1C2321; --text-soft: #52605B; --text-faint: #7C877F;
            --accent: #9C3B2E; --accent-strong: #832F24; --accent-soft: rgba(156,59,46,0.10);
            --teal: #1F6F6B; --teal-soft: rgba(31,111,107,0.10);
            --amber: #96690F; --amber-soft: rgba(150,105,15,0.10);
            --border: #D9DCD5; --panel-2: #F4F6F3;
        }
        * { box-sizing: border-box; }
        body { font-family: 'Work Sans', sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 20px; }
        h1 { font-family: 'Fraunces', serif; font-weight: 500; font-size: 1.3rem; margin: 0 0 6px; }
        p.sub { color: var(--text-soft); font-size: 0.85rem; margin: 0 0 18px; }
        .tips { list-style: none; padding: 0; margin: 0 0 16px; }
        .tips li { font-size: 0.8rem; color: var(--text-soft); margin-bottom: 6px; padding-left: 14px; position: relative; }
        .tips li::before { content: ''; width: 5px; height: 5px; border-radius: 50%; background: var(--accent); position: absolute; left: 0; top: 6px; }
        .camera-card { background: #0D0B09; border-radius: 12px; overflow: hidden; }
        .camera-wrap { position: relative; width: 100%; aspect-ratio: 3/4; background: #0a0908; }
        video, #captureCanvas, #cropCanvas { width: 100%; height: 100%; object-fit: cover; display: block; }
        #cropCanvas, #captureCanvas { display: none; }
        .guide-overlay { position: absolute; inset: 0; pointer-events: none; }
        .guide-oval {
            position: absolute; top: 32%; left: 15%; width: 70%; height: 30%;
            border: 1.5px solid rgba(245,240,232,0.8); border-radius: 50% / 60%;
            box-shadow: 0 0 0 1200px rgba(8,6,5,0.55);
        }
        .guide-text { position: absolute; bottom: 6%; left: 50%; transform: translateX(-50%); color: #F5F0E8; font-size: 0.78rem; text-align: center; padding: 0 16px; }
        .camera-controls { padding: 14px; display: flex; flex-direction: column; gap: 10px; }
        button {
            font-family: 'Work Sans', sans-serif; font-size: 0.9rem; font-weight: 600;
            padding: 13px; border-radius: 24px; border: 1px solid transparent; cursor: pointer;
        }
        .solid { background: var(--accent); color: #FBF6F2; }
        .outline { background: transparent; color: #F5F0E8; border-color: rgba(245,240,232,0.3); }
        .outline:disabled { color: rgba(245,240,232,0.3); }
        .ghost { background: var(--panel-2); color: var(--text); border-color: var(--border); }
        .result { margin-top: 16px; padding: 16px 18px; border-radius: 10px; font-size: 0.88rem; display: none; }
        .result.show { display: block; }
        .result-value { font-family: 'Fraunces', serif; font-size: 1.15rem; font-weight: 500; margin-bottom: 8px; }
        .confidence-row { font-size: 0.8rem; color: var(--text-soft); margin-bottom: 4px; display: flex; justify-content: space-between; }
        .confidence-bar { height: 8px; background: rgba(0,0,0,0.08); border-radius: 5px; overflow: hidden; margin-bottom: 10px; }
        .confidence-fill { height: 100%; border-radius: 5px; }
        .result.anemia { background: var(--accent-soft); } .result.anemia .confidence-fill { background: var(--accent-strong); }
        .result.posible { background: var(--amber-soft); } .result.posible .confidence-fill { background: var(--amber); }
        .result.normal { background: var(--teal-soft); } .result.normal .confidence-fill { background: var(--teal); }
        .result.error { background: var(--panel-2); border: 1px solid var(--border); color: var(--text-soft); }
        .aviso-listo { font-size: 0.82rem; color: var(--teal); margin-top: 10px; text-align: center; }
    </style>
</head>
<body>
    <h1>Captura guiada</h1>
    <p class="sub">Ubica la conjuntiva dentro del marco. El resultado aparecera en la pantalla de la computadora.</p>
    <ul class="tips">
        <li>Busca buena iluminacion natural</li>
        <li>Baja suavemente el parpado inferior</li>
        <li>Manten el telefono firme</li>
    </ul>
    <div class="camera-card">
        <div class="camera-wrap">
            <video id="video" autoplay playsinline></video>
            <div class="guide-overlay">
                <div class="guide-oval"></div>
                <div class="guide-text">Ubica la conjuntiva aqui</div>
            </div>
            <canvas id="captureCanvas"></canvas>
            <canvas id="cropCanvas"></canvas>
        </div>
        <div class="camera-controls">
            <button class="solid" id="startBtn" onclick="iniciarCamara()">Activar camara</button>
            <button class="outline" id="captureBtn" onclick="capturar()" disabled>Capturar y enviar</button>
            <button class="ghost" id="retryBtn" onclick="location.reload()" style="display:none;">Tomar otra foto</button>
        </div>
    </div>
    <div id="result" class="result"></div>

    <script>
        const SESION_ID = "{{ sesion_id }}";
        const video = document.getElementById('video');
        const captureCanvas = document.getElementById('captureCanvas');
        const cropCanvas = document.getElementById('cropCanvas');
        let stream = null;
        const GUIA = { xPct: 0.15, yPct: 0.32, wPct: 0.70, hPct: 0.30 };

        async function iniciarCamara() {
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 960 } }
                });
                video.srcObject = stream;
                document.getElementById('startBtn').style.display = 'none';
                document.getElementById('captureBtn').disabled = false;
            } catch (e) {
                mostrar('No se pudo acceder a la camara. Revisa los permisos.', 'error');
            }
        }

        function capturar() {
            captureCanvas.width = video.videoWidth;
            captureCanvas.height = video.videoHeight;
            captureCanvas.getContext('2d').drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

            const cropX = GUIA.xPct * captureCanvas.width;
            const cropY = GUIA.yPct * captureCanvas.height;
            const cropW = GUIA.wPct * captureCanvas.width;
            const cropH = GUIA.hPct * captureCanvas.height;

            cropCanvas.width = cropW;
            cropCanvas.height = cropH;
            cropCanvas.getContext('2d').drawImage(captureCanvas, cropX, cropY, cropW, cropH, 0, 0, cropW, cropH);

            if (stream) { stream.getTracks().forEach(t => t.stop()); }
            document.getElementById('captureBtn').style.display = 'none';

            cropCanvas.toBlob((blob) => enviar(blob), 'image/jpeg', 0.92);
        }

        async function enviar(blob) {
            mostrarCargando("Enviando fotografia...");
            const formData = new FormData();
            formData.append('image', blob, 'imagen.jpg');
            try {
                const res = await fetch(`/api/sesion/${SESION_ID}/capturar`, { method: 'POST', body: formData });
                const data = await res.json();
                if (data.error) {
                    mostrar(`<div class="result-value">No se pudo procesar</div><p>${data.error}</p>`, 'error');
                    document.getElementById('retryBtn').style.display = 'block';
                } else {
                    mostrar(`
                        <div class="result-value">Captura exitosa</div>
                        <p>El resultado ya aparecio en la pantalla de la computadora.</p>
                    `, 'normal');
                    /* Sin boton de reintentar: la captura ya se envio y la sesion se cierra desde la computadora */
                }
            } catch (e) {
                mostrar('<div class="result-value">Error de conexion</div><p>Intenta de nuevo.</p>', 'error');
                document.getElementById('retryBtn').style.display = 'block';
            }
        }

        function mostrarCargando(texto) {
            const div = document.getElementById('result');
            div.innerHTML = texto;
            div.className = 'result show';
        }
        function mostrar(html, cls) {
            const div = document.getElementById('result');
            div.innerHTML = html;
            div.className = 'result show ' + cls;
        }
    </script>
</body>
</html>
"""


@app.route('/')
def home():
    return render_template_string(PAGINA_PRINCIPAL)


@app.route('/m/<sesion_id>')
def pagina_movil(sesion_id):
    limpiar_sesiones_viejas()
    if sesion_id not in SESIONES:
        return "Este enlace expiro o no es valido. Genera un nuevo codigo QR desde la computadora.", 404
    return render_template_string(PAGINA_MOVIL, sesion_id=sesion_id)


@app.route('/api/sesion/nueva', methods=['POST'])
def nueva_sesion():
    limpiar_sesiones_viejas()
    sesion_id = uuid.uuid4().hex[:10]
    SESIONES[sesion_id] = {'estado': 'esperando', 'creada': time.time()}
    return jsonify({'sesion_id': sesion_id})


@app.route('/api/sesion/<sesion_id>/estado')
def estado_sesion(sesion_id):
    sesion = SESIONES.get(sesion_id)
    if not sesion:
        return jsonify({'estado': 'listo', 'error': 'La sesion expiro o no existe.'})
    return jsonify(sesion)


@app.route('/api/sesion/<sesion_id>/capturar', methods=['POST'])
def capturar_sesion(sesion_id):
    if sesion_id not in SESIONES:
        return jsonify({'error': 'Sesion invalida o expirada'}), 404

    if not model:
        SESIONES[sesion_id] = {'estado': 'listo', 'creada': time.time(), 'error': 'Modelo no disponible'}
        return jsonify({'error': 'Modelo no disponible'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No se subio ningun archivo'}), 400

    file = request.files['image']
    temp_path = f'temp_{sesion_id}.jpg'
    file.save(temp_path)

    # Guardamos una copia de la foto en base64 para mostrarla en la PC
    # (reemplaza al QR en cuanto llega el resultado).
    try:
        with open(temp_path, 'rb') as f:
            imagen_b64 = base64.b64encode(f.read()).decode('utf-8')
    except Exception:
        imagen_b64 = None

    try:
        respuesta, codigo = analizar_imagen(temp_path)
        os.remove(temp_path)

        if codigo == 200:
            SESIONES[sesion_id] = {'estado': 'listo', 'creada': time.time(), 'imagen': imagen_b64, **respuesta}
        else:
            SESIONES[sesion_id] = {'estado': 'listo', 'creada': time.time(), 'imagen': imagen_b64, 'error': respuesta.get('error')}

        return jsonify(respuesta), codigo

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        error_msg = f'Error procesando la imagen: {str(e)}'
        SESIONES[sesion_id] = {'estado': 'listo', 'creada': time.time(), 'imagen': imagen_b64, 'error': error_msg}
        return jsonify({'error': error_msg}), 500


@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Modelo no disponible'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No se subio ningun archivo'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No se selecciono ninguna imagen'}), 400

    temp_path = 'temp_image.jpg'
    file.save(temp_path)

    try:
        respuesta, codigo = analizar_imagen(temp_path)
        os.remove(temp_path)
        return jsonify(respuesta), codigo

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': f'Error procesando la imagen: {str(e)}'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
