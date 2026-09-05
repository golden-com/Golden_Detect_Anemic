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

SESIONES = {}
SESION_TTL_SEGUNDOS = 15 * 60


def limpiar_sesiones_viejas():
    ahora = time.time()
    vencidas = [sid for sid, s in SESIONES.items() if ahora - s['creada'] > SESION_TTL_SEGUNDOS]
    for sid in vencidas:
        SESIONES.pop(sid, None)


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

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("ADVERTENCIA: GEMINI_API_KEY no esta configurada. La verificacion de ojo/conjuntiva no puede ejecutarse.")


def validar_imagen_es_ojo(ruta_imagen):
    if not GEMINI_API_KEY:
        return False, "LA VERIFICACION DE IMAGEN NO ESTA DISPONIBLE EN ESTE MOMENTO. Intenta de nuevo mas tarde."

    try:
        with open(ruta_imagen, "rb") as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={GEMINI_API_KEY}"

        partes = []

        if REFERENCIAS_B64:
            partes.append({
                "text": "Las siguientes 1 o 2 imagenes son EJEMPLOS de referencia: fotografias validas de una conjuntiva palpebral inferior humana, bien enfocadas y encuadradas, donde el ojo ocupa una parte significativa de la imagen."
            })
            for ref_b64 in REFERENCIAS_B64:
                partes.append({"inline_data": {"mime_type": "image/jpeg", "data": ref_b64}})

        partes.append({
            "text": (
                "Ahora evalua la SIGUIENTE fotografia (la ultima imagen adjunta), comparandola con los "
                "ejemplos anteriores. Responde UNICAMENTE 'SI' si muestra claramente un ojo humano real, "
                "con la conjuntiva palpebral inferior visible, ocupando al menos un tercio del encuadre. "
                "Responde 'NO' si no es un ojo humano, si es un dibujo, objeto, paisaje, animal, pantalla, "
                "rostro completo sin acercamiento al ojo, si el ojo ocupa una porcion muy pequeña de la foto, "
                "si esta muy borrosa, muy oscura, o si es cualquier imagen no relacionada con un ojo humano. "
                "Ante cualquier duda, responde 'NO'."
            )
        })
        partes.append({"inline_data": {"mime_type": "image/jpeg", "data": image_data}})

        payload = {"contents": [{"parts": partes}]}

        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()
        data = response.json()
        texto = data["candidates"][0]["content"]["parts"][0]["text"].strip().upper()

        print(f"[Validacion ojo] Respuesta de Gemini: {texto[:80]}")

        if texto.startswith("SI"):
            return True, "OK"
        else:
            return False, "LA IMAGEN BRINDADA NO CONTIENE LA CONJUNTIVA PALPEBRAL REQUERIDA. Por favor, toma una foto clara del ojo donde se vea la parte interna del parpado inferior (conjuntiva), ocupando gran parte de la imagen. La foto debe estar enfocada y con buena iluminacion."

    except Exception as e:
        print(f"[Validacion ojo] Error: {e}")
        return False, "NO SE PUDO VERIFICAR LA IMAGEN. Intenta de nuevo con otra fotografia."


def consultar_gemini(ruta_imagen):
    if not GEMINI_API_KEY:
        return None

    try:
        with open(ruta_imagen, "rb") as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={GEMINI_API_KEY}"

        payload = {
            "contents": [{
                "parts": [
                    {
                        "text": "Eres un asistente de apoyo medico. Analiza esta imagen de un ojo humano, enfocandote especificamente en la conjuntiva palpebral inferior. Indica el nivel de probabilidad de anemia basado en la palidez observada. Responde UNICAMENTE con una de estas tres frases exactas: 'ALTA PROBABILIDAD', 'LEVE PROBABILIDAD', o 'BAJA PROBABILIDAD'. No uses comillas, no agregues explicaciones, ni saludos, ni otro texto."
                    },
                    {
                        "inline_data": {"mime_type": "image/jpeg", "data": image_data}
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
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Golden Detect Anemic</title>
<style>
*{margin:0;padding:0;box-sizing:border-box;font-family:'Segoe UI',system-ui,sans-serif}
body{background:linear-gradient(135deg,#fef3f2 0%,#fee2e2 100%);min-height:100vh;color:#1f2937}
.container{max-width:1100px;margin:0 auto;padding:20px}
header{text-align:center;padding:30px 0}
header h1{font-size:2.5rem;color:#b91c1c;margin-bottom:10px}
header p{color:#6b7280;font-size:1.1rem}
.hero{text-align:center;padding:40px 20px;background:white;border-radius:20px;box-shadow:0 10px 30px rgba(185,28,28,0.1);margin:20px 0}
.hero h2{font-size:2rem;color:#991b1b;margin-bottom:15px}
.hero p{color:#4b5563;margin-bottom:25px;font-size:1.05rem;line-height:1.6}
.btn{display:inline-block;padding:14px 32px;border-radius:50px;font-weight:600;text-decoration:none;cursor:pointer;border:none;font-size:1rem;transition:all 0.3s}
.btn-primary{background:linear-gradient(135deg,#dc2626,#b91c1c);color:white;box-shadow:0 4px 15px rgba(220,38,38,0.3)}
.btn-primary:hover{transform:translateY(-2px);box-shadow:0 6px 20px rgba(220,38,38,0.4)}
.btn-secondary{background:white;color:#b91c1c;border:2px solid #b91c1c}
.btn-secondary:hover{background:#b91c1c;color:white}
.features{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:20px;margin:30px 0}
.feature{background:white;padding:25px;border-radius:15px;text-align:center;box-shadow:0 4px 15px rgba(0,0,0,0.05)}
.feature-icon{font-size:2.5rem;margin-bottom:10px}
.feature h3{color:#991b1b;margin-bottom:8px}
.feature p{color:#6b7280;font-size:0.95rem}
.modal-overlay{display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.6);z-index:1000;align-items:center;justify-content:center}
.modal-overlay.active{display:flex}
.modal{background:white;border-radius:20px;padding:30px;max-width:500px;width:90%;max-height:90vh;overflow-y:auto;position:relative}
.modal h3{color:#991b1b;margin-bottom:15px}
.modal-close{position:absolute;top:15px;right:15px;background:none;border:none;font-size:1.5rem;cursor:pointer;color:#6b7280}
.upload-area{border:2px dashed #dc2626;border-radius:15px;padding:40px;text-align:center;cursor:pointer;transition:all 0.3s;background:#fef2f2}
.upload-area:hover{background:#fee2e2}
.upload-area.dragover{background:#fecaca;border-color:#991b1b}
.upload-area p{color:#6b7280;margin:10px 0}
.upload-icon{font-size:3rem;color:#dc2626}
#fileInput{display:none}
.options-grid{display:grid;grid-template-columns:1fr 1fr;gap:15px;margin-top:20px}
.option-card{background:#fef2f2;border:2px solid #fecaca;border-radius:15px;padding:20px;text-align:center;cursor:pointer;transition:all 0.3s}
.option-card:hover{border-color:#dc2626;background:#fee2e2;transform:translateY(-2px)}
.option-card h4{color:#991b1b;margin-bottom:8px}
.option-card p{color:#6b7280;font-size:0.9rem}
.option-icon{font-size:2.5rem;margin-bottom:10px}
.qr-section{text-align:center;padding:20px}
#qrCode{background:white;padding:15px;border-radius:15px;display:inline-block;margin:15px 0;box-shadow:0 4px 15px rgba(0,0,0,0.1)}
#qrCode img{width:250px;height:250px}
.steps{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:15px;margin:20px 0}
.step{background:white;padding:20px;border-radius:12px;text-align:center;box-shadow:0 2px 10px rgba(0,0,0,0.05)}
.step-num{background:#dc2626;color:white;width:35px;height:35px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;font-weight:bold;margin-bottom:10px}
.step h4{color:#991b1b;margin-bottom:5px;font-size:0.95rem}
.step p{color:#6b7280;font-size:0.85rem}
.warning-box{background:#fef3c7;border-left:4px solid #f59e0b;padding:15px;border-radius:8px;margin:20px 0;color:#78350f}
footer{text-align:center;padding:20px;color:#6b7280;font-size:0.9rem}
.loading{display:none;text-align:center;padding:30px}
.loading.active{display:block}
.spinner{border:4px solid #fecaca;border-top:4px solid #dc2626;border-radius:50%;width:50px;height:50px;animation:spin 1s linear infinite;margin:0 auto 15px}
@keyframes spin{0%{transform:rotate(0deg)}100%{transform:rotate(360deg)}}
.result-box{background:white;border-radius:15px;padding:25px;margin:20px 0;box-shadow:0 4px 15px rgba(0,0,0,0.1);display:none}
.result-box.active{display:block}
.result-alta{border-left:5px solid #dc2626}
.result-leve{border-left:5px solid #f59e0b}
.result-baja{border-left:5px solid #10b981}
.confidence-bar{background:#e5e7eb;height:10px;border-radius:5px;overflow:hidden;margin:10px 0}
.confidence-fill{height:100%;background:linear-gradient(90deg,#10b981,#f59e0b,#dc2626);transition:width 0.5s}
@media(max-width:768px){
.hero h2{font-size:1.5rem}
.options-grid{grid-template-columns:1fr}
}
</style>
</head>
<body>
<div class="container">
<header>
<h1>🩸 Golden Detect Anemic</h1>
<p>Evaluacion preliminar de anemia mediante inteligencia artificial</p>
</header>

<div class="hero">
<h2>Una fotografia de la conjuntiva palpebral inferior, analizada en segundos</h2>
<p>Golden Detect Anemic utiliza un modelo de inteligencia artificial entrenado para ofrecer una orientacion preliminar sobre posibles signos de anemia. <strong>No reemplaza un examen medico ni un analisis de sangre.</strong></p>
<button class="btn btn-primary" onclick="mostrarOpciones()">Iniciar mi analisis</button>
<button class="btn btn-secondary" onclick="mostrarPasos()" style="margin-left:10px">Descubre como funciona ↓</button>
</div>

<div class="features">
<div class="feature">
<div class="feature-icon">📋</div>
<h3>Evaluacion preliminar</h3>
<p>No diagnostica, solo orienta</p>
</div>
<div class="feature">
<div class="feature-icon">⚡</div>
<h3>Resultado en segundos</h3>
<p>Analisis rapido con IA</p>
</div>
<div class="feature">
<div class="feature-icon">💚</div>
<h3>Sin costo</h3>
<p>Uso educativo gratuito</p>
</div>
</div>

<div class="warning-box">
<strong>⚠️ Advertencia medica:</strong> Golden Detect Anemic es una herramienta unicamente informativa. Los resultados generados por la IA NO CONSTITUYEN UN DIAGNOSTICO MEDICO. En caso de duda o sospecha de anemia, consulte a un profesional de la salud.
</div>

<div id="seccionOpciones" style="display:none">
<div class="hero">
<h2>Analisis por imagen</h2>
<p>Elige como quieres proporcionar la fotografia.</p>
<div class="options-grid">
<div class="option-card" onclick="mostrarSubir()">
<div class="option-icon">📁</div>
<h4>Subir archivo</h4>
<p>Selecciona una foto ya tomada desde tu dispositivo.</p>
</div>
<div class="option-card" onclick="generarQR()">
<div class="option-icon">📱</div>
<h4>Usar mi celular</h4>
<p>Escanea un QR y toma la foto desde tu telefono.</p>
</div>
</div>
<button class="btn btn-secondary" onclick="ocultarOpciones()" style="margin-top:20px">← Volver</button>
</div>
</div>

<div id="seccionSubir" style="display:none">
<div class="hero">
<h2>Subir archivo</h2>
<p>Elige una foto donde se vea con claridad la conjuntiva palpebral inferior.</p>
<div class="upload-area" id="uploadArea">
<div class="upload-icon">📷</div>
<p><strong>Haz clic para seleccionar una imagen</strong></p>
<p>JPG o PNG</p>
<input type="file" id="fileInput" accept="image/*">
</div>
<div class="warning-box" style="margin-top:15px">
Esta es una evaluacion preliminar y no reemplaza un diagnostico medico ni un analisis de sangre.
</div>
<div class="loading" id="loadingUpload">
<div class="spinner"></div>
<p>Analizando imagen...</p>
</div>
<div class="result-box" id="resultUpload"></div>
<button class="btn btn-secondary" onclick="volverOpciones()" style="margin-top:15px">← Volver</button>
</div>
</div>

<div id="seccionQR" style="display:none">
<div class="hero">
<h2>Usa tu celular</h2>
<p>Escanea este codigo con tu telefono para tomar la foto desde ahi.</p>
<div class="qr-section">
<div id="qrCode"></div>
<p id="qrStatus">Esperando captura desde el celular...</p>
<div class="loading" id="loadingQR">
<div class="spinner"></div>
<p>Procesando imagen desde el celular...</p>
</div>
<div class="result-box" id="resultQR"></div>
</div>
<button class="btn btn-secondary" onclick="volverOpciones()" style="margin-top:15px">← Volver</button>
</div>
</div>

<div id="seccionPasos" style="display:none">
<div class="hero">
<h2>Como funciona el analisis</h2>
<p>Cinco pasos, sin cita previa, disponible cuando lo necesites.</p>
<div class="steps">
<div class="step">
<div class="step-num">1</div>
<h4>Captura</h4>
<p>Subes una foto o usas un QR para tomarla con tu telefono.</p>
</div>
<div class="step">
<div class="step-num">2</div>
<h4>Verificacion</h4>
<p>El modelo comprueba que sea un ojo humano con conjuntiva visible.</p>
</div>
<div class="step">
<div class="step-num">3</div>
<h4>Procesamiento</h4>
<p>La imagen se ajusta al formato que usa el modelo.</p>
</div>
<div class="step">
<div class="step-num">4</div>
<h4>Inteligencia artificial</h4>
<p>El modelo entrenado analiza la imagen recibida.</p>
</div>
<div class="step">
<div class="step-num">5</div>
<h4>Resultado</h4>
<p>Se muestra la evaluacion preliminar y su confianza.</p>
</div>
</div>
<button class="btn btn-primary" onclick="mostrarOpciones()">Iniciar mi analisis ahora</button>
</div>
</div>

<footer>
<p>Proyecto de solucion tecnologica · Institucion Educativa "Victor Manuel Maurtua" · Parcona, Ica</p>
</footer>
</div>

<script>
let sesionId = null;
let pollingQR = null;

function mostrarOpciones(){
document.getElementById('seccionOpciones').style.display='block';
document.getElementById('seccionSubir').style.display='none';
document.getElementById('seccionQR').style.display='none';
document.getElementById('seccionPasos').style.display='none';
document.querySelector('.hero').style.display='none';
document.querySelector('.features').style.display='none';
document.querySelector('.warning-box').style.display='none';
}

function ocultarOpciones(){
document.getElementById('seccionOpciones').style.display='none';
document.querySelector('.hero').style.display='block';
document.querySelector('.features').style.display='block';
document.querySelector('.warning-box').style.display='block';
}

function volverOpciones(){
document.getElementById('seccionSubir').style.display='none';
document.getElementById('seccionQR').style.display='none';
document.getElementById('seccionOpciones').style.display='block';
if(pollingQR){clearInterval(pollingQR);pollingQR=null;}
}

function mostrarPasos(){
document.getElementById('seccionPasos').style.display='block';
document.querySelector('.hero').style.display='none';
document.querySelector('.features').style.display='none';
document.querySelector('.warning-box').style.display='none';
}

function mostrarSubir(){
document.getElementById('seccionOpciones').style.display='none';
document.getElementById('seccionSubir').style.display='block';
}

const uploadArea=document.getElementById('uploadArea');
const fileInput=document.getElementById('fileInput');

uploadArea.addEventListener('click',()=>fileInput.click());
uploadArea.addEventListener('dragover',(e)=>{e.preventDefault();uploadArea.classList.add('dragover');});
uploadArea.addEventListener('dragleave',()=>uploadArea.classList.remove('dragover'));
uploadArea.addEventListener('drop',(e)=>{
e.preventDefault();
uploadArea.classList.remove('dragover');
if(e.dataTransfer.files.length)subirArchivo(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change',(e)=>{
if(e.target.files.length)subirArchivo(e.target.files[0]);
});

async function subirArchivo(file){
const formData=new FormData();
formData.append('image',file);
document.getElementById('loadingUpload').classList.add('active');
document.getElementById('resultUpload').classList.remove('active');
try{
const resp=await fetch('/predict',{method:'POST',body:formData});
const data=await resp.json();
document.getElementById('loadingUpload').classList.remove('active');
mostrarResultado('resultUpload',data);
}catch(e){
document.getElementById('loadingUpload').classList.remove('active');
mostrarResultado('resultUpload',{error:'Error al procesar la imagen: '+e.message});
}
}

async function generarQR(){
document.getElementById('seccionOpciones').style.display='none';
document.getElementById('seccionQR').style.display='block';
document.getElementById('qrCode').innerHTML='<p>Generando codigo QR...</p>';
document.getElementById('resultQR').classList.remove('active');
try{
const resp=await fetch('/api/sesion/nueva',{method:'POST'});
const data=await resp.json();
sesionId=data.sesion_id;
const urlQR=`${window.location.origin}/m/${sesionId}`;
document.getElementById('qrCode').innerHTML=`<img src="https://api.qrserver.com/v1/create-qr-code/?size=250x250&data=${encodeURIComponent(urlQR)}" alt="QR">`;
document.getElementById('qrStatus').textContent='Esperando captura desde el celular...';
iniciarPollingQR();
}catch(e){
document.getElementById('qrCode').innerHTML='<p style="color:#dc2626">Error al generar QR</p>';
}
}

function iniciarPollingQR(){
if(pollingQR)clearInterval(pollingQR);
pollingQR=setInterval(async()=>{
try{
const resp=await fetch(`/api/sesion/${sesionId}/estado`);
const data=await resp.json();
if(data.estado==='listo'){
clearInterval(pollingQR);
pollingQR=null;
document.getElementById('loadingQR').classList.remove('active');
if(data.error){
mostrarResultado('resultQR',{error:data.error});
}else{
mostrarResultado('resultQR',data);
}
}
}catch(e){}
},2000);
}

function mostrarResultado(elementId,data){
const el=document.getElementById(elementId);
el.classList.add('active');
if(data.error){
el.className='result-box active';
el.innerHTML=`<h3 style="color:#dc2626">⚠️ Atencion</h3><p>${data.error}</p>`;
return;
}
let clase='result-baja';
if(data.result.includes('ALTA'))clase='result-alta';
else if(data.result.includes('LEVE'))clase='result-leve';
el.className=`result-box active ${clase}`;
el.innerHTML=`
<h3 style="color:#991b1b">Resultado del analisis</h3>
<p style="font-size:1.2rem;font-weight:600;margin:15px 0">${data.result}</p>
<p>Confianza: <strong>${data.confidence}%</strong></p>
<div class="confidence-bar"><div class="confidence-fill" style="width:${data.confidence}%"></div></div>
<div class="warning-box" style="margin-top:15px">
Recuerda: este resultado es orientativo. Consulta a un profesional de la salud para un diagnostico definitivo.
</div>
`;
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
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>Captura guiada - Golden Detect</title>
<style>
*{margin:0;padding:0;box-sizing:border-box;font-family:'Segoe UI',system-ui,sans-serif;-webkit-tap-highlight-color:transparent}
html,body{height:100%;overflow:hidden;background:#000;color:white}
.camera-container{position:relative;width:100%;height:100vh;display:flex;flex-direction:column}
#video{width:100%;height:100%;object-fit:cover;background:#000;transition:filter 0.2s}
.top-bar{position:absolute;top:0;left:0;right:0;padding:15px;background:linear-gradient(to bottom,rgba(0,0,0,0.7),transparent);z-index:10;display:flex;justify-content:space-between;align-items:center}
.top-bar h2{font-size:1.1rem;color:white}
.status{font-size:0.85rem;color:#fca5a5}
.controls{position:absolute;bottom:0;left:0;right:0;padding:20px;background:linear-gradient(to top,rgba(0,0,0,0.85),transparent);z-index:10}
.zoom-controls{display:flex;justify-content:center;gap:10px;margin-bottom:15px}
.zoom-btn{background:rgba(255,255,255,0.15);color:white;border:1px solid rgba(255,255,255,0.3);padding:8px 16px;border-radius:20px;font-size:0.85rem;font-weight:600;cursor:pointer;backdrop-filter:blur(10px);transition:all 0.2s}
.zoom-btn.active{background:#dc2626;border-color:#dc2626}
.zoom-btn:active{transform:scale(0.95)}
.brightness-control{display:flex;align-items:center;gap:10px;margin-bottom:15px;padding:0 10px}
.brightness-control span{font-size:1.2rem}
.brightness-control input[type=range]{flex:1;-webkit-appearance:none;appearance:none;height:4px;background:rgba(255,255,255,0.3);border-radius:2px;outline:none}
.brightness-control input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:22px;height:22px;background:#dc2626;border-radius:50%;cursor:pointer;border:2px solid white}
.brightness-control input[type=range]::-moz-range-thumb{width:22px;height:22px;background:#dc2626;border-radius:50%;cursor:pointer;border:2px solid white}
.main-controls{display:flex;justify-content:space-around;align-items:center}
.ctrl-btn{background:rgba(255,255,255,0.15);color:white;border:none;width:55px;height:55px;border-radius:50%;font-size:1.5rem;cursor:pointer;backdrop-filter:blur(10px);transition:all 0.2s;display:flex;align-items:center;justify-content:center}
.ctrl-btn:active{transform:scale(0.9)}
.capture-btn{background:#dc2626;width:70px;height:70px;border:4px solid white;box-shadow:0 0 20px rgba(220,38,38,0.5)}
.capture-btn:disabled{background:#6b7280;border-color:#9ca3af;box-shadow:none}
.loading-overlay{position:absolute;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.85);display:none;align-items:center;justify-content:center;flex-direction:column;z-index:100}
.loading-overlay.active{display:flex}
.spinner{border:4px solid rgba(255,255,255,0.2);border-top:4px solid #dc2626;border-radius:50%;width:60px;height:60px;animation:spin 1s linear infinite;margin-bottom:20px}
@keyframes spin{0%{transform:rotate(0deg)}100%{transform:rotate(360deg)}}
.result-card{background:white;color:#1f2937;border-radius:20px;padding:25px;max-width:90%;width:350px;text-align:center;box-shadow:0 10px 40px rgba(0,0,0,0.5)}
.result-card h3{color:#991b1b;margin-bottom:15px}
.result-card.alta{border-top:5px solid #dc2626}
.result-card.leve{border-top:5px solid #f59e0b}
.result-card.baja{border-top:5px solid #10b981}
.result-card.error{border-top:5px solid #6b7280}
.confidence-bar{background:#e5e7eb;height:8px;border-radius:4px;overflow:hidden;margin:10px 0}
.confidence-fill{height:100%;background:linear-gradient(90deg,#10b981,#f59e0b,#dc2626)}
.result-btn{background:#dc2626;color:white;border:none;padding:12px 24px;border-radius:25px;font-weight:600;cursor:pointer;margin-top:15px;width:100%}
.camera-error{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center;padding:20px;color:white;max-width:90%}
.camera-error h3{color:#fca5a5;margin-bottom:10px}
</style>
</head>
<body>
<div class="camera-container">
<video id="video" autoplay playsinline muted></video>

<div class="top-bar">
<h2>🩸 Captura guiada</h2>
<span class="status" id="status">Iniciando camara...</span>
</div>

<div class="controls">
<div class="zoom-controls">
<button class="zoom-btn active" data-zoom="1">1x</button>
<button class="zoom-btn" data-zoom="1.5">1.5x</button>
<button class="zoom-btn" data-zoom="2">2x</button>
<button class="zoom-btn" data-zoom="3">3x</button>
</div>

<div class="brightness-control">
<span>☀️</span>
<input type="range" id="brightness" min="50" max="200" value="100">
<span id="brightnessValue">100%</span>
</div>

<div class="main-controls">
<button class="ctrl-btn" id="switchBtn" title="Cambiar camara">🔄</button>
<button class="ctrl-btn capture-btn" id="captureBtn" title="Capturar" disabled>📸</button>
<button class="ctrl-btn" id="retakeBtn" title="Tomar otra" style="display:none">↺</button>
</div>
</div>

<div class="loading-overlay" id="loadingOverlay">
<div class="spinner"></div>
<p id="loadingText">Analizando imagen...</p>
<div class="result-card" id="resultCard" style="display:none"></div>
</div>

<div class="camera-error" id="cameraError" style="display:none">
<h3>⚠️ No se pudo acceder a la camara</h3>
<p id="errorMsg">Verifica los permisos e intenta de nuevo.</p>
</div>
</div>

<script>
const SESION_ID = "{{ sesion_id }}";
const video = document.getElementById('video');
const captureBtn = document.getElementById('captureBtn');
const switchBtn = document.getElementById('switchBtn');
const retakeBtn = document.getElementById('retakeBtn');
const statusEl = document.getElementById('status');
const loadingOverlay = document.getElementById('loadingOverlay');
const loadingText = document.getElementById('loadingText');
const resultCard = document.getElementById('resultCard');
const cameraError = document.getElementById('cameraError');
const errorMsg = document.getElementById('errorMsg');
const brightnessSlider = document.getElementById('brightness');
const brightnessValue = document.getElementById('brightnessValue');
const zoomBtns = document.querySelectorAll('.zoom-btn');

let currentStream = null;
let facingMode = 'environment';
let currentZoom = 1;
let currentTrack = null;
let maxZoom = 1;
let captured = false;

async function iniciarCamara(){
try{
if(currentStream){
currentStream.getTracks().forEach(t=>t.stop());
}
const constraints = {
video: {
facingMode: facingMode,
width: {ideal: 1280},
height: {ideal: 720}
},
audio: false
};
const stream = await navigator.mediaDevices.getUserMedia(constraints);
currentStream = stream;
video.srcObject = stream;
await video.play();
currentTrack = stream.getVideoTracks()[0];
const settings = currentTrack.getSettings();
const capabilities = currentTrack.getCapabilities ? currentTrack.getCapabilities() : {};
maxZoom = capabilities.zoom ? Math.min(capabilities.zoom.max, 5) : 3;
statusEl.textContent = 'Camara activa - Encuadra tu ojo';
captureBtn.disabled = false;
cameraError.style.display = 'none';
aplicarZoom(currentZoom);
}catch(e){
console.error('Error camara:', e);
cameraError.style.display = 'block';
if(e.name === 'NotAllowedError'){
errorMsg.textContent = 'Permiso de camara denegado. Habilita el acceso en la configuracion de tu navegador.';
} else if(e.name === 'NotFoundError'){
errorMsg.textContent = 'No se encontro ninguna camara en este dispositivo.';
} else {
errorMsg.textContent = e.message || 'Error desconocido al acceder a la camara.';
}
statusEl.textContent = 'Error de camara';
captureBtn.disabled = true;
}
}

async function aplicarZoom(valor){
currentZoom = valor;
zoomBtns.forEach(b => {
b.classList.toggle('active', parseFloat(b.dataset.zoom) === valor);
});
if(!currentTrack) return;
const capabilities = currentTrack.getCapabilities ? currentTrack.getCapabilities() : {};
if(capabilities.zoom){
const targetZoom = Math.min(valor, capabilities.zoom.max);
try{
await currentTrack.applyConstraints({advanced: [{zoom: targetZoom}]});
video.style.transform = 'none';
} catch(e){
aplicarZoomCSS(valor);
}
} else {
aplicarZoomCSS(valor);
}
}

function aplicarZoomCSS(valor){
video.style.transform = `scale(${valor})`;
video.style.transformOrigin = 'center center';
}

switchBtn.addEventListener('click', () => {
facingMode = facingMode === 'environment' ? 'user' : 'environment';
statusEl.textContent = 'Cambiando camara...';
iniciarCamara();
});

zoomBtns.forEach(btn => {
btn.addEventListener('click', () => {
const valor = parseFloat(btn.dataset.zoom);
aplicarZoom(valor);
});
});

brightnessSlider.addEventListener('input', (e) => {
const val = e.target.value;
brightnessValue.textContent = val + '%';
video.style.filter = `brightness(${val/100})`;
});

captureBtn.addEventListener('click', async () => {
if(captured) return;
captured = true;
captureBtn.disabled = true;
statusEl.textContent = 'Capturando...';

const canvas = document.createElement('canvas');
const realWidth = video.videoWidth;
const realHeight = video.videoHeight;
canvas.width = realWidth;
canvas.height = realHeight;
const ctx = canvas.getContext('2d');

ctx.filter = `brightness(${brightnessSlider.value/100})`;

if(currentZoom > 1){
const zoomW = realWidth / currentZoom;
const zoomH = realHeight / currentZoom;
const sx = (realWidth - zoomW) / 2;
const sy = (realHeight - zoomH) / 2;
ctx.drawImage(video, sx, sy, zoomW, zoomH, 0, 0, realWidth, realHeight);
} else {
ctx.drawImage(video, 0, 0, realWidth, realHeight);
}

canvas.toBlob(async (blob) => {
const formData = new FormData();
formData.append('image', blob, 'captura.jpg');
loadingOverlay.classList.add('active');
loadingText.textContent = 'Analizando imagen...';
try{
const resp = await fetch(`/api/sesion/${SESION_ID}/capturar`, {
method: 'POST',
body: formData
});
const data = await resp.json();
mostrarResultado(data);
} catch(e){
mostrarResultado({error: 'Error al enviar la imagen: ' + e.message});
}
}, 'image/jpeg', 0.92);
});

retakeBtn.addEventListener('click', () => {
loadingOverlay.classList.remove('active');
resultCard.style.display = 'none';
captureBtn.disabled = false;
retakeBtn.style.display = 'none';
captureBtn.style.display = 'flex';
captured = false;
statusEl.textContent = 'Camara activa - Encuadra tu ojo';
});

function mostrarResultado(data){
loadingText.textContent = '';
resultCard.style.display = 'block';
retakeBtn.style.display = 'flex';
captureBtn.style.display = 'none';

if(data.error){
resultCard.className = 'result-card error';
resultCard.innerHTML = `
<h3>⚠️ Atencion</h3>
<p style="margin:15px 0;color:#4b5563">${data.error}</p>
<button class="result-btn" onclick="document.getElementById('retakeBtn').click()">Intentar de nuevo</button>
`;
return;
}

let clase = 'baja';
if(data.result.includes('ALTA')) clase = 'alta';
else if(data.result.includes('LEVE')) clase = 'leve';

resultCard.className = `result-card ${clase}`;
resultCard.innerHTML = `
<h3>Resultado del analisis</h3>
<p style="font-size:1.1rem;font-weight:600;margin:15px 0;color:#1f2937">${data.result}</p>
<p style="color:#6b7280">Confianza: <strong>${data.confidence}%</strong></p>
<div class="confidence-bar"><div class="confidence-fill" style="width:${data.confidence}%"></div></div>
<p style="font-size:0.85rem;color:#6b7280;margin-top:15px">
Recuerda: este resultado es orientativo. Consulta a un profesional de la salud para un diagnostico definitivo.
</p>
<button class="result-btn" onclick="document.getElementById('retakeBtn').click()">Tomar otra foto</button>
`;
}

iniciarCamara();
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
