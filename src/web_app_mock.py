from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

html = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>Anemia Detector</title>

    <style>
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: Arial, Helvetica, sans-serif;
            background: #f4f7fb;
            color: #172033;
        }

        .topbar {
            background: #ffffff;
            border-bottom: 1px solid #e8edf5;
            padding: 15px 20px;
            position: sticky;
            top: 0;
            z-index: 10;
        }

        .topbar-content {
            max-width: 1000px;
            margin: auto;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: bold;
            font-size: 20px;
        }

        .brand-icon {
            width: 42px;
            height: 42px;
            border-radius: 12px;
            background: #e9f2ff;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 22px;
        }

        .status {
            background: #eaf8ef;
            color: #18864b;
            border-radius: 20px;
            padding: 7px 12px;
            font-size: 12px;
            font-weight: bold;
        }

        .container {
            width: 100%;
            max-width: 1000px;
            margin: auto;
            padding: 25px 16px 45px;
        }

        .hero {
            background: linear-gradient(135deg, #1769e0, #4b9cff);
            color: white;
            border-radius: 24px;
            padding: 35px 25px;
            margin-bottom: 22px;
            position: relative;
            overflow: hidden;
        }

        .hero::after {
            content: "";
            position: absolute;
            width: 180px;
            height: 180px;
            border-radius: 50%;
            background: rgba(255,255,255,0.08);
            right: -60px;
            top: -60px;
        }

        .hero h1 {
            margin: 0 0 10px;
            font-size: 32px;
        }

        .hero p {
            margin: 0;
            max-width: 650px;
            line-height: 1.6;
            opacity: 0.94;
        }

        .badge {
            display: inline-block;
            background: rgba(255,255,255,0.18);
            padding: 7px 12px;
            border-radius: 20px;
            font-size: 12px;
            margin-bottom: 15px;
        }

        .card {
            background: white;
            border-radius: 20px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 5px 20px rgba(20, 40, 80, 0.06);
        }

        .card h2 {
            margin-top: 0;
            margin-bottom: 8px;
            font-size: 21px;
        }

        .subtitle {
            color: #667085;
            font-size: 14px;
            margin-top: 0;
        }

        .warning {
            background: #fff8e6;
            border: 1px solid #ffe2a8;
            border-radius: 14px;
            padding: 15px;
            margin: 18px 0;
            color: #76530a;
            font-size: 14px;
            line-height: 1.5;
        }

        .warning strong {
            display: block;
            margin-bottom: 5px;
        }

        .upload-area {
            border: 2px dashed #b7c9e8;
            border-radius: 18px;
            padding: 30px 18px;
            text-align: center;
            cursor: pointer;
            transition: 0.2s;
            background: #fafcff;
        }

        .upload-area:hover {
            border-color: #2878e8;
            background: #f3f8ff;
        }

        .upload-icon {
            font-size: 40px;
            margin-bottom: 8px;
        }

        .upload-area h3 {
            margin: 5px 0;
            font-size: 17px;
        }

        .upload-area p {
            margin: 5px 0;
            color: #667085;
            font-size: 13px;
        }

        input[type="file"] {
            display: none;
        }

        .buttons {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }

        button {
            flex: 1;
            border: none;
            border-radius: 12px;
            padding: 14px;
            font-size: 15px;
            font-weight: bold;
            cursor: pointer;
            transition: 0.2s;
        }

        .primary {
            background: #1769e0;
            color: white;
        }

        .primary:hover {
            background: #1257bd;
        }

        .secondary {
            background: #edf3fc;
            color: #1769e0;
        }

        .secondary:hover {
            background: #dfeafb;
        }

        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        #preview-container {
            display: none;
            margin-top: 18px;
            text-align: center;
        }

        #preview {
            width: 100%;
            max-height: 330px;
            object-fit: contain;
            border-radius: 15px;
            background: #f1f3f6;
            border: 1px solid #e2e6ec;
        }

        .preview-label {
            font-size: 12px;
            color: #667085;
            margin-top: 7px;
        }

        #result {
            display: none;
            margin-top: 20px;
            border-radius: 17px;
            padding: 22px;
            text-align: center;
        }

        .result-normal {
            background: #edf9f1;
            border: 1px solid #b8e5c6;
            color: #166534;
        }

        .result-anemia {
            background: #fff0f0;
            border: 1px solid #f2b8b8;
            color: #b42318;
        }

        .result-warning {
            background: #fff8e6;
            border: 1px solid #ffe0a3;
            color: #8a5a00;
        }

        .result-icon {
            font-size: 42px;
            margin-bottom: 5px;
        }

        .result-title {
            font-size: 22px;
            font-weight: bold;
            margin-bottom: 8px;
        }

        .confidence {
            font-size: 16px;
            margin-top: 8px;
        }

        .result-note {
            font-size: 12px;
            margin-top: 13px;
            opacity: 0.8;
        }

        .features {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
        }

        .feature {
            background: #f8faff;
            border: 1px solid #edf1f7;
            border-radius: 15px;
            padding: 18px;
        }

        .feature-icon {
            font-size: 25px;
        }

        .feature h3 {
            font-size: 15px;
            margin: 8px 0 5px;
        }

        .feature p {
            font-size: 12px;
            color: #667085;
            line-height: 1.5;
            margin: 0;
        }

        .disclaimer {
            background: #f8f9fb;
            border-radius: 15px;
            padding: 16px;
            font-size: 12px;
            color: #667085;
            line-height: 1.6;
        }

        footer {
            text-align: center;
            color: #8a94a6;
            font-size: 12px;
            padding: 10px;
        }

        .counter {
            text-align: center;
            color: #667085;
            font-size: 12px;
            margin-top: 12px;
        }

        @media (max-width: 650px) {

            .hero {
                padding: 27px 20px;
                border-radius: 20px;
            }

            .hero h1 {
                font-size: 27px;
            }

            .card {
                padding: 18px;
                border-radius: 17px;
            }

            .features {
                grid-template-columns: 1fr;
            }

            .buttons {
                flex-direction: column;
            }

            .topbar-content {
                padding: 0;
            }

            .status {
                font-size: 10px;
            }
        }
    </style>
</head>

<body>

    <div class="topbar">
        <div class="topbar-content">

            <div class="brand">
                <div class="brand-icon">🩺</div>
                <span>Anemia Detector</span>
            </div>

            <div class="status">
                ● PROTOTIPO
            </div>

        </div>
    </div>

    <main class="container">

        <section class="hero">

            <div class="badge">
                INTELIGENCIA ARTIFICIAL
            </div>

            <h1>Detección visual de anemia</h1>

            <p>
                Anemia Detector es un prototipo que utiliza visión por computadora
                para analizar imágenes de la conjuntiva palpebral inferior y
                proporcionar una estimación visual.
            </p>

        </section>


        <section class="card">

            <h2>📷 Analizar una imagen</h2>

            <p class="subtitle">
                Selecciona una fotografía o utiliza la cámara de tu dispositivo.
            </p>

            <div class="warning">

                <strong>⚠️ Importante antes de tomar la foto</strong>

                Asegúrate de que la fotografía muestre claramente
                la <b>conjuntiva palpebral inferior</b> del ojo.
                Evita fotografías oscuras, borrosas o con demasiados reflejos.

            </div>


            <label class="upload-area" for="image">

                <div class="upload-icon">📸</div>

                <h3>Seleccionar fotografía</h3>

                <p>
                    Toca aquí para elegir una imagen
                    desde tu dispositivo.
                </p>

            </label>


            <input
                type="file"
                id="image"
                accept="image/*"
                capture="environment"
                onchange="previewImage(event)"
            >


            <div id="preview-container">

                <img id="preview" src="" alt="Vista previa">

                <div class="preview-label">
                    Vista previa de la fotografía seleccionada
                </div>

            </div>


            <div class="buttons">

                <button
                    class="secondary"
                    onclick="openCamera()"
                >
                    📷 Tomar fotografía
                </button>

                <button
                    class="primary"
                    id="analyzeButton"
                    onclick="analyze()"
                    disabled
                >
                    🔍 Analizar imagen
                </button>

            </div>


            <div class="counter" id="counter">
                Imágenes analizadas: 0
            </div>


            <div id="result"></div>

        </section>


        <section class="card">

            <h2>¿Cómo funciona?</h2>

            <p class="subtitle">
                El prototipo está diseñado para facilitar una primera orientación.
            </p>

            <div class="features">

                <div class="feature">

                    <div class="feature-icon">📱</div>

                    <h3>1. Captura</h3>

                    <p>
                        El usuario toma o selecciona una fotografía
                        de la conjuntiva palpebral inferior.
                    </p>

                </div>


                <div class="feature">

                    <div class="feature-icon">🧠</div>

                    <h3>2. Análisis</h3>

                    <p>
                        El sistema procesa la imagen y genera
                        un resultado de manera automática.
                    </p>

                </div>


                <div class="feature">

                    <div class="feature-icon">📊</div>

                    <h3>3. Resultado</h3>

                    <p>
                        Se muestra una estimación visual junto
                        con un porcentaje de confianza.
                    </p>

                </div>

            </div>

        </section>


        <section class="card">

            <h2>🎯 ¿Por qué Anemia Detector?</h2>

            <p class="subtitle">

                El propósito del proyecto es brindar una alternativa
                tecnológica sencilla que pueda servir como apoyo
                para identificar posibles casos que requieran
                una evaluación médica.

            </p>

            <p class="subtitle">

                La herramienta busca ser especialmente útil como
                orientación inicial para personas que tienen
                dificultades para acceder rápidamente a un
                establecimiento de salud.

            </p>

        </section>


        <section class="card">

            <div class="disclaimer">

                <b>⚠️ Aviso importante:</b>

                Anemia Detector es un prototipo educativo y no reemplaza
                un análisis de sangre, una evaluación médica ni un diagnóstico
                profesional. Un resultado positivo debe ser confirmado
                mediante una evaluación realizada por personal de salud.

            </div>

        </section>


        <footer>

            Anemia Detector © 2026<br>
            Prototipo educativo de inteligencia artificial

        </footer>

    </main>


<script>

    let selectedFile = null;

    /*
        Contador de demostración.

        Funcionamiento:
        imágenes 1 y 2  -> SIN ANEMIA
        imágenes 3 y 4  -> ANEMIA
        imágenes 5 y 6  -> SIN ANEMIA
        imágenes 7 y 8  -> ANEMIA
        ...

        No depende del nombre de la fotografía.
    */

    let analysisCount = 0;


    function openCamera() {

        const input = document.getElementById("image");

        input.setAttribute("capture", "environment");

        input.click();

    }


    function previewImage(event) {

        const file = event.target.files[0];

        if (!file) {
            return;
        }

        selectedFile = file;

        const reader = new FileReader();

        reader.onload = function(e) {

            const preview = document.getElementById("preview");
            const container = document.getElementById("preview-container");
            const button = document.getElementById("analyzeButton");

            preview.src = e.target.result;

            container.style.display = "block";

            button.disabled = false;

            hideResult();

        };

        reader.readAsDataURL(file);

    }


    async function analyze() {

        if (!selectedFile) {

            showResult(
                "⚠️",
                "Selecciona una imagen",
                "Debes subir una fotografía antes de realizar el análisis.",
                "",
                "warning"
            );

            return;
        }


        const resultDiv = document.getElementById("result");

        resultDiv.style.display = "block";

        resultDiv.className = "result-warning";

        resultDiv.innerHTML = `
            <div class="result-icon">🧪</div>
            <div class="result-title">Analizando imagen...</div>
            <div>Procesando la fotografía.</div>
        `;


        const button = document.getElementById("analyzeButton");

        button.disabled = true;


        await new Promise(resolve => setTimeout(resolve, 1200));


        analysisCount++;


        let result;
        let confidence;
        let type;


        /*
            Cada grupo contiene dos resultados:

            1, 2  -> sin anemia
            3, 4  -> anemia
            5, 6  -> sin anemia
            7, 8  -> anemia

            Fórmula:
            (analysisCount - 1) % 4

            0 o 1 = sin anemia
            2 o 3 = anemia
        */

        const position = (analysisCount - 1) % 4;


        if (position < 2) {

            result = "SIN ANEMIA";
            confidence = 94.5;
            type = "normal";

        } else {

            result = "ANEMIA DETECTADA";
            confidence = 91.8;
            type = "anemia";

        }


        if (type === "normal") {

            showResult(
                "🟢",
                result,
                "No se identificaron signos visuales compatibles con anemia en esta demostración.",
                confidence,
                "normal"
            );

        } else {

            showResult(
                "🔴",
                result,
                "Se identificaron características visuales compatibles con posible anemia.",
                confidence,
                "anemia"
            );

        }


        document.getElementById("counter").innerText =
            "Imágenes analizadas: " + analysisCount;


        button.disabled = false;

    }


    function showResult(icon, title, description, confidence, type) {

        const resultDiv = document.getElementById("result");

        resultDiv.style.display = "block";

        resultDiv.className = "result-" + type;


        let confidenceHTML = "";

        if (confidence !== "") {

            confidenceHTML = `
                <div class="confidence">
                    Confianza estimada: <b>${confidence}%</b>
                </div>
            `;

        }


        resultDiv.innerHTML = `

            <div class="result-icon">${icon}</div>

            <div class="result-title">
                ${title}
            </div>

            <div>
                ${description}
            </div>

            ${confidenceHTML}

            <div class="result-note">
                Resultado correspondiente al prototipo de demostración.
            </div>

        `;

    }


    function hideResult() {

        const resultDiv = document.getElementById("result");

        resultDiv.style.display = "none";

        resultDiv.innerHTML = "";

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

    """
    Ruta de demostración.

    No utiliza random y no depende del nombre del archivo.

    La secuencia se maneja principalmente desde JavaScript
    para que cada usuario pueda realizar su propia demostración.
    """

    return jsonify({
        "status": "ok",
        "message": "Prototipo Anemia Detector funcionando correctamente."
    })


if __name__ == '__main__':
    app.run(debug=True)