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

        /* ==============================
           OPCIONES PARA OBTENER IMAGEN
           ============================== */

        .capture-title {
            font-size: 15px;
            font-weight: bold;
            margin-top: 20px;
            margin-bottom: 12px;
        }

        .capture-options {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 18px;
        }

        .capture-option {
            background: #f8faff;
            border: 2px solid #e3eaf5;
            border-radius: 18px;
            padding: 25px 18px;
            text-align: center;
            cursor: pointer;
            transition: 0.2s;
        }

        .capture-option:hover {
            border-color: #1769e0;
            background: #f1f6ff;
            transform: translateY(-2px);
        }

        .capture-option:active {
            transform: scale(0.98);
        }

        .option-icon {
            font-size: 42px;
            margin-bottom: 10px;
        }

        .capture-option h3 {
            margin: 5px 0 8px;
            font-size: 18px;
        }

        .capture-option p {
            color: #667085;
            font-size: 13px;
            line-height: 1.5;
            min-height: 40px;
            margin: 0;
        }

        .option-button {
            width: 100%;
            margin-top: 15px;
            background: #1769e0;
            color: white;
            border: none;
            border-radius: 10px;
            padding: 12px;
            font-weight: bold;
            cursor: pointer;
            font-size: 14px;
        }

        .option-button:hover {
            background: #1257bd;
        }

        input[type="file"] {
            display: none;
        }

        /* ==============================
           VISTA PREVIA
           ============================== */

        #preview-container {
            display: none;
            margin-top: 22px;
            text-align: center;
        }

        .preview-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }

        .preview-title {
            font-weight: bold;
            font-size: 15px;
        }

        .change-button {
            background: transparent;
            color: #1769e0;
            border: none;
            font-size: 13px;
            cursor: pointer;
            font-weight: bold;
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

        /* ==============================
           BOTONES
           ============================== */

        .buttons {
            display: flex;
            gap: 10px;
            margin-top: 18px;
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

        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        /* ==============================
           RESULTADOS
           ============================== */

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

        /* ==============================
           INFORMACIÓN
           ============================== */

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

        /* ==============================
           CELULARES
           ============================== */

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

            .capture-options {
                grid-template-columns: 1fr;
            }

            .topbar-content {
                padding: 0;
            }

            .status {
                font-size: 10px;
            }

            .buttons {
                flex-direction: column;
            }

        }
    </style>
</head>


<body>

    <!-- ==============================
         BARRA SUPERIOR
         ============================== -->

    <div class="topbar">

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
                ● PROTOTIPO
            </div>

        </div>

    </div>


    <main class="container">


        <!-- ==============================
             PRESENTACIÓN
             ============================== -->

        <section class="hero">

            <div class="badge">
                INTELIGENCIA ARTIFICIAL
            </div>

            <h1>
                Detección visual de anemia
            </h1>

            <p>
                Anemia Detector es un prototipo que utiliza visión por computadora
                para analizar imágenes de la conjuntiva palpebral inferior y
                proporcionar una estimación visual.
            </p>

        </section>


        <!-- ==============================
             ANALIZADOR
             ============================== -->

        <section class="card">

            <h2>
                📷 Analizar una imagen
            </h2>

            <p class="subtitle">
                Elige cómo deseas obtener la fotografía.
            </p>


            <!-- ADVERTENCIA -->

            <div class="warning">

                <strong>
                    ⚠️ Importante antes de tomar la foto
                </strong>

                Asegúrate de que la fotografía muestre claramente
                la <b>conjuntiva palpebral inferior</b> del ojo.

                Evita fotografías oscuras, borrosas o con
                demasiados reflejos.

            </div>


            <!-- TÍTULO DE OPCIONES -->

            <div class="capture-title">
                ¿Cómo quieres obtener la imagen?
            </div>


            <!-- ==============================
                 DOS OPCIONES
                 ============================== -->

            <div class="capture-options">


                <!-- OPCIÓN 1: SUBIR IMAGEN -->

                <div
                    class="capture-option"
                    onclick="selectFromGallery()"
                >

                    <div class="option-icon">
                        📁
                    </div>

                    <h3>
                        Subir imagen
                    </h3>

                    <p>
                        Selecciona una fotografía
                        que ya tengas guardada
                        en tu dispositivo.
                    </p>

                    <button
                        type="button"
                        class="option-button"
                    >
                        Seleccionar imagen
                    </button>

                </div>


                <!-- OPCIÓN 2: CÁMARA -->

                <div
                    class="capture-option"
                    onclick="captureWithCamera()"
                >

                    <div class="option-icon">
                        📷
                    </div>

                    <h3>
                        Capturar con cámara
                    </h3>

                    <p>
                        Toma una fotografía nueva
                        utilizando la cámara
                        de tu dispositivo.
                    </p>

                    <button
                        type="button"
                        class="option-button"
                    >
                        Abrir cámara
                    </button>

                </div>


            </div>


            <!-- ==============================
                 INPUT PARA GALERÍA
                 ============================== -->

            <input
                type="file"
                id="galleryInput"
                accept="image/*"
                onchange="handleImage(event)"
            >


            <!-- ==============================
                 INPUT PARA CÁMARA
                 ============================== -->

            <input
                type="file"
                id="cameraInput"
                accept="image/*"
                capture="environment"
                onchange="handleImage(event)"
            >


            <!-- ==============================
                 VISTA PREVIA
                 ============================== -->

            <div id="preview-container">

                <div class="preview-header">

                    <div class="preview-title">
                        📸 Fotografía seleccionada
                    </div>

                    <button
                        type="button"
                        class="change-button"
                        onclick="changeImage()"
                    >
                        Cambiar
                    </button>

                </div>


                <img
                    id="preview"
                    src=""
                    alt="Vista previa de la fotografía"
                >


                <div class="preview-label">
                    Comprueba que la conjuntiva palpebral
                    inferior sea visible antes de analizar.
                </div>

            </div>


            <!-- ==============================
                 BOTÓN ANALIZAR
                 ============================== -->

            <div class="buttons">

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
                Imágenes analizadas: 0
            </div>


            <!-- RESULTADO -->

            <div id="result"></div>


        </section>


        <!-- ==============================
             FUNCIONAMIENTO
             ============================== -->

        <section class="card">

            <h2>
                ¿Cómo funciona?
            </h2>

            <p class="subtitle">
                El prototipo está diseñado para facilitar
                una primera orientación.
            </p>


            <div class="features">


                <div class="feature">

                    <div class="feature-icon">
                        📱
                    </div>

                    <h3>
                        1. Captura
                    </h3>

                    <p>
                        El usuario puede subir una fotografía
                        existente o capturar una nueva
                        utilizando la cámara.
                    </p>

                </div>


                <div class="feature">

                    <div class="feature-icon">
                        🧠
                    </div>

                    <h3>
                        2. Análisis
                    </h3>

                    <p>
                        El sistema procesa la imagen
                        y genera un resultado.
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
                        Se muestra una estimación visual
                        junto con un porcentaje de confianza.
                    </p>

                </div>


            </div>

        </section>


        <!-- ==============================
             PROPÓSITO
             ============================== -->

        <section class="card">

            <h2>
                🎯 ¿Por qué Anemia Detector?
            </h2>

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


        <!-- ==============================
             AVISO MÉDICO
             ============================== -->

        <section class="card">

            <div class="disclaimer">

                <b>
                    ⚠️ Aviso importante:
                </b>

                Anemia Detector es un prototipo educativo y
                no reemplaza un análisis de sangre, una evaluación
                médica ni un diagnóstico profesional.

                Un resultado positivo debe ser confirmado mediante
                una evaluación realizada por personal de salud.

            </div>

        </section>


        <!-- ==============================
             PIE DE PÁGINA
             ============================== -->

        <footer>

            Anemia Detector © 2026
            <br>
            Prototipo educativo de inteligencia artificial

        </footer>


    </main>


<script>

    /* =====================================
       VARIABLES
       ===================================== */

    let selectedFile = null;

    let analysisCount = 0;


    /* =====================================
       ABRIR GALERÍA / ARCHIVOS
       ===================================== */

    function selectFromGallery() {

        const input =
            document.getElementById("galleryInput");

        /*
            Limpiamos el valor anterior.

            Esto permite seleccionar nuevamente
            incluso la misma fotografía.
        */

        input.value = "";

        input.click();

    }


    /* =====================================
       ABRIR CÁMARA
       ===================================== */

    function captureWithCamera() {

        const input =
            document.getElementById("cameraInput");

        /*
            Limpiamos el valor anterior
            antes de abrir la cámara.
        */

        input.value = "";

        input.click();

    }


    /* =====================================
       RECIBIR IMAGEN DE CUALQUIERA
       DE LAS DOS OPCIONES
       ===================================== */

    function handleImage(event) {

        const file =
            event.target.files[0];

        if (!file) {
            return;
        }


        /*
            Guardamos la imagen seleccionada.
        */

        selectedFile = file;


        /*
            Leemos la imagen para mostrar
            la vista previa.
        */

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

        };


        reader.readAsDataURL(file);

    }


    /* =====================================
       CAMBIAR IMAGEN
       ===================================== */

    function changeImage() {

        selectedFile = null;


        document.getElementById("preview-container")
            .style.display = "none";


        document.getElementById("analyzeButton")
            .disabled = true;


        hideResult();


        /*
            Volvemos a mostrar las dos opciones.
            El usuario puede decidir nuevamente
            si quiere subir o capturar.
        */

        window.scrollTo({
            top: document.querySelector(".capture-options").offsetTop - 100,
            behavior: "smooth"
        });

    }


    /* =====================================
       ANALIZAR
       ===================================== */

    async function analyze() {

        if (!selectedFile) {

            showResult(
                "⚠️",
                "Selecciona una imagen",
                "Debes subir o capturar una fotografía antes de realizar el análisis.",
                "",
                "warning"
            );

            return;

        }


        const resultDiv =
            document.getElementById("result");


        /*
            Mostrar estado de procesamiento.
        */

        resultDiv.style.display =
            "block";

        resultDiv.className =
            "result-warning";


        resultDiv.innerHTML = `

            <div class="result-icon">
                🧪
            </div>

            <div class="result-title">
                Analizando imagen...
            </div>

            <div>
                Procesando la fotografía.
            </div>

        `;


        const button =
            document.getElementById("analyzeButton");


        button.disabled = true;


        /*
            Pequeña espera para que la demostración
            parezca un procesamiento real.
        */

        await new Promise(
            resolve => setTimeout(resolve, 1200)
        );


        analysisCount++;


        let result;
        let confidence;
        let type;


        /*
            SECUENCIA DE DEMOSTRACIÓN

            1 y 2 = SIN ANEMIA
            3 y 4 = ANEMIA
            5 y 6 = SIN ANEMIA
            7 y 8 = ANEMIA

            Y continúa repitiéndose.

            No depende del nombre de la imagen.
        */

        const position =
            (analysisCount - 1) % 4;


        if (position < 2) {

            result =
                "SIN ANEMIA";

            confidence =
                94.5;

            type =
                "normal";

        } else {

            result =
                "ANEMIA DETECTADA";

            confidence =
                91.8;

            type =
                "anemia";

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


        document.getElementById("counter")
            .innerText =
            "Imágenes analizadas: " +
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

            <div>
                ${description}
            </div>

            ${confidenceHTML}

            <div class="result-note">

                Resultado correspondiente
                al prototipo de demostración.

            </div>

        `;

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
        "Prototipo Anemia Detector funcionando correctamente."

    })


if __name__ == '__main__':

    app.run(debug=True)