# src/web_app.py
from flask import Flask, request, jsonify, render_template_string
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Cargar el modelo
try:
    model = load_model('models/final_model.h5')
    print("✅ Modelo cargado correctamente")
except Exception as e:
    print("❌ Error al cargar el modelo:", e)
    model = None

# 🔹 Ruta principal: interfaz web simplificada
@app.route('/')
def home():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🩺 Golden Detect Anemic</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; text-align: center; margin: 40px; background: #f7f9fc; }
            .container { max-width: 600px; margin: 0 auto; padding: 20px; background: white; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; }
            p { color: #555; line-height: 1.5; }
            .upload-area { border: 2px dashed #3498db; padding: 20px; border-radius: 10px; margin: 20px 0; cursor: pointer; }
            .upload-area:hover { border-color: #2980b9; background: #f0f8ff; }
            button { padding: 12px 24px; font-size: 16px; background: #3498db; color: white; border: none; border-radius: 6px; cursor: pointer; margin-top: 10px; }
            button:hover { background: #2980b9; }
            #preview { max-width: 100%; height: 200px; object-fit: cover; margin: 15px 0; border-radius: 8px; display: none; }
            .result { margin: 20px; padding: 15px; border-radius: 8px; font-weight: bold; font-size: 1.1em; }
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
