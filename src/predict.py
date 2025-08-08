import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def predict_image(image_path):
    # Cargar el modelo
    model = load_model('models/final_model.h5')

    # Preprocesar la imagen
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Realizar la predicción
    prediction = model.predict(img_array)[0][0]
    result = 'Anemia' if prediction > 0.5 else 'No Anemia'

    return result