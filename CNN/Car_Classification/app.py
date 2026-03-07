import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import warnings

warnings.filterwarnings('ignore')

app      = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
MODEL_PATH    = os.path.join(BASE_DIR, 'model', 'final_model.keras')
TRAIN_DIR     = os.path.join(BASE_DIR, 'data', 'Cars Dataset', 'train')
TEST_DIR      = os.path.join(BASE_DIR, 'data', 'Cars Dataset', 'test')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
ALLOWED_EXT   = {'png', 'jpg', 'jpeg', 'webp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



def load_data():
    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )
    test_data = tf.keras.preprocessing.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )
    return train_data, test_data



def build_model(num_classes):
    base_model = EfficientNetB0(
        include_top=False,
        input_shape=(224, 224, 3),
        weights='imagenet'
    )
    base_model.trainable = False

    inputs  = layers.Input(shape=(224, 224, 3))
    x       = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x       = base_model(x, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dense(256, activation='relu')(x)  
    x       = layers.Dropout(0.3)(x)                   
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model, base_model


# ─────────────────────────────────────────
# Train and save model
# ─────────────────────────────────────────
def train_and_save():
    print("Loading dataset...")
    train_data, test_data = load_data()

    global CLASS_NAMES
    CLASS_NAMES = train_data.class_names
    num_classes = len(CLASS_NAMES)
    print(f"Classes ({num_classes}): {CLASS_NAMES}")

    model, base_model = build_model(num_classes)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=3,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=2, min_lr=1e-7, verbose=1
        )
    ]

    model.fit(
        train_data,
        epochs=5,
        validation_data=test_data,
        callbacks=callbacks
    )

    # Phase 2 — fine-tune top 10 layers
    base_model.trainable = True
    for layer in base_model.layers[:-10]:
        layer.trainable = False

    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )

    model.fit(
        train_data,
        epochs=10,
        validation_data=test_data,
        callbacks=callbacks
    )

    loss, acc = model.evaluate(test_data, verbose=0)
    print(f"\nTest Accuracy : {acc * 100:.2f}%")
    print(f"Test Loss     : {loss:.4f}")

    os.makedirs(os.path.join(BASE_DIR, 'model'), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")

    return model


# ─────────────────────────────────────────
# Load or train at startup
# ─────────────────────────────────────────
CLASS_NAMES = ['Audi', 'Hyundai Creta', 'Mahindra Scorpio',
               'Rolls Royce', 'Swift', 'Tata Safari', 'Toyota Innova']

if os.path.exists(MODEL_PATH):
    print("Loading saved model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model ready.")
else:
    print("No model found — training now...")
    model = train_and_save()


# ─────────────────────────────────────────
# Predict helper
# ─────────────────────────────────────────
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


def predict_image(img_path):
    img       = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    predictions   = model.predict(img_array, verbose=0)[0]
    predicted_idx = int(np.argmax(predictions))
    confidence    = float(predictions[predicted_idx]) * 100

    top3 = sorted(
        [(CLASS_NAMES[i], round(float(predictions[i]) * 100, 1))
         for i in range(len(CLASS_NAMES))],
        key=lambda x: x[1], reverse=True
    )[:3]

    return {
        'class':      CLASS_NAMES[predicted_idx],
        'confidence': round(confidence, 1),
        'top3':       top3
    }


# ─────────────────────────────────────────
# Flask
# ─────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', error='No file uploaded.')

    file = request.files['image']

    if file.filename == '':
        return render_template('index.html', error='No file selected.')

    if not allowed_file(file.filename):
        return render_template('index.html', error='Only JPG, PNG, WEBP allowed.')

    filename  = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    result    = predict_image(save_path)
    image_url = f'/static/uploads/{filename}'

    return render_template('result.html', result=result, image_url=image_url)


if __name__ == '__main__':
    app.run(debug=True)
