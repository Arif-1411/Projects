import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import xgboost as xgb

# =========================
# Tabular Data (XGBoost)
# =========================

data = {
    'age': [63, 67],
    'sex': [1, 1],
    'chest_pain_type': [3, 3],
    'resting_bp': [145, 160],
    'cholesterol': [233, 286],
    'fasting_blood_sugar': [1, 0],
    'resting_ecg': [0, 0],
    'max_heart_rate': [150, 108],
    'exercise_induced_angina': [0, 1],
    'st_depression': [2.3, 1.5],
    'slope': [0, 1],
    'num_major_vessels': [0, 3],
    'thal': [1, 2],
    'target': [0, 1]
}

df = pd.DataFrame(data)
X = df.drop('target', axis=1)
y = df['target']

# Train XGBoost
dtrain = xgb.DMatrix(X, label=y, enable_categorical=True)
params = {
    'objective': 'binary:logistic',
    'max_depth': 6,
    'learning_rate': 0.1
}
bst = xgb.train(params, dtrain, num_boost_round=50)
xgb_probs = bst.predict(dtrain)

print("\n XGBoost Tabular Probabilities:")
for i, p in enumerate(xgb_probs):
    print(f"Patient {i}: {p:.3f}")

# =========================
# Dummy CNN on ECG Images
# =========================

img = np.random.rand(2, 224, 224, 3).astype(np.float32)

inputs = tf.keras.Input(shape=(224, 224, 3))
x = layers.Conv2D(16, 3, activation='relu')(inputs)
x = layers.MaxPooling2D()(x)
x = layers.Flatten()(x)
x = layers.Dense(2, activation='softmax')(x)
cnn_model = tf.keras.Model(inputs, x)

probs = cnn_model(img)
cnn_probs = probs[:, 1].numpy()

print("\n CNN ECG Probabilities:")
for i, p in enumerate(cnn_probs):
    print(f"Patient {i}: {p:.3f}")

# =========================
# Fusion
# =========================

fusion_weight = 0.6
fused = fusion_weight * cnn_probs + (1 - fusion_weight) * xgb_probs

print("\n Fused Risk Scores:")
for i, s in enumerate(fused):
    if s < 0.3:
        level = "Low"
    elif s < 0.7:
        level = "Medium"
    else:
        level = "High"
    print(f"Patient {i}: Risk Score = {s:.3f} → {level}")