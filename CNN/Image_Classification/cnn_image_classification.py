
# ============================================
# 1. IMPORTS
# ============================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

print(f"TensorFlow Version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")


# ============================================
# 2. LOAD DATA
# ============================================
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()



print(f"Train Shape: {X_train.shape}")
print(f"Test Shape: {X_test.shape}")


# ============================================
# 3. VISUALIZE SAMPLE IMAGES
# ============================================

CLASS_NAMES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']


plt.figure(figsize=(10, 10))
plt.suptitle('CIFAR-10 Sample Images', fontsize=14, fontweight='bold')

for i in range(25):
    idx = np.random.randint(0, len(X_train))
    plt.subplot(5, 5, i + 1)
    plt.imshow(X_train[idx])
    plt.title(CLASS_NAMES[y_train[idx][0]], fontsize=8)
    plt.axis('off')

plt.tight_layout()
plt.show()


# ============================================
# 4. VISUALIZE CLASS DISTRIBUTION
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Class Distribution', fontsize=14, fontweight='bold')

_, counts_train = np.unique(y_train, return_counts=True)
axes[0].bar(CLASS_NAMES, counts_train, color='blue', edgecolor='white')
axes[0].set_title('Training Set')
axes[0].tick_params(axis='x', rotation=45)

_, counts_test = np.unique(y_test, return_counts=True)
axes[1].bar(CLASS_NAMES, counts_test, color='red', edgecolor='white')
axes[1].set_title('Test Set')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# ============================================
# 5. PREPROCESSING
# ============================================
X_train_norm = X_train.astype('float32') / 255.0
X_test_norm = X_test.astype('float32') / 255.0

y_train_enc = keras.utils.to_categorical(y_train, 10)
y_test_enc = keras.utils.to_categorical(y_test, 10)

X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train_norm, y_train_enc,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

print(f"Train: {X_train_split.shape[0]}")
print(f"Validation: {X_val.shape[0]}")
print(f"Test: {X_test_norm.shape[0]}")


# ============================================
# 6. DATA AUGMENTATION
# ============================================
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    shear_range=0.1,
    fill_mode='nearest'
)
datagen.fit(X_train_split)

# Show augmented images
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle('Data Augmentation Examples', fontsize=14, fontweight='bold')

sample_img = X_train_split[np.random.randint(0, len(X_train_split))]

for i in range(5):
    axes[0, i].imshow(sample_img)
    axes[0, i].set_title('Original' if i == 2 else '')
    axes[0, i].axis('off')

aug_iter = datagen.flow(sample_img.reshape(1, 32, 32, 3), batch_size=1)
for i in range(5):
    axes[1, i].imshow(next(aug_iter)[0])
    axes[1, i].set_title(f'Augmented {i+1}')
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()


# ============================================
# 7. BUILD CNN MODEL
# ============================================
cnn_model = models.Sequential(name='Custom_CNN')

# Block 1
cnn_model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
cnn_model.add(layers.BatchNormalization())
cnn_model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
cnn_model.add(layers.BatchNormalization())
cnn_model.add(layers.MaxPooling2D((2, 2)))
cnn_model.add(layers.Dropout(0.25))

# Block 2
cnn_model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
cnn_model.add(layers.BatchNormalization())
cnn_model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
cnn_model.add(layers.BatchNormalization())
cnn_model.add(layers.MaxPooling2D((2, 2)))
cnn_model.add(layers.Dropout(0.25))

# Block 3
cnn_model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
cnn_model.add(layers.BatchNormalization())
cnn_model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
cnn_model.add(layers.BatchNormalization())
cnn_model.add(layers.MaxPooling2D((2, 2)))
cnn_model.add(layers.Dropout(0.25))

# Fully Connected
cnn_model.add(layers.Flatten())
cnn_model.add(layers.Dense(256, activation='relu'))
cnn_model.add(layers.BatchNormalization())
cnn_model.add(layers.Dropout(0.5))
cnn_model.add(layers.Dense(128, activation='relu'))
cnn_model.add(layers.BatchNormalization())
cnn_model.add(layers.Dropout(0.5))
cnn_model.add(layers.Dense(10, activation='softmax'))

cnn_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cnn_model.summary()


# ============================================
# 8. TRAIN CNN
# ============================================
cnn_callbacks = [
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1)
]

cnn_history = cnn_model.fit(
    datagen.flow(X_train_split, y_train_split, batch_size=64),
    steps_per_epoch=len(X_train_split) // 64,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=cnn_callbacks,
    verbose=1
)


# ============================================
# 9. CNN TRAINING HISTORY
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('CNN Training History', fontsize=14, fontweight='bold')

# Accuracy
axes[0].plot(cnn_history.history['accuracy'], label='Train', color='#3498db', linewidth=2)
axes[0].plot(cnn_history.history['val_accuracy'], label='Validation', color='#e74c3c', linewidth=2, linestyle='--')
axes[0].set_title('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(cnn_history.history['loss'], label='Train', color='#3498db', linewidth=2)
axes[1].plot(cnn_history.history['val_loss'], label='Validation', color='#e74c3c', linewidth=2, linestyle='--')
axes[1].set_title('Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ============================================
# 10. CNN EVALUATION
# ============================================
cnn_loss, cnn_acc = cnn_model.evaluate(X_test_norm, y_test_enc, verbose=0)
print(f"\nCNN Test Accuracy: {cnn_acc * 100:.2f}%")
print(f"CNN Test Loss: {cnn_loss:.4f}")

cnn_preds = cnn_model.predict(X_test_norm, verbose=0)
cnn_pred_classes = np.argmax(cnn_preds, axis=1)
y_test_labels = y_test.flatten()

print("\nClassification Report:")
print(classification_report(y_test_labels, cnn_pred_classes, target_names=CLASS_NAMES, digits=4))


# ============================================
# 11. CNN CONFUSION MATRIX
# ============================================
cm = confusion_matrix(y_test_labels, cnn_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5, square=True)
plt.title('CNN Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ============================================
# 12. CNN PREDICTIONS VISUALIZATION
# ============================================
fig, axes = plt.subplots(3, 5, figsize=(15, 10))
fig.suptitle('CNN Predictions', fontsize=14, fontweight='bold')

for i in range(15):
    ax = axes[i // 5, i % 5]
    ax.imshow(X_test_norm[i])
    
    pred_class = cnn_pred_classes[i]
    true_class = y_test_labels[i]
    conf = np.max(cnn_preds[i]) * 100
    
    color = 'green' if pred_class == true_class else 'red'
    ax.set_title(f'True: {CLASS_NAMES[true_class]}\nPred: {CLASS_NAMES[pred_class]}\n{conf:.1f}%',
                 fontsize=8, color=color)
    ax.axis('off')

plt.tight_layout()
plt.show()


# ============================================
# 13. TRANSFER LEARNING - RESIZE IMAGES
# ============================================
X_train_resized = tf.image.resize(X_train_split, (224, 224)).numpy()
X_val_resized = tf.image.resize(X_val, (224, 224)).numpy()
X_test_resized = tf.image.resize(X_test_norm, (224, 224)).numpy()

print(f"Resized Train: {X_train_resized.shape}")
print(f"Resized Test: {X_test_resized.shape}")


# ============================================
# 14. BUILD TRANSFER LEARNING MODEL
# ============================================
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

tl_model = models.Sequential(name='EfficientNetB0_TL')
tl_model.add(base_model)
tl_model.add(layers.GlobalAveragePooling2D())
tl_model.add(layers.BatchNormalization())
tl_model.add(layers.Dense(256, activation='relu'))
tl_model.add(layers.Dropout(0.5))
tl_model.add(layers.Dense(128, activation='relu'))
tl_model.add(layers.Dropout(0.3))
tl_model.add(layers.Dense(10, activation='softmax'))

tl_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

tl_model.summary()


# ============================================
# 15. PHASE 1 - TRAIN HEAD ONLY
# ============================================

tl_callbacks = [
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1)
]

history_p1 = tl_model.fit(
    X_train_resized, y_train_split,
    batch_size=32,
    epochs=15,
    validation_data=(X_val_resized, y_val),
    callbacks=tl_callbacks,
    verbose=1
)


# ============================================
# 16. PHASE 2 - FINE TUNING
# ============================================

# Unfreeze last 20 layers
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

print(f"Trainable layers: {sum(1 for l in base_model.layers if l.trainable)}")

tl_model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_p2 = tl_model.fit(
    X_train_resized, y_train_split,
    batch_size=32,
    epochs=10,
    validation_data=(X_val_resized, y_val),
    callbacks=tl_callbacks,
    verbose=1
)


# ============================================
# 17. TRANSFER LEARNING HISTORY
# ============================================
# Combine histories
combined_acc = history_p1.history['accuracy'] + history_p2.history['accuracy']
combined_val_acc = history_p1.history['val_accuracy'] + history_p2.history['val_accuracy']
combined_loss = history_p1.history['loss'] + history_p2.history['loss']
combined_val_loss = history_p1.history['val_loss'] + history_p2.history['val_loss']
phase1_end = len(history_p1.history['accuracy'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Transfer Learning Training History', fontsize=14, fontweight='bold')

# Accuracy
axes[0].plot(combined_acc, label='Train', color='#3498db', linewidth=2)
axes[0].plot(combined_val_acc, label='Validation', color='#e74c3c', linewidth=2, linestyle='--')
axes[0].axvline(x=phase1_end - 1, color='#2ecc71', linestyle=':', linewidth=2, label='Fine-tune Start')
axes[0].set_title('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(combined_loss, label='Train', color='#3498db', linewidth=2)
axes[1].plot(combined_val_loss, label='Validation', color='#e74c3c', linewidth=2, linestyle='--')
axes[1].axvline(x=phase1_end - 1, color='#2ecc71', linestyle=':', linewidth=2, label='Fine-tune Start')
axes[1].set_title('Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ============================================
# 18. TRANSFER LEARNING EVALUATION
# ============================================
tl_loss, tl_acc = tl_model.evaluate(X_test_resized, y_test_enc, verbose=0)
print(f"\nTransfer Learning Test Accuracy: {tl_acc * 100:.2f}%")
print(f"Transfer Learning Test Loss: {tl_loss:.4f}")

tl_preds = tl_model.predict(X_test_resized, verbose=0)
tl_pred_classes = np.argmax(tl_preds, axis=1)

print("\nClassification Report:")
print(classification_report(y_test_labels, tl_pred_classes, target_names=CLASS_NAMES, digits=4))


# ============================================
# 19. TRANSFER LEARNING CONFUSION MATRIX
# ============================================
cm_tl = confusion_matrix(y_test_labels, tl_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_tl, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5, square=True)
plt.title('EfficientNetB0 Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# ============================================
# 20. MODEL COMPARISON
# ============================================
fig, ax = plt.subplots(figsize=(8, 6))

models_names = ['Custom CNN', 'EfficientNetB0\n(Transfer Learning)']
accuracies = [cnn_acc * 100, tl_acc * 100]
colors = ['blue', 'red']

bars = ax.bar(models_names, accuracies, color=colors, edgecolor='white', width=0.5)

for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f'{acc:.2f}%', ha='center', fontsize=12, fontweight='bold')

ax.set_title('Model Comparison - Test Accuracy', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy (%)')
ax.set_ylim([0, 100])
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================
# 21. FINAL RESULTS
# ============================================
print(f"Custom CNN           : {cnn_acc * 100:.2f}% | Loss: {cnn_loss:.4f}")
print(f"EfficientNetB0 (TL)  : {tl_acc * 100:.2f}% | Loss: {tl_loss:.4f}")


# ============================================
# 22. SAVE MODELS
# ============================================
cnn_model.save('cnn_model.h5')
tl_model.save('efficientnet_model.h5')
