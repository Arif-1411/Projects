"""
Image Encoder - CNN + CLIP Based
Encodes images into fixed-size embeddings
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from PIL import Image

class ImageEncoder(tf.keras.Model):
    """Image Encoder using CLIP or ResNet"""
    
    def __init__(self, model_name: str, embed_dim: int, image_size: tuple = (224, 224)):
        super(ImageEncoder, self).__init__()
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.image_size = image_size
        
        # Pre-trained model selection
        if "clip" in model_name.lower():
            from transformers import CLIPVisionModel
            self.vision_encoder = CLIPVisionModel.from_pretrained(model_name)
            self.feature_dim = 512
        else:
            # ResNet50 pre-trained
            base_model = tf.keras.applications.ResNet50(
                include_top=False,
                weights='imagenet',
                input_shape=(*image_size, 3),
                pooling='avg'
            )
            base_model.trainable = False
            self.vision_encoder = base_model
            self.feature_dim = 2048
        
        # Projection layer
        self.projection = layers.Dense(
            embed_dim, 
            activation='relu', 
            name='image_projection'
        )
        self.dropout = layers.Dropout(0.3)
    
    def preprocess_image(self, image_path_or_tensor):
        """Image preprocessing"""
        if isinstance(image_path_or_tensor, str):
            # File path
            img = tf.io.read_file(image_path_or_tensor)
            img = tf.image.decode_jpeg(img, channels=3)
        else:
            # Tensor
            img = image_path_or_tensor
        
        # Resize & normalize
        img = tf.image.resize(img, self.image_size)
        img = tf.keras.applications.resnet50.preprocess_input(img)
        return img
    
    def call(self, images, training=False):
        """
        Forward pass
        images: Batch of image tensors or file paths
        """
        # Preprocessing
        if images.dtype == tf.string:
            # Batch of file paths
            processed_images = tf.map_fn(
                lambda x: self.preprocess_image(x),
                images,
                fn_output_signature=tf.float32
            )
        else:
            processed_images = images
        
        # CLIP or ResNet
        if hasattr(self, 'vision_encoder') and "CLIP" in str(type(self.vision_encoder)):
            # CLIP
            outputs = self.vision_encoder(processed_images, training=training)
            pooled_output = outputs.pooler_output
        else:
            # ResNet
            pooled_output = self.vision_encoder(processed_images, training=False)
        
        # Projection
        projected = self.projection(pooled_output)
        projected = self.dropout(projected, training=training)
        
        return projected
    
    def get_embedding_dim(self):
        return self.embed_dim

# Test
if __name__ == "__main__":
    encoder = ImageEncoder("resnet50", 512)
    test_images = tf.random.uniform((2, 224, 224, 3), 0, 255, dtype=tf.float32)
    embeddings = encoder(test_images)
    print(f"Image Embedding Shape: {embeddings.shape}")