import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, Input, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import pandas as pd
import os

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

def create_lightweight_model(num_classes, input_shape=(160, 160, 3)):
    """Create a lightweight custom model instead of using EfficientNet"""
    inputs = Input(shape=input_shape)
    
    # Use smaller filters and fewer channels
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs)

def optimize_data_pipeline(csv_path, image_dir, batch_size=64):
    """Create an optimized data pipeline"""
    data = pd.read_csv(csv_path)
    data['Image_Path'] = data['filename'].apply(lambda x: os.path.join(image_dir, x))
    
    # Use tf.data for faster data loading
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        validation_split=0.2,
        # Minimal augmentation for speed
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    # Use smaller image size
    train_generator = train_datagen.flow_from_dataframe(
        data,
        x_col='Image_Path',
        y_col='subCategory',
        target_size=(160, 160),  # Reduced from 224x224
        batch_size=batch_size,   # Increased batch size
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_dataframe(
        data,
        x_col='Image_Path',
        y_col='subCategory',
        target_size=(160, 160),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    # Enable prefetching
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = tf.data.Dataset.from_generator(
        lambda: train_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 160, 160, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, len(data['subCategory'].unique())), dtype=tf.float32)
        )
    ).prefetch(AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_generator(
        lambda: validation_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 160, 160, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, len(data['subCategory'].unique())), dtype=tf.float32)
        )
    ).prefetch(AUTOTUNE)
    
    return train_ds, val_ds, len(data['subCategory'].unique())

def compile_and_train(model, train_ds, val_ds, epochs=5):
    """Compile and train the model with optimized settings"""
    optimizer = Adam(learning_rate=0.001)
    
    # Use AMP (Automatic Mixed Precision) for faster training
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        jit_compile=True  # Enable XLA compilation
    )
    
    # Use steps_per_execution to reduce Python overhead
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        steps_per_execution=50  # Process multiple batches per execution
    )
    
    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=1
    )

def train_fast_model(csv_path, image_dir):
    """Main training function with all optimizations"""
    # Set memory growth to avoid OOM errors
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    # Enable XLA optimization
    tf.config.optimizer.set_jit(True)
    
    # Create optimized data pipeline
    train_ds, val_ds, num_classes = optimize_data_pipeline(csv_path, image_dir)
    
    # Create and train the model
    model = create_lightweight_model(num_classes)
    history = compile_and_train(model, train_ds, val_ds)
    
    return model, history

# Usage example:
if __name__ == "__main__":
    csv_path = 'Final Fashion Dataset.csv'
    image_dir = 'images/images'
    
    model, history = train_fast_model(csv_path, image_dir)