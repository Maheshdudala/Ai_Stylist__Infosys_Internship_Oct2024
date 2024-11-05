import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type: ignore
from tensorflow.keras.models import Model#type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout#type: ignore
from tensorflow.keras.applications import EfficientNetB0#type: ignore
from tensorflow.keras.optimizers import Adam#type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau#type: ignore
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
tf.random.set_seed(42)

def prepare_data(csv_path, image_dir):
    """Prepare the dataset with train/validation/test split"""
    data = pd.read_csv(csv_path)
    data['Image_Path'] = data['filename'].apply(lambda x: os.path.join(image_dir, x))
    
    # Create stratified train/validation/test split
    train_val_data, test_data = train_test_split(
        data, test_size=0.1, stratify=data['subCategory'], random_state=42
    )
    train_data, val_data = train_test_split(
        train_val_data, test_size=0.2, stratify=train_val_data['subCategory'], random_state=42
    )
    
    return train_data, val_data, test_data

def create_data_generators(train_data, val_data, test_data):
    """Create data generators with augmentation"""
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_test_datagen = ImageDataGenerator(rescale=1.0/255)

    generators = {}
    for name, data, gen in [
        ('train', train_data, train_datagen),
        ('val', val_data, val_test_datagen),
        ('test', test_data, val_test_datagen)
    ]:
        generators[name] = gen.flow_from_dataframe(
            data,
            x_col='Image_Path',
            y_col='subCategory',
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            shuffle=(name == 'train')
        )
    
    return generators

def build_model(num_classes, learning_rate=0.001):
    """Build and compile the model"""
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # First, freeze the base model
    base_model.trainable = False
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

def create_callbacks(checkpoint_path):
    """Create training callbacks"""
    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        )
    ]
    return callbacks

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'])
    
    # Plot loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'])
    
    plt.tight_layout()
    plt.show()

def train_model(csv_path, image_dir, checkpoint_path):
    """Main training function"""
    # Prepare data
    train_data, val_data, test_data = prepare_data(csv_path, image_dir)
    generators = create_data_generators(train_data, val_data, test_data)
    
    # Build model
    num_classes = len(train_data['subCategory'].unique())
    model, base_model = build_model(num_classes)
    
    # Train with frozen layers
    print("Training with frozen layers...")
    callbacks = create_callbacks(checkpoint_path)
    history1 = model.fit(
        generators['train'],
        validation_data=generators['val'],
        epochs=10,
        callbacks=callbacks
    )
    
    # Fine-tune the model
    print("Fine-tuning the model...")
    base_model.trainable = True
    for layer in base_model.layers[:-20]:  # Freeze all but the last 20 layers
        layer.trainable = False
    
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history2 = model.fit(
        generators['train'],
        validation_data=generators['val'],
        epochs=5,
        callbacks=callbacks
    )
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(generators['test'])
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    
    # Plot training history
    plot_training_history(history2)
    
    return model, history1, history2

# Usage example:
if __name__ == "__main__":
    csv_path = 'Final Fashion Dataset.csv'
    image_dir = 'images/images'
    checkpoint_path = 'best_model.keras'
    
    model, history1, history2 = train_model(csv_path, image_dir, checkpoint_path)