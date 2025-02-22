from keras.src.saving import load_model
import json
import numpy as np
import pickle
from matplotlib import pyplot as plt

# model = load_model('model.h5')
def loadmodel():
    with open('history.pkl', 'rb') as f:
        loaded_history = pickle.load(f)

    # 重新繪製 loss/accuracy 圖表
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(loaded_history['accuracy'], label='Train Accuracy')
    plt.plot(loaded_history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training & Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(loaded_history['loss'], label='Train Loss')
    plt.plot(loaded_history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training & Validation Loss')
    plt.savefig('efficientnetb0_1.png', dpi=300, bbox_inches='tight')

    plt.show()

if __name__ == '__main__':
    loadmodel()