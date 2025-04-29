from keras import Model
from keras.src.applications.efficientnet import EfficientNetB0
from keras.src.applications.resnet import preprocess_input
from keras.src.callbacks import ReduceLROnPlateau,EarlyStopping
from keras.src.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import numpy as np
import pickle
from keras.src.regularizers import L2
from tensorflow.keras.utils import plot_model

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, fbeta_score
import seaborn as sns

batch_size = 128
def visiual(generator):
    batch_images, batch_labels = next(generator)
    mean = np.array([123.68, 116.779, 103.939])

    # 顯示前 8 張圖片
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    for i in range(8):
        img = batch_images[i]  # 取得第 i 張圖片
        img = img + mean  # 加回均值
        img = np.clip(img, 0, 255).astype(np.uint8)  # 限制範圍並轉成 uint8
        img = img[..., ::-1]  # BGR 轉回 RGB
        axes[i].imshow(img)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('output.png')
    plt.show()


def lr_scheduler(epoch, lr):
    return float(0.1 if epoch < 100 else 0.001)


def main():
    train_path = 'datasets/train'
    val_path = 'datasets/val'
    test_path = 'datasets/test'
    img_size = 224

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        horizontal_flip=True,
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],  # 加入亮度變化
        channel_shift_range=30.0,  # 加入顏色變化
        shear_range = 0.2,  # 新增剪切變換

    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical'
    )

    visiual(train_generator)

    val_generator = val_datagen.flow_from_directory(
        val_path,  # 確保驗證集與測試集不同
        target_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    base_model = EfficientNetB0(weights='models/efficientnetb0_notop.h5', include_top=False, input_shape=(img_size, img_size, 3))
    # 讓前 200 層凍結，後面解凍
    for layer in base_model.layers[:200]:
        layer.trainable = False
    for layer in base_model.layers[200:]:
        layer.trainable = True


    # 添加自訂分類層
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(
        512,
        activation='relu',
        kernel_regularizer=L2(0.0001)  # L2 正則化
    )(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(
        2,
        activation='softmax',
        kernel_regularizer=L2(0.0001)  # L2 正則化
    )(x)
    model = Model(inputs=base_model.input, outputs=output_layer)

    model.summary()
    # plot_model(model, to_file='model_structure.png', show_shapes=True, show_layer_names=True)

    model_part = Model(inputs=model.input, outputs=model.layers[10].output)
    plot_model(model_part, to_file='./images/model_part.png', show_shapes=True)
    lr_callback = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )


    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 載入模型

    # 繼續進行推論或訓練
    epochs = 50
    history = model.fit(train_generator, validation_data=val_generator, epochs=epochs, callbacks=[lr_callback,early_stop])
    with open('./models/history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    model.save_weights('efficientnetb1.weights.h5')
    model.save('efficientnetb1.h5')
    # 繪製 accuracy 和 loss 圖
    plt.figure(figsize=(12, 5))  # 新增
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training & Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training & Validation Loss')

    plt.show()
    plt.savefig('efficientnetb1.jpg')
    # 進行測試資料的預測
    print("\n--- 評估測試資料 ---")
    test_steps = len(test_generator)
    predictions = model.predict(test_generator, steps=test_steps)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # 顯示分類報告
    print("Classification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_labels))

    # 計算混淆矩陣
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig('./images/efficient_b1_confusion_matrix.png')
    plt.show()

    # 計算 Accuracy
    accuracy = accuracy_score(true_classes, predicted_classes)
    print(f"Accuracy: {accuracy:.4f}")

    # 計算 Precision
    precision = precision_score(true_classes, predicted_classes, average='weighted')
    print(f"Precision: {precision:.4f}")

    # 計算 Recall
    recall = recall_score(true_classes, predicted_classes, average='weighted')
    print(f"Recall: {recall:.4f}")

    # 計算 F1 Score
    f1 = f1_score(true_classes, predicted_classes, average='weighted')
    print(f"F1 Score: {f1:.4f}")

    # 計算 F2 Score
    f2 = fbeta_score(true_classes, predicted_classes, beta=2, average='weighted')
    print(f"F2 Score: {f2:.4f}")



if __name__ == '__main__':
    main()
