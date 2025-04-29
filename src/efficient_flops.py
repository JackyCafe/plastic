import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import numpy as np
import pickle
from tensorflow.keras.regularizers import L2
from tensorflow.keras.utils import plot_model


from tensorflow.compat.v1 import profiler
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

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


def get_flops_profiler(model, input_shape):
    concrete_func = tf.function(model).get_concrete_function(
        tf.TensorSpec(input_shape, tf.float32)
    )

    # Save model
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    frozen_func = convert_variables_to_constants_v2(concrete_func)
    graph_def = frozen_func.graph.as_graph_def()

    # Profile
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
        return flops.total_float_ops


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

    # 計算 FLOPs
    input_shape = (1, img_size, img_size, 3)  # 定義單個輸入的形狀
    flops_profiler = get_flops_profiler(model, input_shape)
    print(f"Estimated FLOPs (using Profiler): {flops_profiler / 1e9:.2f} GFLOPs")

if __name__ == '__main__':
    main()