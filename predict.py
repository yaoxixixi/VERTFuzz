import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
import tensorflow_addons as tfa


# 设置随机种子
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 超参数
image_size = 340  # 设置输入图片的大小
num_classes = 2   # 类别数，根据训练时的设定调整

class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):  # 增加 **kwargs 以接受额外参数
        super(Patches, self).__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        config = super(Patches, self).get_config()
        config.update({
            "patch_size": self.patch_size,
        })
        return config


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):  # 增加 **kwargs 以接受额外参数
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded_patches = self.projection(patch) + self.position_embedding(positions)
        return encoded_patches

    def get_config(self):
        config = super(PatchEncoder, self).get_config()
        config.update({
            'num_patches': self.num_patches,
            'projection_dim': self.projection.units,  # projection_dim是Dense层的单位数
        })
        return config

# 加载模型时注册自定义层
def load_trained_model(model_path):
    return load_model(model_path, custom_objects={"Patches": Patches, "PatchEncoder": PatchEncoder})

# 预处理输入图片
def preprocess_image(img_path, img_size):
    img = Image.open(img_path).convert("RGB")  # 确保是RGB格式
    img = img.resize((img_size, img_size))  # 调整为适应输入的尺寸
    img_array = np.array(img) / 255.0  # 归一化到[0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # 添加batch维度
    return img_array

# 预测函数
def predict_image(model, img_path):
    img_array = preprocess_image(img_path, image_size)
    predictions = model.predict(img_array)  # 获取预测结果
    predicted_class = np.argmax(predictions, axis=1)[0]  # 获取预测类别索引
    return predicted_class, predictions

# 可视化预测结果
def visualize_prediction(img_path, predicted_class, predictions):
    img = Image.open(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted: Class {predicted_class}, Confidence: {predictions[0][predicted_class]:.4f}")
    plt.show()

if __name__ == "__main__":
    # 加载训练好的模型
    model_path = "vit_model_best.h5"  # 替换为训练好的模型路径
    model = load_trained_model(model_path)

    # 输入图片路径
    img_path = "1.png"  # 替换为待预测的图片路径

    # 进行预测
    predicted_class, predictions = predict_image(model, img_path)

    # 输出预测结果
    print(f"预测的类别索引: {predicted_class}")
    print(f"预测的类别概率: {predictions}")

    # 可视化预测结果
    visualize_prediction(img_path, predicted_class, predictions)
