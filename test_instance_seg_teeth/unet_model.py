"""
U-Net模型定义，用于牙齿分割
基于Instance_seg_teeth仓库的OralBBNet架构
"""
import tensorflow as tf
from tensorflow import keras


def get_model(img_size=(512, 512), num_classes=32, drop_rate=0.12):
    """
    创建OralBBNet模型 (YOLOv8 + U-Net)

    Args:
        img_size: 输入图像大小 (高, 宽)
        num_classes: 分类数量 (牙齿类别数)
        drop_rate: Dropout率

    Returns:
        keras.Model: 编译好的模型
    """
    inputs = keras.Input(shape=img_size + (35,))

    # 分离图像和边界框输入
    inputs0 = inputs[:, :, :, 32:]  # 图像数据 (3通道)
    inputs1 = inputs[:, :, :, :32]  # 边界框数据 (32通道)

    skip_connections = []  # 存储编码器的特征图
    bb_out = []  # 存储边界框处理层的输出

    # Entry block - 编码器开始
    x = keras.layers.Conv2D(64, 3, strides=1, padding="same")(inputs0)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.SpatialDropout2D(drop_rate)(x)

    x = keras.layers.Conv2D(64, 3, strides=1, padding="same")(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.SpatialDropout2D(drop_rate)(x)

    skip_connections.append(x)

    # 编码器 - 下采样
    for filters in [128, 256, 512, 1024]:
        x = keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)
        x = keras.layers.Conv2D(filters, 3, padding="same")(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.SpatialDropout2D(drop_rate)(x)

        x = keras.layers.Conv2D(filters, 3, padding="same")(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.SpatialDropout2D(drop_rate)(x)
        skip_connections.append(x)

    # 边界框处理分支
    bb = tf.keras.layers.MaxPool2D(pool_size=(1, 1), strides=1)(inputs1)
    bb = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same")(bb)
    bb = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same",
                                activation=tf.nn.sigmoid)(bb)
    bb_out.append(bb)

    for idx, filters in enumerate([128, 256, 512]):
        bb = tf.keras.layers.MaxPool2D(pool_size=(pow(2, idx+1), pow(2, idx+1)),
                                       strides=pow(2, idx+1))(inputs1)
        bb = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding="same")(bb)
        bb = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding="same",
                                    activation=tf.nn.sigmoid)(bb)
        bb_out.append(bb)

    skip_connections.pop()  # 移除最后一个skip connection（因为不用在解码器中）

    # 解码器 - 上采样
    for filters in [512, 256, 128]:
        x = keras.layers.Conv2DTranspose(filters, 3, strides=2, padding="same")(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.SpatialDropout2D(drop_rate)(x)

        skip_connection = skip_connections.pop()
        bb_layer = bb_out.pop()
        out = tf.multiply(skip_connection, bb_layer)  # 边界框引导的特征调制
        x = keras.layers.concatenate([x, out])

        x = keras.layers.Conv2D(filters, 3, padding="same")(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.SpatialDropout2D(drop_rate)(x)

        x = keras.layers.Conv2D(filters, 3, padding="same")(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.SpatialDropout2D(drop_rate)(x)

    # 最后一个解码器块
    filters = 64
    x = keras.layers.Conv2DTranspose(filters, 3, strides=2, padding="same")(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.SpatialDropout2D(drop_rate)(x)

    skip_connection = skip_connections.pop()
    bb_layer = bb_out.pop()
    out = tf.multiply(skip_connection, bb_layer)
    x = keras.layers.concatenate([x, out])

    x = keras.layers.Conv2D(filters, 3, padding="same")(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.SpatialDropout2D(drop_rate)(x)

    x = keras.layers.Conv2D(filters, 3, padding="same")(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.BatchNormalization()(x)

    # 输出层
    outputs = keras.layers.Conv2D(num_classes, 1, activation="softmax", padding="same")(x)

    model = keras.Model(inputs, outputs)
    return model


def dice_coef(target, predicted, epsilon=1e-7):
    """
    计算Dice系数

    Args:
        target: 真实标签
        predicted: 预测结果
        epsilon: 平滑项

    Returns:
        Dice系数
    """
    predicted = tf.where(predicted < 0.51, 0.00, 1.00)
    intersection = tf.reduce_sum(predicted * target, axis=[1, 2])
    predicted_square = tf.square(predicted)
    target_square = tf.square(target)
    union = tf.reduce_sum(predicted_square, axis=[1, 2]) + tf.reduce_sum(target_square, axis=[1, 2])
    dice = (2 * intersection + epsilon) / (union + epsilon)
    mean_dice_loss = -tf.reduce_mean(dice)
    return -mean_dice_loss


def dice_loss_with_l2_regularization(target, predicted, epsilon=1e-7, l2_weight=0.1):
    """
    带L2正则化的Dice损失

    Args:
        target: 真实标签
        predicted: 预测结果
        epsilon: 平滑项
        l2_weight: L2正则化权重

    Returns:
        总损失
    """
    intersection = tf.reduce_sum(predicted * target, axis=[1, 2])
    predicted_square = tf.square(predicted)
    target_square = tf.square(target)
    union = tf.reduce_sum(predicted_square, axis=[1, 2]) + tf.reduce_sum(target_square, axis=[1, 2])
    dice = (2 * intersection + epsilon) / (union + epsilon)
    mean_dice_loss = tf.reduce_mean(dice)

    l2_norm = tf.reduce_sum(tf.square(predicted - target), axis=[1, 2])
    l2_regularization = l2_weight * tf.reduce_mean(l2_norm)

    total_loss = mean_dice_loss + l2_regularization
    return total_loss
