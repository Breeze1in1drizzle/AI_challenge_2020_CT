# coding=utf-8
#################################################
# fine tune
#################################################
import pandas as pd
import numpy as np
import os

import datetime
from tqdm import tqdm
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint


from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import Adam, SGD

from keras.applications.densenet import DenseNet169, DenseNet121
from keras.applications.densenet import preprocess_input
from PIL import Image, ImageFont, ImageDraw

#######################################
# 在训练的时候置为1
from keras import backend as K

K.set_learning_phase(1)     #
#######################################

EPOCHS = 60
RANDOM_STATE = 2018
learning_rate = 0.001  # 0.001

TRAIN_DIR = '../data/data_for_train'
VALID_DIR = '../data/data_for_valid'


# 获得每一类的图片张数，索引顺序与keras中的读取顺序对应
# 大文件夹下，每一个子文件夹内的图片的数量
def get_cls_num(directory):
    '''
        获得每一类的图片张数，索引顺序与keras中的读取顺序对应
    '''
    classes = list()
    # 遍历一次文件夹的子文件夹，将名字append至名为classes的list结构中
    for subdir in sorted(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, subdir)):
            classes.append(subdir)

    # 大文件夹下，每一个子文件夹内的图片的数量
    classes_num = list()
    # 遍历一遍classed，也就是遍历一次各个子文件夹
    for cls_ in classes:
        path = os.path.join(directory, cls_)
        # os.listdir()方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。（返回列表）
        pics = os.listdir(path)
        classes_num.append(len(pics))   # 获取图片数量(列表长度)
    return classes_num


def get_classes_indice(directory):
    '''
        得到索引与label对应的dict
    '''
    classes = []
    for subdir in sorted(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, subdir)):
            classes.append(subdir)
    num_classes = len(classes)
    class_indices = dict(zip(classes, range(len(classes))))
    return class_indices


def get_callbacks(filepath, patience=2):
    lr_reduce = ReduceLROnPlateau(
        monitor='val_acc', factor=0.1, epsilon=5e-6,
        patience=patience, verbose=1, min_lr=0.00001
    )

    # 该回调函数将在每个epoch后保存模型到 filepath
    msave = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True)
    earlystop = EarlyStopping(monitor='val_acc', min_delta=0, patience=patience * 3 + 4, verbose=1, mode='auto')
    return [lr_reduce, msave, earlystop]


def add_new_last_layer(base_model, nb_classes, drop_rate=0.):
    """
    Add last layer to the conv-net
    Args:
        base_model: keras model excluding top       之前是把最高层进行了冻结吗？
        nb_classes: # of classes
    Returns:
        new keras model with last layer
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x = base_model.output
    # x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,name='bn_final')(x)
    x = Dropout(0.5)(x)
    x = GlobalAveragePooling2D()(x)     # 增加了drop out以及池化层，最后加上dense全连接，包含softmax预测。
    predictions = Dense(nb_classes, activation='softmax')(x)  # new softmax layer
    model = Model(input=base_model.input, output=predictions)   # 定义模型的输入输出
    return model


# LSR(label smoothing regularization)
def mycrossentropy(e=0.25, nb_classes=11):
    '''
        https://spaces.ac.cn/archives/4493
    '''
    # 交叉熵
    def mycrossentropy_fixed(y_true, y_pred):   # y_true, y_pred ??
        return (
            (1-e)*K.categorical_crossentropy(y_true, y_pred) +
            e*K.categorical_crossentropy(K.ones_like(y_pred) / nb_classes, y_pred)
        )
    return mycrossentropy_fixed


# LSR(label smoothing regularization) + focal_loss
def myloss(classes_num, e=0.25, nb_classes=11):
    def myloss_fixed(y_true, y_pred):
        import tensorflow as tf
        from tensorflow.python.ops import array_ops
        from keras import backend as K

        zeros = array_ops.zeros_like(y_pred, dtype=y_pred.dtype)
        crossentropy_loss = -1 * y_true * tf.log(tf.clip_by_value(y_pred, 1e-8, 1.0))

        classes_weight = array_ops.zeros_like(y_true, dtype=y_true.dtype)

        total_num = float(sum(classes_num))
        classes_w_t1 = [total_num/ff for ff in classes_num]
        sum_ = sum(classes_w_t1)
        classes_w_t2 = [ff/sum_ for ff in classes_w_t1]  # scale
        classes_w_tensor = tf.convert_to_tensor(classes_w_t2, dtype=y_true.dtype)
        classes_weight += classes_w_tensor

        alpha = array_ops.where(y_true > zeros, classes_weight, zeros)

        balanced_fl = alpha * crossentropy_loss
        balanced_fl = tf.reduce_sum(balanced_fl)

        return ((1-e)*balanced_fl + e*K.categorical_crossentropy(K.ones_like(y_pred)/nb_classes, y_pred))
    return myloss_fixed


def get_model(INT_HEIGHT, IN_WIDTH):
    '''
        获得模型
    '''
    import config
    # base_model = DenseNet169(include_top=False, weights='imagenet', input_shape=(INT_HEIGHT,IN_WIDTH, 3))
    base_model_loading_path = config.MODELS_PATH + 'final/densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5'
    print("path: ", base_model_loading_path)
    # base_model_loading_path = '../models/densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5'
    base_model = DenseNet169(
        include_top=False,
        weights=base_model_loading_path,
        input_shape=(INT_HEIGHT, IN_WIDTH, 3)
    )

    # add attention
    # base_model = add_attention(base_model)

    model = add_new_last_layer(base_model, 11)
    model.compile(optimizer=SGD(lr=learning_rate, momentum=0.9), loss=[mycrossentropy()], metrics=['accuracy'])
    # classes_num = get_cls_num('../data/data_for_train')
    # model.compile(optimizer=SGD(lr=learning_rate, momentum=0.9), loss=[myloss(classes_num)], metrics=['accuracy'])
    # model.summary()
    return model


def train_model(save_model_path, BATCH_SIZE, IN_SIZE):
    INT_HEIGHT = IN_SIZE[0]
    IN_WIDTH = IN_SIZE[1]

    callbacks = get_callbacks(filepath=save_model_path, patience=3)
    model = get_model(INT_HEIGHT, IN_WIDTH)

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        horizontal_flip=True,
        vertical_flip=True
    )

    valid_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_generator = train_datagen.flow_from_directory(
        directory=TRAIN_DIR,
        target_size=(INT_HEIGHT, IN_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        seed=RANDOM_STATE,
        interpolation='antialias',  # PIL默认插值下采样的时候会模糊
    )

    valid_generator = valid_datagen.flow_from_directory(
        directory=VALID_DIR,
        target_size=(INT_HEIGHT, IN_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        seed=RANDOM_STATE,
        interpolation='antialias',  # PIL默认插值下采样的时候会模糊
    )

    model.fit_generator(
        train_generator,
        steps_per_epoch=2 * (train_generator.samples // BATCH_SIZE + 1),
        epochs=EPOCHS,
        max_queue_size=100,
        workers=1,
        verbose=1,
        validation_data=valid_generator,  # valid_generator,
        validation_steps=valid_generator.samples // BATCH_SIZE,
        # valid_generator.samples // BATCH_SIZE + 1, #len(valid_datagen)+1,
        callbacks=callbacks
    )


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def letterbox_image(image, size):
    '''
        resize image with unchanged aspect ratio using padding
    '''
    image_w, image_h = image.size
    w, h = size
    new_w = int(image_w*min(w/image_w, h/image_h))
    new_h = int(image_h*min(w/image_w, h/image_h))
    resized_image = image.resize((new_w, new_h), Image.BICUBIC)

    boxed_image = Image.new('RGB', size, (128, 128, 128))
    boxed_image.paste(resized_image, ((w-new_w) // 2, (h-new_h) // 2))      # //是整数除
    return boxed_image


def detect_image(self, image):
    #
    # if self.model_image_size != (None, None):
    #     assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
    #     assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
    #     boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
    # else:
    new_image_size = (image.width - (image.width % 32),
                      image.height - (image.height % 32))
    boxed_image = letterbox_image(image, new_image_size)
    image_data = np.array(boxed_image, dtype='float32')

    print(image_data.shape)
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

    out_boxes, out_scores, out_classes = self.sess.run(
        [self.boxes, self.scores, self.classes],
        feed_dict={
            self.yolo_model.input: image_data,
            self.input_image_shape: [image.size[1], image.size[0]],
            K.learning_phase(): 0
        })

    print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

    font = ImageFont.truetype(
        font='font/FiraMono-Medium.otf',
        size=np.floor(3e-2*image.size[1]+0.5).astype('int32')
    )
    thickness = (image.size[0]+image.size[1]) // 300

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = self.class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top-label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=self.colors[c])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=self.colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

    return image, out_boxes, out_scores, out_classes


def predict(weights_path, IN_SIZE):
    '''
        对测试数据进行预测
    '''
    INT_HEIGHT = IN_SIZE[0]
    IN_WIDTH = IN_SIZE[1]

    K.set_learning_phase(0)

    # test_pic_root_path = './data'
    test_pic_root_path = '../data'
    filename = []
    probability = []
    # 1#得到模型
    model = get_model(INT_HEIGHT, IN_WIDTH)
    model.load_weights(weights_path)

    # #2#预测
    for parent, _, files in os.walk(test_pic_root_path):
        for line in tqdm(files):

            pic_path = os.path.join(test_pic_root_path, line.strip())
            if pic_path == "../data/.DS_Store":
                continue
            print(pic_path)
            img = load_img(
                pic_path, target_size=(INT_HEIGHT, IN_WIDTH, 3),
                interpolation='bilinear'#'antialias'
            )
            img = img_to_array(img)
            img = preprocess_input(img)
            img = np.expand_dims(img, axis=0)
            prediction = model.predict(img, verbose=0)[0]
            print(prediction)
            prediction = list(prediction)
            index = prediction.index(max(prediction))
            max_pro = prediction[index]

            classes_dict = ['norm', '扎洞', '毛斑', '擦洞', '毛洞', '织稀', '吊经', '缺经', '跳花', '油／污渍', '其他']
            label = classes_dict[index]
            filename.append(line.strip() + '|' + label)
            probability.append(' ' + str(max_pro))

    # #3#写入csv
    res_path = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv"
    dataframe = pd.DataFrame({'filename|defect': filename, 'probability': probability})
    dataframe.to_csv(res_path, index=False, header=True)
    predict("res_path: ", res_path)


def main():
    batch_size = 4
    in_size = [550, 600]  # [H,W]
    # weights_path1 = './models/model_weight.hdf5'  # dense169, [550,600]
    # path = '../models/model_weight.hdf5'
    weights_path1 = '../models/model_weight.hdf5'
    predict(weights_path=weights_path1, IN_SIZE=in_size)


if __name__ == '__main__':
    main()
    predict("end")
