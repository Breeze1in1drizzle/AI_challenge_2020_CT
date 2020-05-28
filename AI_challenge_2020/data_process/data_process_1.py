from PIL import ImageDraw
from PIL import Image

import configuration


def draw_rectangle_on_one_ct_image(label_path=None, image_path=None, id=None, save_path=None):
    '''
    在一个CT图中画矩形方框（标注息肉）
    :param id: CT图片的id号码
    :return:
    '''
    label = open('data/labels_train/IM_' + id + '.txt')
    image = Image.open('data/image_train/IM_' + id + '.jpg')  # 打开一张图片

    i_size = image.size
    i_x = i_size[0]
    i_y = i_size[1]

    type, x, y, w, h = 0, 0, 0, 0, 0

    for line in label:
        lineArr = line.split()
        x = lineArr[1] * i_x
        y = lineArr[2] * i_y
        w = lineArr[3] * i_x
        h = lineArr[4] * i_y
        break       # ?

    min_x = x - w / 2
    min_y = y - w / 2
    max_x = x + w / 2
    max_y = x + w / 2

    draw = ImageDraw.Draw(image)  # 在上面画画
    # draw.rectangle([645,465,200,200], outline=(255,0,0))
    # [左上角x，左上角y，右下角x，右下角y]，outline边框颜色
    draw.rectangle([min_x, min_y, max_x, max_y], outline=(255, 0, 0))  # [左上角x，左上角y，右下角x，右下角y]，outline边
    image.show()
    # image save


if __name__ == "__main__":
    # 识别一张图片
    data_path = configuration.DATA_PATH + "official_data_1/"
    id = "0196"
    label_path = data_path + "labels_train/"
    image_path = data_path + "image_train/"
    save_path = data_path + "image_train_drawnWithRectangle/"
    draw_rectangle_on_one_ct_image(
        label_path=label_path, image_path=image_path,
        id=id, save_path=save_path
    )
