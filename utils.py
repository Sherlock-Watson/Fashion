import numpy as np
from PIL import Image


# col_img_id = 'ImageId'
# mask_rle = 'EncodedPixels'
# col_height = 'Height'
# col_width = 'Width'
# col_class_id = 'ClassId'
# col_ = 'AttributesIds'


def rle_decode(mask_rle, shape):
    """
    解码Run-Length Encoded格式的掩码。

    Parameters:
    - param_mask_rle (str): RLE格式的掩码字符串
    - shape (tuple): 输出掩码的形状 (height, width)

    Returns:
    - mask (numpy.ndarray): 解码后的二进制掩码数组
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape)


def get_segmentation_mask(mask_rle, shape):
    binary_mask = rle_decode(mask_rle, shape)
    result = []
    row_num, col_num = shape
    print(f"row_num: {row_num}, col_num: {col_num}")
    for i in range(1, row_num):
        for j in range(1, col_num):
            i_ratio = float(i) / row_num
            j_ratio = float(j) / col_num
            if binary_mask[i][j] != binary_mask[i - 1][j] or binary_mask[i][j] != binary_mask[i][j - 1]:
                result.append(i_ratio)
                result.append(j_ratio)
    return result


def save_label(file_path, mask_rle, shape, class_id):
    result_list = get_segmentation_mask(mask_rle, shape)
    content = f"{class_id} {' '.join([str(f) for f in result_list])}\n"
    with open(file_path, mode='a') as file:
        file.write(content)


# test methods
def test_segmentation_mask(binary_mask, result_list):
    original_image = Image.fromarray(binary_mask * 255, "L").convert('RGB')
    print(original_image.size)
    x_list, y_list = [np.asarray(x, dtype=int) for x in (result_list[0:][::2], result_list[1:][::2])]
    pixels = original_image.load()
    for x, y in zip(x_list, y_list):
        pixels[y, x] = (255, 0, 0)
    original_image.save("output/image.png")


def test_segmentation_mask_on_original_file(file_path, result_list):
    original_image = Image.open(file_path)
    print(f"original_image.size={original_image.size}")
    x_list, y_list = [np.asarray(x, dtype=int) for x in (result_list[0:][::2], result_list[1:][::2])]
    pixels = original_image.load()
    for x, y in zip(x_list, y_list):
        pixels[x, y] = (255, 0, 0)
    original_image.save("output/image.png")


def test_label_file(file_path, shape):
    with open(file_path, mode="r") as file:
        content = file.readline()
    value_list = content.split()
    x_list, y_list = [np.asarray(x, dtype=float) for x in (value_list[1:][::2], value_list[2:][::2])]
    img = np.ones(shape, dtype=np.uint8)
    for x, y in zip(x_list, y_list):
        x_index = int(x * shape[1])
        y_index = int(y * shape[0])
        img[y_index][x_index] = 0
    image = Image.fromarray(img * 255)
    image.save("output/mask_boundary_line.png")
