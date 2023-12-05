import numpy as np
from PIL import Image


# col_img_id = 'ImageId'
# mask_rle = 'EncodedPixels'
# col_height = 'Height'
# col_width = 'Width'
# col_class_id = 'ClassId'
# col_ = 'AttributesIds'


def rle_decode(param_mask_rle, shape):
    """
    解码Run-Length Encoded格式的掩码。

    Parameters:
    - param_mask_rle (str): RLE格式的掩码字符串
    - shape (tuple): 输出掩码的形状 (height, width)

    Returns:
    - mask (numpy.ndarray): 解码后的二进制掩码数组
    """
    s = param_mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape)


# test methods
def get_segmentation_mask(binary_mask):
    result = []
    row_num, col_num = binary_mask.shape
    print(f"row_num: {row_num}, col_num: {col_num}")
    for i in range(1, row_num):
        for j in range(1, col_num):
            i_ratio = float(i) / row_num
            j_ratio = float(j) / col_num
            if binary_mask[i][j] != binary_mask[i - 1][j]:
                result.append(i_ratio)
                result.append(j_ratio)
            elif binary_mask[i][j] != binary_mask[i][j - 1]:
                result.append(i_ratio)
                result.append(j_ratio)
    return result


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
