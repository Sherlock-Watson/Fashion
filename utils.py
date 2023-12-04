import numpy as np

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


# 示例
# param_mask_rle = "1 3 10 5 20 2"  # 替换为实际的RLE格式字符串
# image_shape = (256, 256)  # 替换为实际图像的形状
#
# # 解码RLE格式的掩码
# decoded_mask = rle_decode(param_mask_rle, image_shape)
#
# # 可以选择使用Pillow库将二进制掩码数组保存为图像文件
# mask_image = Image.fromarray(decoded_mask * 255)  # 乘以255将二进制数组转换为灰度图像
# mask_image.save('decoded_mask.png')
