import imageio
import numpy as np
from numpy.lib.stride_tricks import as_strided
import glob
import random

def load_list(dir):
    ref_list=[]
    dis_list=[]
    with open(dir,'r') as file:
        lines=file.readlines()
    lines=lines[1:]
    for line in lines:
        items = line.strip("\n").split(" ")
        ref_list.append(items[0])
        dis_list.append(items[1])
    return ref_list, dis_list

def load_data_sample(ref_list, dis_list, sample_shape, sample_num, is_gray=True, shuffle=True):
    ref_data, dis_data, score_data = [], [], []
    for i in range(len(ref_list)):
        ref_img = imageio.imread(ref_list[i])
        dis_img = imageio.imread(dis_list[i])
        ref_img = (ref_img - 127.5) / 127.5
        dis_img = (dis_img - 127.5) / 127.5
        sample_ref, sample_dis = sample(ref_img, dis_img, sample_shape, sample_num)
        ref_data.extend(sample_ref)
        dis_data.extend(sample_dis)
    ref_data = np.array(ref_data).astype(np.float32)
    dis_data = np.array(dis_data).astype(np.float32)
    if is_gray:
        ref_data = ref_data[:, :, :, None]
        dis_data = dis_data[:, :, :, None]
    if shuffle:
        seed = np.random.randint(99999)
        np.random.seed(seed)
        np.random.shuffle(ref_data)
        np.random.seed(seed)
        np.random.shuffle(dis_data)
    return ref_data, dis_data

def load_data_split(ref_list, dis_list, split_shape, split_stride, is_gray=True, shuffle=True):
    ref_data, dis_data = [], []
    if shuffle:
        seed = np.random.randint(99999)
        np.random.seed(seed)
        np.random.shuffle(ref_data)
        np.random.seed(seed)
        np.random.shuffle(dis_data)
    for i in range(len(ref_list)):
        ref_img = imageio.imread(ref_list[i])
        dis_img = imageio.imread(dis_list[i])
        ref_img = (ref_img - 127.5) / 127.5
        dis_img = (dis_img - 127.5) / 127.5
        split_dis, _ = extract_patches(dis_img, split_shape, split_stride)
        split_ref, _ = extract_patches(ref_img, split_shape, split_stride)
        ref_data.extend(split_ref)
        dis_data.extend(split_dis)

    ref_data = np.array(ref_data).astype(np.float32)
    dis_data = np.array(dis_data).astype(np.float32)
    if is_gray:
        ref_data = ref_data[:, :, :, None]
        dis_data = dis_data[:, :, :, None]
    return ref_data, dis_data

def load_data(ref_list, dis_list, split_shape, split_stride, is_gray=True, shuffle=True):
    ref_data, dis_data = [], []
    if shuffle:
        seed = np.random.randint(99999)
        np.random.seed(seed)
        np.random.shuffle(ref_data)
        np.random.seed(seed)
        np.random.shuffle(dis_data)
    for i in range(len(ref_list)):
        ref_img = imageio.imread(ref_list[i])
        dis_img = imageio.imread(dis_list[i])
        ref_img = (ref_img - 127.5) / 127.5
        dis_img = (dis_img - 127.5) / 127.5

        ref_data.append(ref_img)
        dis_data.append(dis_img)

    ref_data = np.array(ref_data).astype(np.float32)
    dis_data = np.array(dis_data).astype(np.float32)
    if is_gray:
        ref_data = ref_data[:, :, :, None]
        dis_data = dis_data[:, :, :, None]
    return ref_data, dis_data

def sample(ref_img, dis_img, shape, num):
    sample_ref, sample_dis, sample_score = [], [], []
    for i in range(num):
        ref_img_crop, dis_img_crop = random_crop(ref_img, dis_img, shape)
        sample_dis.append(dis_img_crop)
        sample_ref.append(ref_img_crop)
    sample_dis = np.array(sample_dis).astype(np.float32)
    sample_ref = np.array(sample_ref).astype(np.float32)
    return sample_ref, sample_dis

def random_crop(ref_image, dis_image, crop_shape):
    h, w = ref_image.shape[0], ref_image.shape[1]
    if h < crop_shape[0] or w < crop_shape[1]:
        print("crop_shape error")
        exit()
    start_h = random.randint(0, h - crop_shape[0])
    start_w = random.randint(0, w - crop_shape[1])
    ref_image_crop = ref_image[start_h:start_h + crop_shape[0], start_w:start_w + crop_shape[1]]
    dis_image_crop = dis_image[start_h:start_h + crop_shape[0], start_w:start_w + crop_shape[1]]
    return ref_image_crop, dis_image_crop

def split_img(batch_ref, batch_dis, split_shape, split_stride):
    split_batch_dis, split_batch_ref = [], []
    batch_size = batch_dis.shape[0]
    for i in range(batch_size):
        dis_img = batch_dis[i]
        ref_img = batch_ref[i]

        split_dis, _ = extract_patches(dis_img, split_shape, split_stride)
        split_batch_dis.extend(split_dis)
        split_ref, _ = extract_patches(ref_img, split_shape, split_stride)
        split_batch_ref.extend(split_ref)

    split_batch_ref = np.array(split_batch_ref).astype(np.float32)
    split_batch_dis = np.array(split_batch_dis).astype(np.float32)
    return split_batch_ref, split_batch_dis

def extract_patches(image, patch_shape, stride):
    image_ndim = image.ndim
    extraction_stride = tuple([stride] * image_ndim)

    patch_indices_shape = ((np.array(image.shape) - np.array(patch_shape)) // np.array(extraction_stride)) + 1
    shape = tuple(list(patch_indices_shape) + list(patch_shape))

    indexing_strides = image.strides * np.array(extraction_stride)
    strides = tuple(list(indexing_strides) + list(image.strides))

    patches = as_strided(image, shape=shape, strides=strides)
    patches = patches.reshape([-1] + list(patch_shape))
    return patches, shape

def sample_img(batch_ref, batch_dis, crop_shape, sample_num):
    sample_batch_ref, sample_batch_dis = [], []
    batch_size = batch_dis.shape[0]
    for i in range(batch_size):
        ref_img = batch_ref[i]
        dis_img = batch_dis[i]

        for j in range(sample_num):
            ref_img_crop, dis_img_crop = random_crop(ref_img, dis_img, crop_shape)
            sample_batch_dis.append(dis_img_crop)
            sample_batch_ref.append(ref_img_crop)
    sample_batch_dis = np.array(sample_batch_dis).astype(np.float32)
    sample_batch_fake_ref = np.array(sample_batch_ref).astype(np.float32)
    return sample_batch_fake_ref, sample_batch_dis


def random_crop1(ref_image, dis_image, crop_shape):
    h, w = ref_image.shape[0], ref_image.shape[1]
    if h < crop_shape[0] or w < crop_shape[1]:
        print("crop_shape error")
        exit()
    start_h = random.randint(0, h - crop_shape[0])
    start_w = random.randint(0, w - crop_shape[1])
    ref_image_crop = ref_image[start_h:start_h + crop_shape[0], start_w:start_w + crop_shape[1]]
    dis_image_crop = dis_image[start_h:start_h + crop_shape[0], start_w:start_w + crop_shape[1]]
    return ref_image_crop, dis_image_crop

def merge_imgs(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], images.shape[3]))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w] = image
    return np.squeeze(img)

def load_files(data_path):
    types = ['bmp', 'png', 'pgm', 'j2k', 'jpg'] # , 'BMP'
    output = []
    for type in types:
        pattern = data_path + "/*." + type
        output.extend(glob.glob(pattern))
    output.sort()
    return output

# if __name__=='__main__':
#     load_list('/media/yangying/senere1T/codes/vsigan/testdata/enc01_testdata/test_list.txt')


