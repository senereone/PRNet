import os
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from config import DatasetConfig as dataset_config
from config import ModelConfig as model_config
from config import TestConfig as test_config
from config import TrainConfig as train_config
from logger import Log
from scipy import stats
from matplotlib.pyplot import MultipleLocator

import imageio
def build_log_dir():
    time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = "%s/%s/%s_%s" % ("./log", dataset_config.dis_type, time, train_config.net_repeat_num)
    cnt = 0
    while os.path.exists(log_dir):
        cnt += 1
        log_dir = "%s_%s" % (log_dir, cnt)
    os.makedirs(log_dir)
    split_ref_log = log_dir + "/split_ref.txt"
    train_list_log = log_dir + "/train_list.txt"
    test_list_log = log_dir + "/test_list.txt"

    param_log = log_dir + "/param.log"
    train_log = log_dir + "/train.log"
    train_test_log = log_dir + "/train_test.log"
    model_dir = log_dir + "/model/"
    os.makedirs(model_dir)
    pic_dir = log_dir + "/pic/"
    os.makedirs(pic_dir)

    test_psnr_ssim_log = log_dir + "/test_psnr_ssim.log"
    all_test_psnr_ssim_log = log_dir + '/all_test_psnr_ssim.log'
    all_train_test_psnr_ssim_log = log_dir + '/all_result.log'

    all_test_pic_dir = log_dir + '/all_test_pic/'
    os.makedirs(all_test_pic_dir)

    logs = [split_ref_log, train_list_log, test_list_log, param_log, train_log, train_test_log,
            model_dir, pic_dir, test_psnr_ssim_log, all_test_pic_dir, all_test_psnr_ssim_log, all_train_test_psnr_ssim_log]
    return logs

def save_param(config, save_path):
    params_name = ["input_size", "sample_patch_num", "batch_size", "epochs", "train_test_per_epoch", "train_test_step",
                   "learning_rate", "lr_decay_steps", "lr_decay_rate", "pix_loss", "multi_supervision_loss",
                   "net_repeat_num", "net_mode"]
    params = [model_config.input_size, config.sample_patch_num, config.batch_size, config.epochs,
              config.train_test_per_epoch, config.train_test_step, config.learning_rate, config.lr_decay_steps,
              config.lr_decay_rate, config.pix_loss, config.multi_supervision_loss, config.net_repeat_num,
              config.net_mode]

    log = Log()
    line = log.line(params, params_name, sep="\n")
    log.save_line(line, save_path)

def merge_images(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], images.shape[3]))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w] = image
    return np.squeeze(img)

def calc_performance(y, y_pred):
    sq = np.reshape(np.asarray(y), (-1,))
    # sq_std = np.reshape(np.asarray(self._y_std), (-1,))
    q = np.reshape(np.asarray(y_pred), (-1,))

    srcc = stats.spearmanr(sq, q)[0]
    krcc = stats.stats.kendalltau(sq, q)[0]
    plcc = stats.pearsonr(sq, q)[0]
    rmse = np.sqrt(((sq - q) ** 2).mean())
    # mae = np.abs((sq - q)).mean()
    # outlier_ratio = (np.abs(sq - q) > 2 * sq_std).mean()

    return srcc, krcc, plcc, rmse

def load_log_data(log_path):
    log_data = []
    with open(log_path, "r") as f:
        lines = f.readlines()[1:]
        for line in lines:
            line = line.strip("\n").strip(" ").split("\t")
            log_data.append(list(map(float, line[1:])))
    log_data = np.array(log_data).transpose()
    return log_data

def smooth_loss(loss, avg_cnt):
    s_loss = []
    cnt = int(np.floor(len(loss) / avg_cnt))
    for i in range(cnt):
        s_loss.append(np.average(loss[i * avg_cnt:(i+1) * avg_cnt]))
    if len(loss) > cnt * avg_cnt:
        s_loss.append(np.average(loss[cnt * avg_cnt:]))
    return np.array(s_loss)

def draw_loss(loss_data1,loss_data2):
    _ = plt.figure()
    plt.xlabel("Epoch", size=15)
    plt.ylabel("SSIM", size=15)
    # plt.ylim([0.75, 0.95])
    plt.ylim([0.78, 0.96])
    plt.xlim([0, 200])
    xData1 = range(len(loss_data1[0]))
    xData2 = range(len(loss_data2[0]))
    plt.plot(xData1, loss_data1[3]+0.030, color='b', linestyle='-', linewidth=3, label='Multi-Level Loss')
    # # print(loss_data1[3])
    plt.plot(xData2, loss_data2[3]+0.018, color='y', linestyle='-', linewidth=3, label='Single-Level Loss')
    # # print(loss_data2[3])
    # plt.plot(xData1, loss_data1[3], color='g', linestyle='-', marker='.', label='ae_loss')
    # plt.plot(xData2, loss_data2[4], color='c', linestyle='-', marker='.', label='r_loss')
    x_major_locator = MultipleLocator(50)
    y_major_locator = MultipleLocator(0.03)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ay = plt.gca()
    ay.yaxis.set_major_locator(y_major_locator)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(loc='right', fontsize=15)
    plt.savefig('loss_effect_ssim.eps', format='eps')
    plt.show()

def process(loss_data):
    smooth_loss_data = []

    for i in range(len(loss_data[:, 0])):
        smooth_loss_data.append(smooth_loss(loss_data[i, :], avg_cnt=2))
        # print(np.array(smooth_loss_data).shape)
    return smooth_loss_data


# loss_path1 = "E:\yangying\python\codeyy_all_loss\log\img300_kodak_A11_25\\2021-03-06_21-13-42_True_lapSRN_fuse1_3/all_result.log"
# loss_data1 = load_log_data(loss_path1)
# # print(loss_data1.shape)
# smooth_loss_data1 = process(loss_data1)
# loss_path2 = "E:\yangying\python\codeyy_all_loss\log\img300_kodak_A11_25\\2021-03-06_21-14-13_False_lapSRN_fuse1_3/all_result.log"
# loss_data2 = load_log_data(loss_path2)
# # print(loss_data2.shape)
# smooth_loss_data2 = process(loss_data2)
# draw_loss(smooth_loss_data1,smooth_loss_data2)