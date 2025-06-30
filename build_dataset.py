import os
import shutil
from read_data import ReadData
from config import DatasetConfig as config

def get_out_path(out_dir, dis_file, index):
    if config.dis_type in config.csiq_dis_type:
        file_name, file_type = os.path.basename(dis_file).split(".p")
    else:
        file_name, file_type = os.path.basename(dis_file).split(".")
    if config.dis_type in ["live"]:
        dis_type = "_" + dis_file.split("/")[-2]
    else:
        dis_type = ""
    file_type = "png"
    out_ref_file = "%sref%s_%s_sample_%d.%s" % (out_dir, dis_type, file_name, index, file_type)
    out_dis_file = "%sdis%s_%s_sample_%d.%s" % (out_dir, dis_type, file_name, index, file_type)
    return out_ref_file, out_dis_file



def check_out_exist():
    if os.path.exists(config.out_dir):
        print("%s exist, please delete it" % config.out_dir)
        #os.removedirs(config.out_dir)
        shutil.rmtree(config.out_dir)
        #exit()
    if os.path.exists(config.train_meta):
        print("%s exist, please delete it" % config.train_meta)
        exit()
    if os.path.exists(config.test_meta):
        print("%s exist, please delete it" % config.test_meta)
        exit()
    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir)

def write_head(): # 写表头
    line = "ref_file_path dis_file_path"
    with open(config.train_meta, "w") as f:
        f.write(line)
    with open(config.test_meta, "w") as f:
        f.write(line)

def write_split(file_name, train_ref_list, test_ref_list):
    train_lines = "\n".join(train_ref_list)
    test_lines = "\n".join(test_ref_list)
    with open(config.split_info, "w") as f:
        f.write("train:\n")
        f.write(train_lines)
        f.write("\n")
        f.write("test:\n")
        f.write(test_lines)

def write_data(file_name, ref_list, dis_list):
    with open(file_name, "a") as f:
        line = "\n%s %s" % (ref_list, dis_list)
        f.write(line)

def write_meta(file_name, ref_file, dis_file):
    with open(file_name, "a") as f:
        line = "\n%s %s" % (ref_file, dis_file)
        f.write(line)

def process():
    read_data = ReadData(config)
    check_out_exist() #检查out文件是否已经生成
    write_head()

    # split ref
    print("generate split ref")
    train_ref_list, test_ref_list = read_data.split_ref()
    write_split(config.split_info, train_ref_list, test_ref_list)

    # get train ref, dis, score
    print("generate train data")
    train_ref_list, train_dis_list, train_dis_idx = read_data.get_ref2dis(train_ref_list)

    # train_ref_list, train_dis_list = read_data.quality_range(train_ref_list, train_dis_list)


    # write_head(config.train_meta)
    for ref, dis in zip(train_ref_list, train_dis_list):
        write_data(config.train_meta, ref, dis)

    print("generate test data")
    test_ref_list, test_dis_list, test_dis_idx = read_data.get_ref2dis(test_ref_list)
    # test_ref_list, test_dis_list = read_data.quality_range(test_ref_list, test_dis_list)

    # write_head(config.test_meta)
    for ref, dis in zip(test_ref_list, test_dis_list):
        write_data(config.test_meta, ref, dis)

if __name__ == "__main__":
    process()
