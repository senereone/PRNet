m_database_path = "E:\yangying\comparedmethod\database/"

m_dis_type = "img300_kodak_A07_4"
m_quality_range = None # "high", "median", "low", None
m_gpu = '5'

if m_dis_type in ['img300_kodak_A11_15', 'img300_kodak_A11_25', 'img300_kodak_A11_45',"img300_kodak_A11_3",
                  'img300_kodak_A07_3', 'img300_kodak_A07_4', 'img300_kodak_A07_5', 'img300_kodak_A07_2',
                  'img300_kodak_A22_1', 'img300_kodak_A22_2', 'img300_kodak_A22_3', 'set5_A22_1', 'set5_A22_2',
                  'set5_A22_3', 'set12_A22_1', 'set12_A22_2', 'set12_A22_3',
                  'set5_A07_3', 'set5_A07_4', 'set5_A07_5', 'set5_A07_2', 'set12_A07_3', 'set12_A07_4', 'set12_A07_5',
                  ' set12_A07_2',
                  'set5_A11_15', 'set5_A11_25', 'set5_A11_45', "set5_A11_3",
                  'set12_A11_15', 'set12_A11_25', 'set12_A11_45', "set12_A11_3",
                  'peid20_A11_3', "peid20_A11_15", "peid20_A11_25",
                  'peid20_A22_1', 'peid20_A22_2', 'peid20_A22_3']: # gray
    m_c_dim = 1
    m_is_gray = True
else:
    m_c_dim = 3
    m_is_gray = False

class DatasetConfig:
    dis_type = m_dis_type
    database_path = m_database_path
    quality_range = m_quality_range
    is_gray = m_is_gray

    # img300_path
    img300_dis_type = ['img300_kodak_A11_15', 'img300_kodak_A11_25', 'img300_kodak_A11_45', "img300_kodak_A11_3", 'img300_kodak_A07_2',
                       'img300_kodak_A22_1', 'img300_kodak_A22_2', 'img300_kodak_A22_3', 'img300_kodak_A07_3', 'img300_kodak_A07_4', 'img300_kodak_A07_5',]
    img300_path = database_path + "img300_kodak24/"

    # set5_path
    set5_dis_type = ['set5_A11_15', 'set5_A11_25', 'set5_A11_45', "set5_A11_3", 'set5_A22_1', 'set5_A22_2',
                     'set5_A22_3', 'set5_A07_3', 'set5_A07_4', 'set5_A07_5', 'set5_A07_2']
    set5_path = database_path + "set5/"

    # set12_path
    set12_dis_type = ['set12_A11_15', 'set12_A11_25', 'set12_A11_45', "set12_A11_3", 'set12_A22_1', 'set12_A22_2',
                      'set12_A22_3', 'set12_A07_3', 'set12_A07_4', 'set12_A07_5', 'set12_A07_2']
    set12_path = database_path + "set12/"

    is_norm = True
    train_split_ratio = 0.8

    out_dir = "./dataset_out/" + m_dis_type
    split_info = out_dir + "/split.txt"
    train_meta = out_dir + "/train.txt"
    # train_test_meta = out_dir + "/train_test.txt"
    test_meta = out_dir + "/test.txt"


class ModelConfig:
    gpu = m_gpu
    input_size = 32
    input_c_dim = m_c_dim
    output_c_dim = m_c_dim
    input_shape = [input_size, input_size, input_c_dim]


class TrainConfig:
    gpu = m_gpu

    batch_size = 64
    train_test_batch_size = 256
    train_test_per_epoch = 5
    train_test_step = 24

    sample_patch_num = 256
    epochs = 401
    max_to_keep = 5
    snapshot_epoch = 5

    learning_rate = 1e-4
    lr_decay_steps = 200 * 1200
    lr_decay_rate = 0.5

    net_repeat_num = 4

    pretrained_ckpt = None



class TestConfig:
    batch_size = 256
    train_test_batch_size = 256
    sample_patch_num = 256
    learning_rate = 1e-4
    net_repeat_num = 4

    test_model_path = "E:\yangying\python\codeyy_all_loss\log\img300_kodak_A07_4/2021-03-08_00-13-21_False_lapSRN_fuse1_4\model/"
    test_model = test_model_path + "model.ckpt-400"

    out_dir = "E:\yangying\python\codeyy_all_loss/dataset_out/set12_A07_4/"
    test_meta = out_dir + "/test.txt"





