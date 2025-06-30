import os, random
from scipy.io import loadmat
import numpy as np
from data_helper import load_files

class ReadData(object):
    def __init__(self, config):
        self.config = config
        self.dis_type = config.dis_type

    def get_ref_files(self):
        ref_files = None
        if self.dis_type in self.config.ivc_dis_type:
            ref_files = load_files(self.config.ivc_path + "refimg/")
        elif self.dis_type in self.config.peid_dis_type:
            ref_files = load_files(self.config.peid_path + "refimg/")
        elif self.dis_type in self.config.live_dis_type:
            ref_files = load_files(self.config.live_path + "refimg/")
        elif self.dis_type in self.config.tid2013_dis_type:
            ref_files = load_files(self.config.tid2013_path + "refimg/")
        elif self.dis_type in self.config.csiq_dis_type:
            ref_files = load_files(self.config.csiq_path + "refimg/")
        elif self.dis_type in self.config.img300_dis_type:
            ref_files = load_files(self.config.img300_path + "img300_kodak_ref/")
        elif self.dis_type in self.config.set5_dis_type:
            ref_files = load_files(self.config.set5_path + "set5_ref/")
        elif self.dis_type in self.config.set12_dis_type:
            ref_files = load_files(self.config.set12_path + "set12_ref/")
        elif self.dis_type in self.config.peid20_dis_type:
            print(self.config.peid20_path + "peid20_ref/")
            ref_files = load_files(self.config.peid20_path + "peid20_ref/")

        else:
            print('get ref files dis type %s error' % self.dis_type)
            exit()
        return ref_files

    def split_ref(self):
        ref_files = self.get_ref_files()
        print(len(ref_files))
        # random.shuffle(ref_files)
        # num_train_ref = int(np.floor(len(ref_files) * self.config.train_split_ratio))

        train_ref_files = ref_files[:300]
        test_ref_files = ref_files[300:]

        return train_ref_files, test_ref_files

    def get_ref2dis(self, files):
        if self.dis_type in self.config.ivc_dis_type:
            return self.get_ivc_ref2dis(files)
        elif self.dis_type in self.config.peid_dis_type:
            return self.get_peid_ref2dis(files)
        elif self.dis_type in self.config.live_dis_type:
            return self.get_live_ref2dis(files)
        elif self.dis_type in self.config.tid2013_dis_type:
            return self.get_tid2013_ref2dis(files)
        elif self.dis_type in self.config.csiq_dis_type:
            return self.get_csiq_ref2dis(files)
        elif self.dis_type in self.config.img300_dis_type:
            return self.get_img300_ref2dis(files)
        elif self.dis_type in self.config.set5_dis_type:
            return self.get_set5_ref2dis(files)
        elif self.dis_type in self.config.set12_dis_type:
            return self.get_set12_ref2dis(files)
        elif self.dis_type in self.config.peid20_dis_type:
            return self.get_peid20_ref2dis(files)
        else:
            print('get ref2dis dis type %s error' % self.dis_type)
            exit()

    def get_set12_ref2dis(self, ref_lists):
        # 获取参考图像对应的失真图像，以及失真图像在mos中的idx
        if self.dis_type == 'set12':
            dis_path = self.config.set12_path + "disimg/"
        else:
            dis_path = self.config.set12_path + self.dis_type + "/"

        m_dis_list, m_ref_list = [], []
        lists = load_files(dis_path)

        # 获取失真图像在mos中的idx
        idx_file = {}
        for i, file in enumerate(lists):
            idx_file[file] = i

        for ref_file in ref_lists:
            name = os.path.basename(ref_file).split('.')[0]
            dis_file = [file_name for file_name in lists if name in os.path.basename(file_name)]
            ref_file = [ref_file for i in range(len(dis_file))]
            m_dis_list.extend(dis_file)
            m_ref_list.extend(ref_file)
        dis_idx = np.array([idx_file[dis_list] for dis_list in m_dis_list])
        return m_ref_list, m_dis_list, dis_idx

    def get_peid20_ref2dis(self, ref_lists):
        # 获取参考图像对应的失真图像，以及失真图像在mos中的idx
        if self.dis_type == 'peid20':
            dis_path = self.config.peid20_path + "disimg/"
        else:
            dis_path = self.config.peid20_path + self.dis_type + "/"

        m_dis_list, m_ref_list = [], []
        lists = load_files(dis_path)

        # 获取失真图像在mos中的idx
        idx_file = {}
        for i, file in enumerate(lists):
            idx_file[file] = i

        for ref_file in ref_lists:
            name = os.path.basename(ref_file).split('.')[0]
            dis_file = [file_name for file_name in lists if name in os.path.basename(file_name)]
            ref_file = [ref_file for i in range(len(dis_file))]
            m_dis_list.extend(dis_file)
            m_ref_list.extend(ref_file)
        dis_idx = np.array([idx_file[dis_list] for dis_list in m_dis_list])
        return m_ref_list, m_dis_list, dis_idx

    def get_set5_ref2dis(self, ref_lists):
        # 获取参考图像对应的失真图像，以及失真图像在mos中的idx
        if self.dis_type == 'set5':
            dis_path = self.config.set5_path + "disimg/"
        else:
            dis_path = self.config.set5_path + self.dis_type + "/"

        m_dis_list, m_ref_list = [], []
        lists = load_files(dis_path)

        # 获取失真图像在mos中的idx
        idx_file = {}
        for i, file in enumerate(lists):
            idx_file[file] = i

        for ref_file in ref_lists:
            name = os.path.basename(ref_file).split('.')[0]
            dis_file = [file_name for file_name in lists if name in os.path.basename(file_name)]
            ref_file = [ref_file for i in range(len(dis_file))]
            m_dis_list.extend(dis_file)
            m_ref_list.extend(ref_file)
        dis_idx = np.array([idx_file[dis_list] for dis_list in m_dis_list])
        return m_ref_list, m_dis_list, dis_idx

    def get_img300_ref2dis(self, ref_lists):
        # 获取参考图像对应的失真图像，以及失真图像在mos中的idx
        if self.dis_type == 'img300':
            dis_path = self.config.img300_path + "disimg/"
        else:
            dis_path = self.config.img300_path + self.dis_type + "/"

        m_dis_list, m_ref_list = [], []
        lists = load_files(dis_path)

        # 获取失真图像在mos中的idx
        idx_file = {}
        for i, file in enumerate(lists):
            idx_file[file] = i

        for ref_file in ref_lists:
            name = os.path.basename(ref_file).split('.')[0]
            dis_file = [file_name for file_name in lists if name in os.path.basename(file_name)]
            ref_file = [ref_file for i in range(len(dis_file))]
            m_dis_list.extend(dis_file)
            m_ref_list.extend(ref_file)
        dis_idx = np.array([idx_file[dis_list] for dis_list in m_dis_list])
        return m_ref_list, m_dis_list, dis_idx

    def get_ivc_ref2dis(self, ref_lists):
        # 获取参考图像对应的失真图像，以及失真图像在mos中的idx
        if self.dis_type == 'ivc':
            dis_path = self.config.ivc_path + "disimg/"
        else:
            dis_path = self.config.ivc_path + self.dis_type + "/"

        m_dis_list, m_ref_list = [], []
        lists = load_files(dis_path)

        # 获取失真图像在mos中的idx
        idx_file = {}
        for i, file in enumerate(lists):
            idx_file[file] = i

        for ref_file in ref_lists:
            name = os.path.basename(ref_file).split('.')[0]
            dis_file = [file_name for file_name in lists if name in file_name]
            ref_file = [ref_file for i in range(len(dis_file))]
            m_dis_list.extend(dis_file)
            m_ref_list.extend(ref_file)
        dis_idx = np.array([idx_file[dis_list] for dis_list in m_dis_list])
        return m_ref_list, m_dis_list, dis_idx

    def get_peid_ref2dis(self, ref_lists):
        if self.dis_type == 'peid':
            dis_path = self.config.peid_path + "encimg/"
        else:
            dis_path = self.config.peid_path + self.dis_type + "/"
        m_dis_list, m_ref_list = [], []
        lists = load_files(dis_path)

        # 获取图像在mos中的idx
        idx_file = {}
        for i, file in enumerate(lists):
            idx_file[file] = i

        for ref_file in ref_lists:
            name = os.path.basename(ref_file).split('.')[0]
            dis_file = [file_name for file_name in lists if name in file_name]
            ref_file = [ref_file for i in range(len(dis_file))]
            m_ref_list.extend(ref_file)
            m_dis_list.extend(dis_file)
        dis_idx = np.array([idx_file[dis_list] for dis_list in m_dis_list])
        return m_ref_list, m_dis_list, dis_idx

    def get_live_ref2dis(self, ref_lists):
        refnames_path = self.config.live_path + "refnames_all.mat"
        refnames = loadmat(refnames_path)['refnames_all'][0]
        dmos_path = self.config.live_path + "dmos.mat"
        is_ref = loadmat(dmos_path)['orgs'][0]

        files = self.build_live_files()

        m_dis_list, m_ref_list, m_dis_idx = [], [], []
        idex = -1
        for dis_file, filename, is_ref in zip(files, refnames, is_ref):
            idex += 1
            file = self.config.live_path + "refimg/" + filename[0]

            if file in ref_lists and is_ref == 0 and self.config.dis_type in dis_file:
                m_dis_list.append(dis_file)
                m_ref_list.append(file)
                m_dis_idx.append(idex)
        m_dis_idx = np.array(m_dis_idx)
        return m_ref_list, m_dis_list, m_dis_idx

    def build_live_files(self):
        files = []
        for dis_type in self.config.live_dis_type[1:]:
            dis_path = self.config.live_path + dis_type + "/"
            img_files = load_files(dis_path)
            before_num = len(dis_path + 'img')
            img_files.sort(key=lambda x: int(x[before_num:-4]))
            files.extend(img_files)
        return files

    def get_tid2013_ref2dis(self, ref_lists):
        if self.dis_type == "tid2013":
            dis_path = self.config.tid2013_path + "disimg/"
        else:
            dis_path = self.config.tid2013_path + self.dis_type + "/"

        m_dis_list, m_ref_list = [], []
        lists = load_files(dis_path)
        before_num = len(dis_path + 'i')
        lists.sort(key=lambda x: int(x[before_num:-4]))

        idx_file = {}
        for i, file in enumerate(lists):
            idx_file[file] = i

        for ref_file in ref_lists:
            name = os.path.basename(ref_file).split('.')[0]
            name1 = 'i' + name[1:]
            name2 = "I" + name[1:]
            dis_file1 = [file_name for file_name in lists if name1 in file_name]
            dis_file2 = [file_name for file_name in lists if name2 in file_name]
            dis_file1.extend(dis_file2)
            ref_file = [ref_file for i in range(len(dis_file1))]
            m_dis_list.extend(dis_file1)
            m_ref_list.extend(ref_file)
        dis_idx = np.array([idx_file[dis_list] for dis_list in m_dis_list])
        return m_ref_list, m_dis_list, dis_idx

    def build_csiq_files(self):
        files = []
        for dis_type in self.config.csiq_dis_type[1:]:
            dis_path = self.config.csiq_path + dis_type + "/"
            img_files = load_files(dis_path)
            files.extend(img_files)
        return files

    def get_csiq_ref2dis(self, ref_lists):
        if self.dis_type == "csiq":
            lists = self.build_csiq_files()
        else:
            dis_path = self.config.csiq_path + self.dis_type + '/'
            lists = load_files(dis_path)

        m_dis_list, m_ref_list = [], []

        idx_file = {}
        for i, file in enumerate(lists):
            idx_file[file] = i

        for ref_file in ref_lists:
            name = os.path.basename(ref_file).split('.')[0]
            dis_file = [file_name for file_name in lists if name in file_name]
            ref_file = [ref_file for i in range(len(dis_file))]
            m_dis_list.extend(dis_file)
            m_ref_list.extend(ref_file)
        dis_idx = np.array([idx_file[dis_list] for dis_list in m_dis_list])
        return m_ref_list, m_dis_list, dis_idx


    def get_mos(self, idx):
        mos = None
        if self.dis_type in self.config.ivc_dis_type:
            mos = self.get_ivc_mos(idx)
        elif self.dis_type in self.config.peid_dis_type:
            mos = self.get_peid_mos(idx)
        elif self.dis_type in self.config.live_dis_type:
            mos = self.get_live_mos(idx)
        elif self.dis_type in self.config.tid2013_dis_type:
            mos = self.get_tid2013_mos(idx)
        elif self.dis_type in self.config.csiq_dis_type:
            mos = self.get_csiq_mos(idx)
        else:
            print("get mos dis type %s error" % self.dis_type)
            exit()
        return mos

    def get_ivc_mos(self, idx):
        mos_path = self.config.ivc_path + self.dis_type + "_mos.mat"
        load_mos = loadmat(mos_path)[self.config.dis_type + "_mos"]
        mos = load_mos[idx].reshape(-1)
        return mos

    def get_peid_mos(self, idx):
        mos_path = self.config.peid_path + self.dis_type + "_mos.mat"
        load_mos = loadmat(mos_path)[self.config.dis_type + "_mos"]
        print(idx)
        mos = load_mos[idx].reshape(-1)
        return mos

    def get_live_mos(self, idx):
        mos_path = self.config.live_path + "dmos.mat"
        load_mos = loadmat(mos_path)["dmos"][0]
        mos = load_mos[idx].reshape(-1)
        return mos

    def get_tid2013_mos(self, idx):
        mos_path = self.config.tid2013_path + self.dis_type + "_mos.mat"
        load_mos = loadmat(mos_path)[self.config.dis_type + "_mos"]
        mos = load_mos[idx].reshape(-1)
        return mos

    def get_csiq_mos(self, idx):
        mos_path = self.config.csiq_path + self.dis_type + "_dmos.mat"
        load_mos = loadmat(mos_path)[self.config.dis_type + "_dmos"]
        mos = load_mos[idx].reshape(-1)
        return mos

    def norm_mos(self, mos):
        if self.dis_type in self.config.ivc_dis_type:
            if self.config.quality_range is None:
                mos = (mos - 1) / (5 - 1)
            elif self.config.quality_range == "high":
                mos = (mos - 3.5) / (5 - 3.5)
            elif self.config.quality_range == "median":
                mos = (mos - 2.5) / (3.5 - 2.5)
            elif self.config.quality_range == "low":
                mos = (mos - 1) / (2.5 - 1)
        elif self.dis_type in self.config.peid_dis_type:
            if self.config.quality_range is None:
                mos = (mos - 0) / (6 - 0)
            elif self.config.quality_range == "high":
                mos = (mos - 4) / (6 - 4)
            elif self.config.quality_range == "median":
                mos = (mos - 2) / (4 - 2)
            elif self.config.quality_range == "low":
                mos = (mos - 0) / (2 - 0)
        elif self.dis_type in self.config.live_dis_type:
            if self.config.quality_range is None:
                mos = (mos - 0) / (85 - 0)
            elif self.config.quality_range == "high":
                mos = (mos - 0) / (40 - 0)
            elif self.config.quality_range == "median":
                mos = (mos - 40) / (60 - 40)
            elif self.config.quality_range == "low":
                mos = (mos - 60) / (85 - 60)
        elif self.dis_type in self.config.tid2013_dis_type:
            if self.config.quality_range is None:
                mos = (mos - 0) / (7.25 - 0)
            elif self.config.quality_range == "high":
                mos = (mos - 5.75) / (7.25 - 5.75)
            elif self.config.quality_range == "median":
                mos = (mos - 4.25) / (5.75 - 4.25)
            elif self.config.quality_range == "low":
                mos = (mos - 0) / (4.25 - 0)
        elif self.dis_type in self.config.csiq_dis_type:
            if self.config.quality_range is None:
                mos = (mos - 0) / (1 - 0)
            elif self.config.quality_range == "high":
                mos = (mos - 0) / (0.25 - 0)
            elif self.config.quality_range == "median":
                mos = (mos - 0.25) / (0.5 - 0.25)
            elif self.config.quality_range == "low":
                mos = (mos - 0.5) / (1 - 0.5)
            else:
                print("norm mos quality range %s error" % self.config.quality_type)
                exit()
        else:
            print("norm mos dis type %s error" % self.dis_type)
            exit()
        return mos

    def ivc_quality_range(self, ref_files, dis_files, moss):
        moss = np.array(moss)
        idxs = None
        if self.config.quality_range is None:
            return ref_files, dis_files, moss
        elif self.config.quality_range == "high":
            idxs = np.where(moss >= 3.5)[0]
        elif self.config.quality_range == "median":
            idxs = np.where((moss > 2.5) & (moss < 3.5))[0]
        elif self.config.quality_range == "low":
            idxs = np.where(moss <= 2.5)[0]
        else:
            print("ivc quality range %s error" % self.config.quality_range)
            exit()
        moss = moss[idxs]
        dis_files = np.array(dis_files)[idxs]
        ref_files = np.array(ref_files)[idxs]
        return ref_files, dis_files, moss

    def peid_quality_range(self, ref_files, dis_files, moss):
        moss = np.array(moss)
        idxs = None
        if self.config.quality_range is None:
            return ref_files, dis_files, moss
        elif self.config.quality_range == "high":
            idxs = np.where(moss >= 4)[0]
        elif self.config.quality_range == "median":
            idxs = np.where((moss > 2) & (moss < 4))[0]
        elif self.config.quality_range == "low":
            idxs = np.where(moss <= 2)[0]
        else:
            print("peid quality range %s error" % self.config.quality_range)
            exit()
        moss = moss[idxs]
        print(moss)
        dis_files = np.array(dis_files)[idxs]
        ref_files = np.array(ref_files)[idxs]
        return ref_files, dis_files, moss

    def live_quality_range(self, ref_files, dis_files, moss):
        moss = np.array(moss)
        idxs = None
        if self.config.quality_range is None:
            return ref_files, dis_files, moss
        elif self.config.quality_range == "low":
            idxs = np.where(moss >= 60)[0]
        elif self.config.quality_range == "median":
            idxs = np.where((moss > 40) & (moss < 60))[0]
        elif self.config.quality_range == "high":
            idxs = np.where(moss <= 40)[0]
        else:
            print("live quality range %s error" % self.config.quality_range)
            exit()
        moss = np.array(moss)[idxs]
        dis_files = np.array(dis_files)[idxs]
        ref_files = np.array(ref_files)[idxs]
        return ref_files, dis_files, moss

    def tid2013_quality_range(self, ref_files, dis_files, moss):
        moss = np.array(moss)
        idxs = None
        if self.config.quality_range is None:
            return ref_files, dis_files, moss
        elif self.config.quality_range == "high":
            idxs = np.where(moss >= 5.75)[0]
        elif self.config.quality_range == "median":
            idxs = np.where((moss > 4.25) & (moss < 5.75))[0]
        elif self.config.quality_range == "low":
            idxs = np.where(moss <= 4.25)[0]
        else:
            print("tid2013 quality range %s error" % self.config.quality_range)
            exit()
        moss = moss[idxs]
        dis_files = np.array(dis_files)[idxs]
        ref_files = np.array(ref_files)[idxs]
        return ref_files, dis_files, moss

    def csiq_quality_range(self, ref_files, dis_files, moss):
        moss = np.array(moss)
        idxs = None
        if self.config.quality_range is None:
            return ref_files, dis_files, moss
        elif self.config.quality_range == "low":
            idxs = np.where(moss >= 0.5)[0]
        elif self.config.quality_range == "median":
            idxs = np.where((moss > 0.25) & (moss < 0.5))[0]
        elif self.config.quality_range == "high":
            idxs = np.where(moss <= 0.25)[0]
        else:
            print("csiq quality range %s error" % self.config.quality_range)
            exit()
        moss = np.array(moss)[idxs]
        dis_files = np.array(dis_files)[idxs]
        ref_files = np.array(ref_files)[idxs]
        return ref_files, dis_files, moss

    def quality_range(self, ref_files, dis_files, moss):
        print(self.dis_type)
        if self.dis_type in self.config.ivc_dis_type:
            ref_files, dis_files, moss = self.ivc_quality_range(ref_files, dis_files, moss)
            return ref_files, dis_files, moss
        elif self.dis_type in self.config.peid_dis_type:
            ref_files, dis_files, moss = self.peid_quality_range(ref_files, dis_files, moss)
            return ref_files, dis_files, moss
        elif self.dis_type in self.config.live_dis_type:
            ref_files, dis_files, moss = self.live_quality_range(ref_files, dis_files, moss)
            return ref_files, dis_files, moss
        elif self.dis_type in self.config.tid2013_dis_type:
            ref_files, dis_files, moss = self.tid2013_quality_range(ref_files, dis_files, moss)
            return ref_files, dis_files, moss
        elif self.dis_type in self.config.csiq_dis_type:
            ref_files, dis_files, moss = self.csiq_quality_range(ref_files, dis_files, moss)
            return ref_files, dis_files, moss
        else:
            print("quality range dis type % error" % self.dis_type)
            exit()


