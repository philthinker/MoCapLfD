import os

import cv2
import numpy as np
import pandas
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from mypath import Path


class VideoDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            dataset (str): Name of dataset. Defaults to 'ucf101'（默认为ucf101）.     'data0927' added by HY.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            clip_len (int): Determines how many frames are there in each clip(在每个剪辑中有多少帧). Defaults to 16.
            preprocess (bool): Determines whether to preprocess dataset. Default is False.
    clip_len是处理一个视频所抽出的帧数
    """
    # 注意第一次要预处理数据的,第一次执行 preprocess=True，后续改为false
    def __init__(self, dataset='ContinuousAct', split='train', clip_len=16, batch_clip=88, preprocess=False, lablestxt='./dataloaders/0504_VideoandMocapData_lables.txt'):
        self.root_dir, self.output_dir = Path.db_dir(dataset)
        folder = os.path.join(self.output_dir, split)  #路径拼接,结果为"output_dir\\split"
        self.clip_len = clip_len
        self.split = split
        self.batch_clip = batch_clip  # 每个视频抽出clip_len+batch_clip帧，一次网络吞进去clip_len，一个视频进行了batch_clip次训练
        # The following three parameters are chosen as described in the paper section 4.1
        # 图片为171*128像素
        self.resize_width = 171
        self.resize_height = 128
        self.crop_size = 112

        if not self.check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You need to download it from official website.')

        if preprocess or (not self.check_preprocess()) :
            print('Preprocessing of {} dataset, this will take long, but it will be done only once.'.format(dataset))
            self.preprocess()

        # Obtain all the filenames of files inside all the class folders
        # Going through each class folder one at a time
        self.fnames, self.labels = [], []
        with open(lablestxt, 'r') as f:
            self.labels = f.readlines()
            f.close()
        for fname in sorted(os.listdir(folder)):
            self.fnames.append(os.path.join(folder, fname))
            # labels.append(label)

        # assert len(labels) == len(self.fnames)
        print('Number of {} videos: {:d}'.format(split, len(self.fnames)))

        if dataset == '0330_VideoandMocapData':
            if not os.path.exists('dataloaders/0330_VideoandMocapData_lables.txt'):
                with open('C:\\YHY\\Python_code\\HY_ActionRec\\MultiFus_RGB_Motion2\\dataloaders\\0330_VideoandMocapData_lables.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id+1) + ' ' + label + '\n')
        elif dataset == '0504_VideoandMocapData':
            if not os.path.exists('dataloaders/0504_VideoandMocapData_lables.txt'):
                with open('C:\\YHY\\Python_code\\HY_ActionRec\\MultiFus_RGB_Motion2\\dataloaders\\0504_VideoandMocapData_lables.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id+1) + ' ' + label + '\n')

    def __len__(self):
        return len(self.fnames)

    #todo 需要重写__getitem__方法
    def __getitem__(self, index):
        # Loading and preprocessing.

        # labels = np.array(self.label_array[index])
        buffer, buffer_csv, buffer_labels_hand = self.load_frames(self.fnames[index])  # ucf101一共有8460个文件夹
        if buffer.shape[0] < self.clip_len + self.batch_clip:
            index = 1 # 我自己随便选的，用来替换掉帧数少的数据
            # labels = np.array(self.label_array[index])
            buffer, buffer_csv, buffer_labels_hand = self.load_frames(self.fnames[index])
        buffer, crop_time_index = self.crop(buffer, self.clip_len, self.crop_size)
        buffer_csv = self.crop_csv(buffer_csv, crop_time_index)
        buffer_labels_hand = self.crop_csv(buffer_labels_hand, crop_time_index)

        if self.split == 'test':
            # Perform data augmentation
            buffer = self.randomflip(buffer)
        buffer = self.normalize(buffer)
        buffer_csv = self.normalize_Mocapcsv(buffer_csv)
        buffer = self.to_tensor(buffer)
        buffer_labels_hand.resize(len(buffer_labels_hand))
        return torch.from_numpy(buffer), torch.from_numpy(buffer_csv), torch.from_numpy(buffer_labels_hand)


    def check_integrity(self):
        if not os.path.exists(self.root_dir):
            return False
        else:
            return True

    def check_preprocess(self):
        # TODO: Check image size in output_dir
        if os.path.exists(self.output_dir) and os.path.exists(os.path.join(self.output_dir, 'train')):
            return True
        # elif not os.path.exists(os.path.join(self.output_dir, 'train')):
        #     return False

        for ii, video_class in enumerate(os.listdir(os.path.join(self.output_dir, 'train'))):
            for video in os.listdir(os.path.join(self.output_dir, 'train', video_class)):
                video_name = os.path.join(os.path.join(self.output_dir, 'train', video_class, video),
                                    sorted(os.listdir(os.path.join(self.output_dir, 'train', video_class, video)))[0])
                image = cv2.imread(video_name)
                if np.shape(image)[0] != 128 or np.shape(image)[1] != 171:
                    return False
                else:
                    break

            if ii == 10:
                break

        return True

    def preprocess(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            os.mkdir(os.path.join(self.output_dir, 'train'))
            os.mkdir(os.path.join(self.output_dir, 'val'))
            os.mkdir(os.path.join(self.output_dir, 'test'))

        # Split train/val/test sets遍历
        video_files = []
        for file in os.listdir(self.root_dir):
            # file_path = os.path.join(self.root_dir, file)
            if file.split('.')[-1] == 'avi':
                # fileName = file.split('.')[0]
                video_files.append(file)


        # 用train——test——split将目标文件夹video_files分为train_and_valid和test
        train_and_valid, test = train_test_split(video_files, test_size=0.10, random_state=42)
        # 用train——test——split将目标文件夹train_and_valid分为train和val
        train, val = train_test_split(train_and_valid, test_size=0.20, random_state=42)

        train_dir = os.path.join(self.output_dir, 'train')
        val_dir = os.path.join(self.output_dir, 'val')
        test_dir = os.path.join(self.output_dir, 'test')

        if not os.path.exists(train_dir):
            os.mkdir(train_dir)
        if not os.path.exists(val_dir):
            os.mkdir(val_dir)
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)
        for video in train:
            self.process_video_csv(video, train_dir, Video_Mocap_tag=1)
        for video in val:
            self.process_video_csv(video, val_dir, Video_Mocap_tag=1)
        for video in test:
            self.process_video_csv(video, test_dir, Video_Mocap_tag=1)

        print('Preprocessing finished.')

    def process_video_csv(self, video, save_dir, Video_Mocap_tag=1):
        # 2022-03-03: Video_Mocap_tag = 1 表示数据输出为视频帧加和动捕数据

        # Initialize a VideoCapture object to read video data into a numpy array
        video_filename = video.split('.')[0]
        if not os.path.exists(os.path.join(save_dir, video_filename)):
            os.mkdir(os.path.join(save_dir, video_filename))
        csvfilename = os.path.join(self.root_dir, video_filename) + '.csv'
        mocapdata_video = np.loadtxt(open(csvfilename), dtype=str, delimiter=',')
        # mocapdata_video: [t, qw, qx, qy, qz, px, py, pz, vx, vy, vz, ax, ay, az, wx, wy, wz, alphax, alphay, alphaz]
        #读取视频
        capture = cv2.VideoCapture(os.path.join(self.root_dir, video))

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Make sure splited video has at least 16 frames
        # EXTRACT_FREQUENCY=4指每4帧取一次数据
        #TODO: 考虑将EXTRACT_FREQUENCY的选取问题
        EXTRACT_FREQUENCY = 1
        Picturenum_min = 18
        # 在整数除法中，除法/总是返回一个浮点数。如果只想得到整数的结果，丢弃可能的小数部分，可以使用运算符//(当然，//得到的并不一定是整数类型的数，它与分母分子的数据类型有关系)。
        if frame_count // EXTRACT_FREQUENCY <= Picturenum_min:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= Picturenum_min:
                EXTRACT_FREQUENCY -= 1
                if frame_count // EXTRACT_FREQUENCY <= Picturenum_min:
                    EXTRACT_FREQUENCY -= 1

        count = 0
        i = 0
        retaining = True

        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            if frame is None:
                continue
            if count % EXTRACT_FREQUENCY == 0:
                if Video_Mocap_tag==0:  # 保存图片
                    if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                        frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                    cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=frame)
                if Video_Mocap_tag==1:  # 保存图片+对应每一帧的mocap数据
                    if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                        frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                    cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=frame)

                    # TODO: 改变动捕csv数据的选取策略
                    # 目前建数据集时的选取的是直接按照（120hzMocap与60Hz视频这样2：1直接对应来选的，，后期需要修改）
                    mocap_video_framei = mocapdata_video[int(i * mocapdata_video.shape[0]/frame_count)]
                    # fd_video_framei = self.getFourierDescriptor(frame)  # 32*1
                    # fm_video_framei = np.append(fd_video_framei,mocap_video_framei[1:8])  # fm_video_framei融合手部轮廓傅里叶算子+mocap轨迹
                    tiemdatacsv = pandas.DataFrame(mocap_video_framei)
                    output_filename = os.path.join(save_dir, video_filename, '0000{}.csv'.format(str(i)))
                    tiemdatacsv.to_csv(output_filename, header=None, index=None)
                i += 1
            count += 1

        # Release the VideoCapture once it is no longer needed
        capture.release()

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer

    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame
        return buffer

    def normalize_Mocapcsv(self,buffer_csv):
        # todo 归一化？？？
        tmp_buffer_csv = buffer_csv
        tmp_buffer_csv[:, 4:10] = buffer_csv[:, 4:10]/100  # P(除100后单位从mm变为dm),v(dm/s)
        tmp_buffer_csv[:, 10:13] = buffer_csv[:, 10:13] / 1000  # a(m/s2)
        tmp_buffer_csv[:, 16:19] = buffer_csv[:, 16:19] / 10  # 角加速度
        buffer_csv = tmp_buffer_csv
        return buffer_csv

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir):
        # os.listdir(file_dir)[2].split('.')[1]
        jpg_csv = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frames = []
        frames_csv = []
        for tempt in jpg_csv:
            if tempt.split('.')[1] == 'jpg':
                frames.append(tempt)
            if tempt.split('.')[1] == 'csv':
                frames_csv.append(tempt)
        frame_count = len(frames)
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        # mocapdata_video:[(t,qw,qx,qy,qz,px,py,pz),vx,vy,vz,ax,ay,az,wx,wy,wz,alpha(x,y,z)], 舍去t
        buffer_csv = np.empty((frame_count, 19), np.dtype('float32'))
        # todo 人工标签（读取str转为int存入buffer_labels_hand中返回）
        class_names = {}
        for i in range(len(self.labels)):
            class_names[self.labels[i].split(' ')[-1].strip()] = i
        buffer_labels_hand = np.empty((frame_count, 1), np.dtype('int8'))

        frames.sort(key=lambda x: int((x.split('.')[0])[-4:-1]))
        # for i, frame_name in enumerate(frames):
        for i in range(len(frames)):
            frame_name = frames[i]
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame
        frames_csv.sort(key=lambda x: int((x.split('.')[0])[-4:-1]))
        # for i_csv, frame_csv_name in enumerate(frames_csv):
        for i_csv in range(len(frames_csv)):
            frame_csv_name = frames_csv[i_csv]
            csv_org = np.loadtxt(open(frame_csv_name), dtype=str, delimiter=',')
            labels_csv_org = csv_org[-1]  # 手工标记
            #todo 这里的label的值应该在0-5之间
            buffer_labels_hand[i_csv] = class_names[labels_csv_org]
            frame_csv = csv_org[1:20] # 第0列为t--不要，要第1列到19列。
            if frame_csv[0] != '""':
                buffer_csv[i_csv] = frame_csv
            elif frame_csv[0] == '""':  # 如果遇到丢点情况
                if i_csv != 0:
                    buffer_csv[i_csv] = buffer_csv[i_csv-1]
                else:
                    tmptname = frame_csv_name.split('.')[0]
                    tmptname_address = tmptname.split('0000')[0]
                    # tmptname_name = tmptname.split('0000')[1]
                    newname = os.path.join(tmptname_address, '00001.csv')
                    frame_csv_org = np.loadtxt(open(newname), dtype=str, delimiter=',')
                    frame_csv = frame_csv_org[1:20]
                    buffer_csv[i_csv] = frame_csv
        return buffer, buffer_csv, buffer_labels_hand

    def crop(self, buffer, clip_len, crop_size, data_aug_jittering=True):

        # randomly select time index for temporal jittering

        crop_size_h = crop_size
        crop_size_w = crop_size
        if data_aug_jittering:
            crop_size_basic = [128, 112, 96, 84]
            crop_size_h = crop_size_basic[np.random.randint(3)]
            crop_size_w = crop_size_basic[np.random.randint(3)]
        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size_h + 1)
        width_index = np.random.randint(buffer.shape[2] - crop_size_w + 1)

        # todo 选定一个视频抽几组--目前定的是8组batch_clip=8
        if clip_len + self.batch_clip > buffer.shape[0]:
            buffer = buffer[:,
                     height_index:height_index + crop_size_h,
                     width_index:width_index + crop_size_w, :]  # 最终大小为(clip_len, crop_size_h, crop_size_w, 3)

            return buffer, 0
        else:
            time_index = np.random.randint(buffer.shape[0] - clip_len - self.batch_clip)
        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index: time_index + clip_len+self.batch_clip,
                 height_index:height_index + crop_size_h,
                 width_index:width_index + crop_size_w, :]  # 最终大小为(clip_len, crop_size_h, crop_size_w, 3)
        if data_aug_jittering:
            tmp = np.zeros((buffer.shape[0], crop_size, crop_size, 3),dtype='float32')
            for i in range(buffer.shape[0]):
                tmp[i] = cv2.resize(buffer[i], (crop_size, crop_size))
            return tmp, time_index
        else:
            return buffer, time_index

    def crop_csv(self, buffer_csv, time_index):
        if self.clip_len + self.batch_clip > buffer_csv.shape[0]:
            return buffer_csv
        else:
            buffer_csv = buffer_csv[time_index: time_index + self.clip_len + self.batch_clip, :]
        return buffer_csv


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    # 注意第一次要预处理数据的,第一次执行 preprocess=True，后续改为False
    # train_data = VideoDataset(dataset='all_withgloves_VideoandMocap', split='test', clip_len=16, preprocess=False)
    train_data = VideoDataset(dataset='ContiACT', split='test', clip_len=16, preprocess=True)
    train_loader = DataLoader(train_data, batch_size=10, shuffle=True, num_workers=4)

    for i, sample in enumerate(train_loader):
        inputs1 = sample[0]
        inputs2 = sample[1]
        labels = sample[2]
        print(inputs1.size())
        print(inputs2.size())
        print(labels.size())
        # print(labels)

        if i == 1:
            break
