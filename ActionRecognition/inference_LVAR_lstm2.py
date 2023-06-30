import copy
import csv
import os
import time

import cv2
import numpy as np
import torch

from network import C3D_MoCap3_LVAR_lstm2 as C3D_MoCap

# from dataset import VideoDataset
torch.backends.cudnn.benchmark = True


def CenterCrop(frame, size):
    h, w = np.shape(frame)[0:2]
    th, tw = size
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))

    frame = frame[y1:y1 + th, x1:x1 + tw, :]
    return np.array(frame).astype(np.uint8)


def center_crop(frame):
    frame = frame[8:120, 30:142, :]  # 截取原始帧高128*宽171中大小为112*112（*3）的画面
    return np.array(frame).astype(np.uint8)


def read_csvData(csvpath):
    frame_csv_org = np.loadtxt(open(csvpath), dtype=str, delimiter=',')  # mocapdata_video:[(t,qw,qx,qy,qz,px,py,pz),vx,vy,vz,ax,ay,az,wx,wy,wz,alpha(x,y,z)]
    length = frame_csv_org.shape[0]
    buffer_csv = np.empty((length, 19), np.dtype('float32'))
    for i_csv in range(length):
        if frame_csv_org[i_csv, 1] != '':
            tmp_csv = copy.deepcopy(frame_csv_org[i_csv, 1:20])  # 深拷贝（不是引用）
        elif frame_csv_org[i_csv, 1] == '':  # 如果遇到丢点情况
            ii = copy.deepcopy(i_csv)
            while(frame_csv_org[ii, 1] == ''):
                ii = ii + 1
            tmp_csv = copy.deepcopy(frame_csv_org[ii, 1:20])
        # tmp_csv = VideoDataset.normalize_Mocapcsv(tmp_csv)
        tmp_csv[4:10] = tmp_csv[4:10].astype("float64") / 100  # P(除100后单位从mm变为dm),v(dm/s)
        tmp_csv[10:13] = tmp_csv[10:13].astype("float64") / 1000  # a(m/s2)
        tmp_csv[16:19] = tmp_csv[16:19].astype("float64") / 10  # 角加速度
        buffer_csv[i_csv] = tmp_csv
    return buffer_csv


def Rec_main(video, CSVPath, lablestxt=r'C:\YHY\Python_code\HY_ActionRec\MultiFus_RGB_Motion2\dataloaders\0504_VideoandMocapData_lables.txt',
             modeltarPath='C:\\YHY\\Python_code\\HY_ActionRec\\MultiFus_RGB_Motion2\\run\\run13_2\\models\\C3D_MoCap19f-0330_VideoandMocapData_epoch-149.pth.tar',
             x_pre0 = torch.zeros(1, 3, 2, 112, 112).clone().detach(),
             x_pre1 = torch.zeros(1, 64, 2, 56, 56).clone().detach(),
             x_pre2 = torch.zeros(1, 128, 2, 28, 28).clone().detach(),
             x_pre3 = torch.zeros(1, 256, 2, 14, 14).clone().detach(),
             x_pre4 = torch.zeros(1, 512, 2, 7, 7).clone().detach(),
             h = torch.zeros(2, 1, 30).clone().detach(), c = torch.zeros(2, 1, 30).clone().detach(),
             feature_out=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("MultiFus_RGB_Motion2.inference.Rec_main--Device being used:", device)

    with open(lablestxt, 'r') as f:
        class_names = f.readlines()
        f.close()
    num_classes = int(len(class_names))
    # init model
    model = C3D_MoCap.C3D_MoCap(num_classes)
    checkpoint = torch.load(modeltarPath, map_location = lambda storage, loc: storage)

    model.load_state_dict(checkpoint['state_dict'])  #模型参数
    # optimizer.load_state_dict(checkpoint['opt_dict'])  #优化参数
    
    model.to(device)
    model.eval()

    # READ DATA and Video
    CSVData = read_csvData(CSVPath)
    cap = cv2.VideoCapture(video)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    retaining = True
    i_frame = 0
    clip = []
    clip_csv = []
    prob_all = []
    lables_all = []
    csvused_all = []  # 用到的
    feature = []
    print("video_address:" + video)
    print("MoCap_address:" + CSVPath)
    while retaining:
        retaining, frame = cap.read()
        if not retaining and frame is None:
            continue
        tmp_ = center_crop(cv2.resize(frame, (171, 128))) #cv2.resize原始图片大小到宽171，高128，再通过center_crop截取该帧中的一部分画面
        tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
        clip.append(tmp)

        # TODO: 改变动捕csv数据的选取策略
        # 目前的选取的是直接按照（120hzMocap与60Hz视频这样2：1直接对应，，后期需要修改）选取对应的csv数据
        tmp_csv = CSVData[int(i_frame * CSVData.shape[0] / frame_count)]
        csvused_all.append(int(i_frame * CSVData.shape[0] / frame_count))
        clip_csv.append(tmp_csv)

        if len(clip) == 16 and len(clip_csv) == 16:
            inputs = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            inputs = torch.from_numpy(inputs)
            inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)  # Size([1, 3, 16, 112, 112])

            inputs_csv_i = np.array(clip_csv).astype(np.float32)
            inputs_csv_i = np.expand_dims(inputs_csv_i, axis=0)
            inputs_csv_i = torch.from_numpy(inputs_csv_i)
            inputs_csv_i = torch.autograd.Variable(inputs_csv_i, requires_grad=False).to(device)  # Size([1, 16, 19])
            # inputs_csv_i = torch.from_numpy(CSVData[i_frame].astype('float32')).to(device)
            # inputs_csv_i = inputs_csv_i.view(-1, 16*19)  #在model.forrward里面转了

            x_pre0 = torch.autograd.Variable(x_pre0).to(device)
            x_pre1 = torch.autograd.Variable(x_pre1).to(device)
            x_pre2 = torch.autograd.Variable(x_pre2).to(device)
            x_pre3 = torch.autograd.Variable(x_pre3).to(device)
            x_pre4 = torch.autograd.Variable(x_pre4).to(device)
            h = torch.autograd.Variable(h).to(device)
            c = torch.autograd.Variable(c).to(device)
            with torch.no_grad():
                outputs, x_pre0,x_pre1,x_pre2,x_pre3,x_pre4 ,h,c, featurei = model.forward(inputs, inputs_csv_i, x_pre0,x_pre1,x_pre2,x_pre3,x_pre4,h,c, feature_out=feature_out)
            if feature_out:
                feature.append(featurei)
            # # TODO：改变Softmax(会使得置信度爆表--本来高的高的太多)，只求幂不归一化+开根号，还是relu，sigmod
            probs = torch.nn.Softmax(dim=1)(outputs)
            # # 只求幂不归一化 +开根号
            # tmp_probs = [math.exp(i) for i in outputs[0]]
            # for i in range(len(tmp_probs)):
            #     probs[0][i] = math.sqrt(tmp_probs[i])

            # # 用 relu：
            # probs = torch.nn.ReLU(inplace=True)(outputs)

            # # 用Sigmoid：
            # probs = torch.sigmoid(outputs)
            # # for i in range(len(outputs[0])):
            # #     probs[0][i] = 1./(1+math.exp(-outputs[0][i]))

            label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]
            # topk取tensor中前k个大小的值及标签probs.topk(2,dim=1,largest=True)，kthvalue只取第k给大小的值与标签
            label_2 = probs.kthvalue(2)[1].detach().cpu().numpy()[0]
            prob_all.append(probs.detach().cpu().numpy()[0])
            lables_all.append(label)
            cv2.putText(frame, (class_names[label].split(' ')[-1].strip()+" : %.4f" % probs[0][label]), (365, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)
            cv2.putText(frame, (class_names[label_2].split(' ')[-1].strip()+" : %.4f" % probs[0][label_2]), (365, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 0, 255), 1)

            # # # 绘制流程图
            # if frame_count > frame.shape[1]:
            #     # if (i_frame-15) < (frame.shape[1] / 2):
            #     #     pi_x = (i_frame-15+1)*2
            #     #     pi_y = frame.shape[0] - 50
            #     # else:
            #     pi_x = int(((i_frame-15) % (frame.shape[1] / 2)) * 2)
            #     pi_y = frame.shape[0] - 50 + int(((i_frame-15) / (frame.shape[1] / 2)))*10
            #     cv2.rectangle(frame, (pi_x, pi_y), (pi_x + 2, pi_y + 10), (255, 255, 255), thickness=-1)
            #
            # else:
            #     unit_length = int(frame.shape[1]/frame_count)
            #     cv2.rectangle(frame, ((i_frame-15)*unit_length, frame.shape[0]-30), ((i_frame-15+1)*unit_length, frame.shape[0]-10), (255, 255, 255), thickness=-1)

            clip.pop(0)
            clip_csv.pop(0)
        i_frame = i_frame + 1
        cv2.imshow('result', frame)
        cv2.waitKey(1)  # 单位为ms

    cap.release()
    cv2.destroyAllWindows()
    return prob_all, lables_all, csvused_all, feature


def writeans_csv(outpath, CSVPath, lables_all, prob_all, csvused_all, lablestxt='./dataloaders/0504_VideoandMocapData_lables.txt', oriclass=False):
    # 输出为用到的帧数,概率,种类,[(t,qw,qx,qy,qz,px,py,pz),vx,vy,vz,ax,ay,az,wx,wy,wz,alpha(x,y,z)]
    # 注意输出的标签为lables_all[i]+1，即从1开始，而不是0
    # prob_all = np.asarray(prob_all, dtype=np.float32)
    print("Res_addresss: " + outpath)
    prob_all = np.asarray(prob_all, dtype=np.float32)
    lables_all = np.asarray(lables_all, dtype=np.float32)
    CSVData = np.loadtxt(open(CSVPath), dtype=np.float32, delimiter=',', usecols=np.arange(0,20))  # mocapdata_video:[(t,qw,qx,qy,qz,px,py,pz),vx,vy,vz,ax,ay,az,wx,wy,wz,alpha(x,y,z)]20列

    if oriclass:
        # 读取label构建字典class_names
        with open(lablestxt, 'r') as f:
            class_names_tmp = f.readlines()
            f.close()
        class_names = {}
        for i in range(len(class_names_tmp)):
            class_names[class_names_tmp[i].split(' ')[-1].strip()] = i

        ori_ActClass = np.loadtxt(open(CSVPath), dtype=str, delimiter=',', usecols=20)
        for i in range(len(ori_ActClass)):
            ori_ActClass[i] = class_names[ori_ActClass[i]] + 1

    i = 0
    f = open(outpath, 'w', newline='')
    writer = csv.writer(f)
    for csv_ui in csvused_all:
        if i < len(lables_all):
            if oriclass == False:
                writer.writerow(np.hstack((csv_ui,  prob_all[i], lables_all[i]+1, CSVData[int(csv_ui), :])))  # 输出为用到的帧数,概率,种类,[(t,qw,qx,qy,qz,px,py,pz),vx,vy,vz,ax,ay,az,wx,wy,wz,alpha(x,y,z)]
            else:
                writer.writerow(np.hstack((csv_ui,  prob_all[i], lables_all[i]+1, CSVData[int(csv_ui), :], ori_ActClass[csv_ui]))) #输出为用到的帧数,概率,种类,[(t,qw,qx,qy,qz,px,py,pz),vx,vy,vz,ax,ay,az,wx,wy,wz,alpha(x,y,z)],人工标记类别
        i = i+1
    return 0


if __name__ == '__main__':
    filename = 'ct10_0'

    videoPath = 'C://Users//G314-Optitrack//Desktop//Demonstration_data//testdata_2//' + filename + '.avi'
    CSVPath = 'C://Users//G314-Optitrack//Desktop//Demonstration_data//testdata_2//' + filename + '.csv'
    needconcat_ornot = True  # 为True表示需要在最后一列插入标签, 这样在最后writeans_csv时会转换approaching等str标签到int序号

    lablestxt = './dataloaders/0504_VideoandMocapData_lables.txt'
    modeltarPath = r'C:\YHY\Python_code\HY_ActionRec\MultiFus_RGB_Motion2\run\run14_4\models\C3D_MoCapLvarLstm2-0504_VideoandMocapData_epoch-95.pth.tar'
    prob_all, lables_all, csvused_all, _ = Rec_main(videoPath, CSVPath, lablestxt, modeltarPath)

    t = int(time.time())
    dt = time.strftime("%m%d", time.localtime(t))
    output_dir = './res/' + str(dt)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    res_path = output_dir + '/' + filename+'_RecAns_'+'R14_3.csv'
    writeans_csv(res_path, CSVPath, lables_all, prob_all, csvused_all, lablestxt=lablestxt, oriclass=needconcat_ornot)











