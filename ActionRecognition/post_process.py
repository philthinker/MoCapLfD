"""
@author: HY
@file:DataConcat.py
@time:2022-04-17
@function: 进行后处理，得到混淆矩阵图，以及带进度条的视频，并保存进度条。
"""

import math

import cv2 as cv
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

# from PIL import Image,ImageDraw,ImageFont


def de_sensitivity(rec_res, rec_probs, cliplen=16):
    """
    To reduce the sensitivity of the recognition results. (每次识别输入的都是16帧， 将得到的结果加值到这16帧上， 最后再归一化--->降低识别结果的敏感性)
    ref: Hwang.《Development of a Mimic Robot—Learning From Demonstration Incorporating Object Detection and Multiaction Recognition》. IEEE Consumer Electronics Magazine 9.
    time: 2022-04-27
    :param rec_res: recognition results（每一帧对应的动作class）。shape:n*1, dtype:int32, 值大小为0 ~ classnums-1
    :param rec_probs: shape: n*classnums,  dtype:float32
    :param cliplen: 窗口范围
    :return: new_rec_res
    :return: new_rec_res
    """

    new_rec_probs = np.zeros([len(rec_probs), len(rec_probs[0])])
    # new_rec_probs = rec_probs
    for i in range(len(rec_probs)):
        for j in range(rec_probs.shape[1]):
            for clipi in range(cliplen):
                if i + clipi < len(rec_probs):  # 避免越界（其实是因为舍弃了视频的最后15帧--往往为静止动作，无影响）

                    if clipi == 0:
                        continue
                    new_rec_probs[i + clipi][j] += rec_probs[i][j]

                    # if clipi == 0:
                    #     continue
                    # new_rec_probs[i + clipi][j] += rec_probs[i][j]/(clipi+1)

                    # new_rec_probs[i + clipi][j] += rec_probs[i][j] / (1+abs((clipi - 7)))

                    # if clipi == 0:
                    #     new_rec_probs[i + clipi][j] = new_rec_probs[i + clipi][j] * math.exp(clipi)
                    # else:
                    #     new_rec_probs[i + clipi][j] += rec_probs[i][j] * math.exp(clipi)/ 5171508.755625341


    new_rec_res = rec_res
    for i in range(len(new_rec_probs)):
        for j in range(len(new_rec_probs[i])):
            new_rec_probs[i][j] = math.exp(new_rec_probs[i][j])  # 其实就是个softmax
        new_rec_probs[i] /= sum(new_rec_probs[i]) # 归一化概率值
        new_rec_res[i] = np.argmax(new_rec_probs[i])  # 提取概率值最大的标签

    return new_rec_res, new_rec_probs


if __name__ == "__main__":
    CM_ornot = True  # False表示不绘制混淆矩阵(当没有手工标记数据时，，)
    display = True  # 为True表示需要同时显示视频和轨迹----目前没有写false的后面的代码，不要修改此值
    deSensitivity_ornot = True  # False表示对轨迹 不进行deSensitivty处理。True进行

    # 输入(修改videoPath和fileName,savepackage-->csv文件)
    filename = 'ct10_0'
    # recFileName = '0525' + '//' + filename + '_RecAns_' + 'R14_2'
    recFileName = '0601' + '//' + 'R14_2//' + filename + '_RecAns'
    videoPath = 'C://Users//G314-Optitrack//Desktop//Demonstration_data//testdata_2//' + filename + '.avi'
    savepackage = r'C:\YHY\Python_code\HY_ActionRec\MultiFus_RGB_Motion2\res'
    rec_csvpath_ori = savepackage + '//' + recFileName + '.csv'  # 动作识别结果所在的文件位置
    labelpath = './dataloaders/0504_VideoandMocapData_lables.txt'  # 动作标签文件
    with open(labelpath, 'r') as f:
        labels_all = f.readlines()  # 所有动作标签（带1，2，3序号，要class_names[label].split(' ')[-1].strip()提取标签名）
        f.close()

    rec_model_res = np.loadtxt(rec_csvpath_ori, dtype=int, delimiter=',', usecols=len(labels_all)+1) - 1  # 模型识别结果  # 注意这里的dtype为int
    rec_model_probs = np.zeros([len(rec_model_res), len(labels_all)])
    for i in range(len(labels_all)):
        rec_model_probs[:,i] = np.loadtxt(rec_csvpath_ori, dtype=np.float32, delimiter=',', usecols=i+1)  # 概率值

    if deSensitivity_ornot:
        rec_model_res, rec_model_probs = de_sensitivity(rec_model_res, rec_model_probs, cliplen=16)
        recFileName += '_ds'

    # qw, qx, qy, qz,  px, py, pz
    Traj_qp = np.loadtxt(rec_csvpath_ori, dtype=np.float32, delimiter=',',
                         usecols=(len(labels_all)+3, len(labels_all)+4, len(labels_all)+5, len(labels_all)+6, len(labels_all)+7, len(labels_all)+8, len(labels_all)+9))


    # # 视频加bar一起显示
    # h_choose = [50, 100, 150, 200, 250]
    h_choose = [88, 88, 88, 88, 88, 88, 88, 88, 88]
    # w_unit, h_unit, pi_x, pi_y = 1, 88, 0, 0
    w_unit, pi_x, pi_y = 1, 0, h_choose[np.argmax(h_choose)]
    # color顺序为(b,g,r)----cv
    color = [(199, 207, 28), (141, 199, 251), (118, 214, 167), (152, 161, 184), (224, 222, 168), (57, 104, 205)]
    barImg = np.zeros((h_choose[np.argmax(h_choose)], len(rec_model_res) * w_unit, 3), np.uint8)
    if display:
        # 三维MoCapTraj图设置
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        plt.rcParams['axes.unicode_minus'] = False
        ax1 = plt.axes(projection='3d')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')
        ax1.set_zscale('linear')
        # ax1.xaxis.set_major_locator(plt.MultipleLocator(10))
        # ax1.yaxis.set_major_locator(plt.MultipleLocator(10))
        # ax1.zaxis.set_major_locator(plt.MultipleLocator(10))
        ax1.set_title('MoCap轨迹图')
        # color_traj顺序为(r,g,b)，大小0-1，也可 color_traj = ['b','g','r','c','m','y','k']
        color_traj = [[28 / 255, 207 / 255, 199 / 255], [251 / 255, 199 / 255, 141 / 255],
                      [167 / 255, 214 / 255, 118 / 255], [184 / 255, 161 / 255, 152 / 255],
                      [168 / 255, 222 / 255, 224 / 255], [205 / 255, 104 / 255, 57 / 255]]

        patches = [mpatches.Patch(color=color_traj[i], label="{:s}".format(labels_all[i].split(' ')[-1].strip())) for i
                   in range(len(labels_all))]
        ax1.legend(handles=patches, bbox_to_anchor=(1.2, 1.12), ncol=1)  # 显示图例 loc='best''upper right''lower left'
        plt.grid(True)
        plt.ion()  # Turn the interactive mode on. 有了他就不需要plt.show()了

        cap1 = cv.VideoCapture(videoPath)
        for i in range(len(rec_model_res)):
            probs_i = torch.from_numpy(rec_model_probs[i, :])
            # label = np.argmax(probs_i)
            label_1 = torch.max(probs_i, 0)[1].detach().cpu().numpy()
            label_2 = probs_i.kthvalue(2)[1].detach().cpu().numpy()
            retaining, frame = cap1.read()
            if not retaining and frame is None:
                continue
            cv.putText(frame, (labels_all[label_1].split(' ')[-1].strip() + " : %.4f" % probs_i[label_1]), (365, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 0, 255), 2)
            cv.putText(frame, (labels_all[label_2].split(' ')[-1].strip() + " : %.4f" % probs_i[label_2]), (365, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 0.65,
                       (0, 0, 255), 1)
            # 进度条：
            cv.rectangle(barImg, (pi_x, pi_y), (pi_x + w_unit, pi_y - h_choose[rec_model_res[i]]),
                         color[rec_model_res[i]], thickness=-1)
            pi_x = pi_x + w_unit

            cv.namedWindow('result', cv.WINDOW_NORMAL)
            # 改变窗口位置：left top
            cv.moveWindow("result", -1000, 0)

            # 拼接进度条和视频:
            cv.imshow('result', np.vstack(
                (frame, cv.resize(barImg, (frame.shape[1], 10)))))  # 将源视频与进度条合并显示，进度条大小resize到宽frame.shape[1],高10
            cv.waitKey(1)  # 单位为ms

            # # 同时刷新图
            # plt.cla()  # 表示清除当前轴axis， 即当前图中的当前活动轴。 它使其他轴保持不变。
            # # plt.clf()  # 表示Clear the current figure. 使用其所有轴清除整个当前图形 ，但使窗口保持打开状态，以便可以将其重新用于其他绘图。
            # # plt.close()  # 关闭一个window，如果没有另外指定，它将是当前窗口。
            # plt.imshow(barImg)
            # plt.pause(0.0001)  # 单位为s

            # 同时刷新轨迹三维图
            if i > 1:
                # 绘制三维MoCapTraj图
                ax1.plot3D(Traj_qp[i - 1:i + 1, 4], Traj_qp[i - 1:i + 1, 5], Traj_qp[i - 1:i + 1, 6],
                           c=color_traj[rec_model_res[i]])  # 连续线c='blue'
                # 散点图 c=rec_model_res[i]
                # ax1.scatter3D(Traj_qp[i, 4], Traj_qp[i, 5], Traj_qp[i, 6], s=4, c=color_traj[rec_model_res[i]])
                plt.pause(0.00001)

            # traj_fig = plt.figure(figsize=(9, 6))
            # ax1 = traj_fig.add_subplot(111, projection='3d')
            # ax1.plot(Traj_qp[0:i, 4], Traj_qp[0:i, 5], Traj_qp[0:i, 6], label=u'路径', c='r')
            # ax1.legend()
    # 保存barImg和MoCapTraj图
    bar_filepath = savepackage + '//' + recFileName + '_bar'
    traj_filepath = savepackage + '//' + recFileName + '_MoCapTraj'
    if deSensitivity_ornot:
        bar_filepath += '_ds'
        traj_filepath += '_ds'
    cv.imwrite(bar_filepath + '.png', barImg, [cv.IMWRITE_PNG_COMPRESSION, 0])
    plt.savefig((traj_filepath + '.png').format(ax1))
    print('...SAVED')
    cap1.release()
    cv.destroyAllWindows()
    plt.ioff()  # 关掉交互模式，进入阻塞模式（默认的模式）
    plt.show()  # 显示图，不至于让MoCapTraj图直接关掉


    # # # 绘制识别结果bar  --->不需要了
    # w_unit, h_unit, pi_x, pi_y = 1, 88, 0, 0
    # color = [(224, 222, 168), (199, 207, 28), (141, 199, 251), (118, 214, 167), (152,161,184)]
    # barImg = np.zeros((h_unit, len(rec_model_res), 3), np.uint8)
    # for i in range(len(rec_model_res)):
    #     # # 红色(255, 0, 0)   # # 猩红(255, 255, 255)  # #紫罗兰红(199, 21, 133)    # #蓝(0, 0, 255)   # #板颜蓝(106, 90, 205)
    #     # #黄(255, 255, 0)     # # 镉黄(255, 153, 18)      # # 橙色(255, 97, 0)        # # 绿(0, 255, 0)
    #     cv.rectangle(barImg, (pi_x, pi_y), (pi_x + w_unit, pi_y + h_unit), color[rec_model_res[i]], thickness=-1)
    #     pi_x = pi_x + w_unit
    #     cv.imshow('barImg', barImg)
    #     cv.waitKey(1)
    # cv.imwrite(r'C:\YHY\Python_code\HY_ActionRec\MultiFus_RGB_Motion2\res' + '//' + recFileName+'_bar.png', barImg, [cv.IMWRITE_PNG_COMPRESSION, 0])