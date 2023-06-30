import os
import time
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

import inference_LVAR_lstm2
import post_process


def rec_save(fileName, modeltarPath, modelRunName, oriDataPackage, model='MultiFus_RGB_Motion2',
             outpackage_ori='./res/',
             lablestxt=r'C:\YHY\Python_code\HY_ActionRec\MultiFus_RGB_Motion2\dataloaders\0504_VideoandMocapData_lables.txt',
             needconcat=True, gt_out=False, writeft_out=True):
    """
    :param needconcat: 为True表示需要在最后一列插入标签, 这样在最后writeans_csv时会转换原始文件中最后一列Approaching等str标签到int序号
    :param gt_out：为True表示要保人工标签
    :param writeft_out: 为True表示要保存识别过程中网络提取到的特征值（此功能内置于network中）
    """
    videoPath = oriDataPackage + fileName + '.avi'
    CSVPath = oriDataPackage + fileName + '.csv'
    t = int(time.time())
    dt = time.strftime("%m%d", time.localtime(t))
    output_dir0 = outpackage_ori + str(dt)
    # output_dir0 = outpackage_ori   # 2022-08-11省去 时间dt
    if not os.path.exists(output_dir0):
        os.mkdir(output_dir0)

    # 输出特征值用于ASFormer的训练
    if writeft_out:
        prob_all, lables_all, csvused_all, feature_all = inference_LVAR_lstm2.Rec_main(videoPath,
                                                                                                            CSVPath,
                                                                                                            lablestxt,
                                                                                                            modeltarPath,
                                                                                                            feature_out=writeft_out)
        output_dir = output_dir0 + '/' + modelRunName + '_feature'
        res_path = output_dir + '/' + fileName + '.npy'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        feature_all = list(zip(*feature_all))  # 输出格式为featurenums*frames
        np.save(res_path, feature_all)

        #  输出groundtruth用于ASFormer的训练，注意需要与特征值的维度一一对应
        if gt_out:
            gt_output_dir = output_dir0 + '/' + modelRunName + '_groundtruth'
            gt_path = gt_output_dir + '/' + fileName + '.txt'
            if not os.path.exists(gt_output_dir):
                os.mkdir(gt_output_dir)
            ori_ActClass = np.loadtxt(open(CSVPath), dtype=str, delimiter=',',
                                      usecols=20)  # np.save(gt_path, ori_ActClass)
            with open(gt_path, "w") as f:
                for used_i in csvused_all:
                    f.write(ori_ActClass[used_i])
                    f.write('\n')

        output_dir_Rec = output_dir0 + '/' + modelRunName
        if not os.path.exists(output_dir_Rec):
            os.mkdir(output_dir_Rec)
        res_path = output_dir_Rec + '/' + fileName + '_RecAns.csv'
        inference_LVAR_lstm2.writeans_csv(res_path, CSVPath, lables_all, prob_all, csvused_all,
                                                    lablestxt=lablestxt, oriclass=needconcat)
    else:
        if model == 'MultiFus_RGB_Motion2_LVAR_lstm2':
            prob_all, lables_all, csvused_all = inference_LVAR_lstm2.Rec_main(videoPath, CSVPath,
                                                                                                   lablestxt,
                                                                                                   modeltarPath,
                                                                                                   feature_out=writeft_out)
        output_dir = output_dir0 + '/' + modelRunName
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        res_path = output_dir + '/' + fileName + '_RecAns.csv'
        inference_LVAR_lstm2.writeans_csv(res_path, CSVPath, lables_all, prob_all, csvused_all,
                                                    lablestxt=lablestxt, oriclass=needconcat)


def read_recres(fileName, filePackage,
                lablestxt=r'C:\YHY\Python_code\HY_ActionRec\MultiFus_RGB_Motion2\dataloaders\0504_VideoandMocapData_lables.txt',
                de_sensitivity=False):
    rec_csvpath_ori = filePackage + fileName
    recFileName = fileName.split('.')[0]
    with open(lablestxt, 'r') as f:
        labels_all = f.readlines()  # 所有动作标签（带1，2，3序号，要class_names[label].split(' ')[-1].strip()提取标签名）
        f.close()
    rec_hand = np.loadtxt(rec_csvpath_ori, dtype=int, delimiter=',',
                          usecols=len(labels_all) + 22) - 1  # 手工标注的结果  # 注意这里的dtype为int
    rec_model_res = np.loadtxt(rec_csvpath_ori, dtype=int, delimiter=',', usecols=len(labels_all) + 1) - 1  # 模型识别结果

    rec_model_probs = np.zeros([len(rec_model_res), len(labels_all)])
    for i in range(len(labels_all)):
        rec_model_probs[:, i] = np.loadtxt(rec_csvpath_ori, dtype=np.float32, delimiter=',', usecols=i + 1)  # 概率值

    if de_sensitivity:
        rec_model_res, rec_model_probs = post_process.de_sensitivity(rec_model_res,
                                                                                          rec_model_probs, cliplen=16)
    return [rec_hand, rec_model_res, rec_model_probs]


def eval(tmp_Recres, fileNs, thr=0):
    """
    :param tmp_Recres: [[rec_hand_i, rec_model_res_i, rec_model_probs_i],...]，是一系列[rec_hand, rec_model_res, rec_model_probs]构成的列表
    :param fileNs: 对应读取的识别结果csv文件名
    :param thr: 人工标记切换帧前后thr帧都可以为切换点
    :return acc:正确帧数/总帧数。
    acc_edit: 编辑距离(莱文斯坦距离)，考量从分割结果替换到真实结果所需的最小操作次数，即考量差异性。(Leetcode动态规划求解)。
    fscore_all: Segmental F1-Score。
    """
    right_all = 0
    wrong_all = 0
    edit_all = 0
    overlap = [.1, .25, .5]
    fscore_all = np.zeros(len(overlap))
    for i in range(len(tmp_Recres)):
        right_i = 0
        wrong_i = 0
        rec_hand_i = tmp_Recres[i][0]
        rec_model_i = tmp_Recres[i][1]
        num_Preparation = 0  # 记录人工标记为Preparation的帧数
        # 当人工标记切换帧前后thr帧都可以为切换点时，因此而改为正确的帧数
        use_1 = 0
        use_0 = 0
        # rec_model_pro_i = tmp_Re[i][2]
        for j in range(len(rec_hand_i)):
            if rec_hand_i[j] == rec_model_i[j]:
                right_i += 1
            # TODO: 确认标签是从0开始还是从1开始---从0开始的
            elif rec_hand_i[j] == 3:  # 人工标记为Preparation的不管识别结果为何都按对
                right_i += 1
                num_Preparation += 1
                rec_hand_i[j] = rec_model_i[j]
            elif j < len(rec_hand_i) - thr and rec_hand_i[j + thr] == rec_model_i[j]:
                right_i += 1
                use_1 += 1
            elif j > thr and rec_hand_i[j - thr] == rec_model_i[j]:
                right_i += 1
                use_0 += 1
            else:
                wrong_i += 1
        if thr > 0:
            print(use_0, use_1)

        tp, fp, fn, fscore = np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(rec_hand_i, rec_model_i, overlap[s])
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1
        for s in range(len(overlap)):
            precision = tp[s] / float(tp[s] + fp[s])
            recall = tp[s] / float(tp[s] + fn[s])
            fscore[s] = 2.0 * (precision * recall) / (precision + recall)
            fscore[s] = np.nan_to_num(fscore[s]) * 100
            fscore_all[s] = fscore_all[s] + fscore[s]
            # print('F1@%0.2f: %.4f' % (overlap[s], fscore[s]))

        acc_i = float(right_i / (wrong_i + right_i)) * 100.0
        # TODO: edit的计算应该采用逐帧计算？(先按照公共的用edit_score这个函数)
        # acc_i_edit = levenstein(rec_model_i, rec_hand_i, True)
        acc_i_edit = edit_score(rec_model_i, rec_hand_i)
        print(fileNs[i], "Acc, Acc_edit, F1@0.10,F1@0.25,F1@0.50: %0.2f, %0.2f, %0.2f, %0.2f, %0.2f" % (
acc_i, acc_i_edit, fscore[0], fscore[1], fscore[2]))
        right_all = right_all + right_i
        wrong_all = wrong_all + wrong_i
        edit_all = edit_all + acc_i_edit
    print('num_Preparation:', num_Preparation)
    print("Num of frames_eval_all:", wrong_all + right_all)
    acc = float(right_all / (wrong_all + right_all)) * 100.0
    acc_edit = float(edit_all / len(tmp_Recres))
    print("Acc:", acc)
    print("Acc_edit:", acc_edit)
    for s in range(len(overlap)):
        fscore_all[s] = fscore_all[s] / len(tmp_Recres)
        print('F1@%0.2f: %.4f' % (overlap[s], fscore_all[s]))
    return acc, acc_edit, fscore_all


def levenstein(p, y, norm=False):
    """
    计算编辑距离。
    :param p: 识别结果model_res
    :param y: 真实结果ground_truth(hand_res)
    :param norm:
    :return: score
    """
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row + 1, n_col + 1], np.float)
    for i in range(m_row + 1):
        D[i, 0] = i
    for i in range(n_col + 1):
        D[0, i] = i
    for j in range(1, n_col + 1):
        for i in range(1, m_row + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1,
                              D[i, j - 1] + 1,
                              D[i - 1, j - 1] + 1)
    if norm:
        score = (1 - D[-1, -1] / max(m_row, n_col)) * 100.0
    else:
        score = D[-1, -1]
    return score


def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []  # labels是动作标签
    starts = []  # starts是动作开始的位置
    ends = []  # end是动作结束的位置
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:  # 如果标签第一帧不是"background"，labels[0]=第一帧的动作
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i + 1)
    return labels, starts, ends


def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap=0.1, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)
    tp = 0
    fp = 0
    hits = np.zeros(len(y_label))
    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0 * intersection / union) * ([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()
        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


def ASFormer_predict(_datanames, _resdir_save, _features_path_save, modelpath,
                     mapping_file='./dataloaders/mapping.txt'):
    # import os
    import model.ASFmodel
    import torch

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    action = "predict_save"
    num_layers = 10
    num_f_maps = 64
    features_dim = 4126
    channel_mask_rate = 0.4

    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])
    index2label = dict()
    for k, v in actions_dict.items():
        index2label[v] = k
    num_classes = len(actions_dict)

    trainer = model.ASFmodel.Trainer(num_layers, 2, 2, num_f_maps, features_dim, num_classes, channel_mask_rate)
    if action == "predict_save":
        # 保存结果如同MMC3D一样
        # resdir_save = r'C:\YHY\Python_code\HY_ActionRec\Transit_area\res\0810'  # 输出目录
        # features_path_save = '.Transit_area/0811/R14_4_feature/'  # 输入特征所在文件夹
        sample_rate_save = 1
        # datanames = ['ct10_00']
        trainer.predict_save(_resdir_save, actions_dict, sample_rate_save, _features_path_save, _datanames,
                             modelpath=modelpath)
    else:
        print("error")


def post_show(filename, rec_model_res,rec_model_probs, Traj_qp, videoPath, CM_ornot = False, labelpath='./dataloaders/mapping.txt'):
    # CM_ornotFalse表示不绘制混淆矩阵(当没有手工标记数据时，，)
    display = True  # 为True表示需要同时显示视频和轨迹----目前没有写false的后面的代码，不要修改此值

    # 输入(修改videoPath和fileName,savepackage-->csv文件)
    # filename = 'ct10_0'
    # recFileName = '0601' + '//' + 'R14_2//' + filename + '_RecAns'
    # videoPath = 'C://Users//G314-Optitrack//Desktop//Demonstration_data//testdata_2//' + filename + '.avi'
    # savepackage = r'C:\YHY\Python_code\HY_ActionRec\MultiFus_RGB_Motion2\res'
    # rec_csvpath_ori = savepackage + '//' + recFileName + '.csv'  # 动作识别结果所在的文件位置
    # labelpath = './dataloaders/0504_VideoandMocapData_lables.txt'  # 动作标签文件
    # labelpath = 'C:\YHY\Python_code\HY_ActionRec\ASFormer_main\data\Assembly0728\mapping.txt' #ASFormer的

    with open(labelpath, 'r') as f:
        labels_all = f.readlines()  # 所有动作标签（带1，2，3序号，要class_names[label].split(' ')[-1].strip()提取标签名）
        f.close()

    # # qw, qx, qy, qz,  px, py, pz
    # Traj_qp = np.loadtxt(ori_csv, dtype=np.float32, delimiter=',',
    #                      usecols=(len(labels_all) + 3, len(labels_all) + 4, len(labels_all) + 5, len(labels_all) + 6,
    #                               len(labels_all) + 7, len(labels_all) + 8, len(labels_all) + 9))


    # # 视频加bar一起显示
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
        ax1.set_title('装配运动轨迹图')
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
            # probs_i = torch.from_numpy(rec_model_probs[i])
            # label = np.argmax(probs_i)
            # label_1 = torch.max(probs_i, 0)[1].detach().cpu().numpy()
            retaining, frame = cap1.read()
            if not retaining and frame is None:
                continue
            cv.putText(frame, (labels_all[rec_model_res[i]].split(' ')[-1].strip() + " : %.4f" % rec_model_probs[i]), (365, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 0, 255), 2)
            # 进度条：
            cv.rectangle(barImg, (pi_x, pi_y), (pi_x + w_unit, pi_y - h_choose[rec_model_res[i]]),
                         color[rec_model_res[i]], thickness=-1)
            pi_x = pi_x + w_unit

            cv.namedWindow('result', cv.WINDOW_NORMAL)
            # 改变窗口位置：left top
            # cv.moveWindow("result", -1000, 0)
            cv.moveWindow("result", 0, 0+50)

            # 拼接进度条和视频:
            cv.imshow('result', np.vstack(
                (frame, cv.resize(barImg, (frame.shape[1], 30)))))  # 将源视频与进度条合并显示，进度条大小resize到宽frame.shape[1],高10
            cv.waitKey(1)  # 单位为ms

            # 同时刷新轨迹三维图
            if i > 1:
                # 绘制三维MoCapTraj图
                ax1.plot3D(Traj_qp[2*i-2:2*i+1, 4], Traj_qp[2*i-2:2*i+1, 5], Traj_qp[2*i-2:2*i+1, 6],
                           c=color_traj[rec_model_res[i]])  # 连续线c='blue'
                # 散点图 c=rec_model_res[i]
                # ax1.scatter3D(Traj_qp[i, 4], Traj_qp[i, 5], Traj_qp[i, 6], s=4, c=color_traj[rec_model_res[i]])
                plt.pause(0.00001)
                curm= plt.get_current_fig_manager()
                curm.window.setGeometry(640,38+50, 520, 470) #x,y,dx,dy

                # 保存barImg和MoCapTraj图
    # bar_filepath = savepackage + '//' + filename + '_bar'
    # traj_filepath = savepackage + '//' + filename + '_MoCapTraj'
    # cv.imwrite(bar_filepath + '.png', barImg, [cv.IMWRITE_PNG_COMPRESSION, 0])
    # plt.savefig((traj_filepath + '.png').format(ax1))
    # print('...SAVED')
    cap1.release()
    cv.destroyAllWindows()
    plt.ioff()  # 关掉交互模式，进入阻塞模式（默认的模式）
    # plt.show()  #todo 显示图，不至于让MoCapTraj图直接关掉.#若要不直接关闭就解注释

def plot_segment_bars_YHY(save_path, *labels):
    num_pics = len(labels)+1
    color_map = plt.get_cmap('seismic')

    axprops = dict(xticks=[], yticks=[], frameon=False)
    barprops = dict(aspect='auto', cmap=color_map,
                    interpolation='nearest', vmin=0)
    fig = plt.figure(figsize=(15, num_pics * 1.5))

    interval = 1 / (num_pics + 1)
    for i, label in enumerate(labels):
        i = i + 1
        ax1 = fig.add_axes([0, 1 - i * interval, 1, interval])
        ax1.imshow([label], **barprops)

    # ax4 = fig.add_axes([0, interval, 1, interval])
    # ax4.set_xlim(0, len(confidence))
    # ax4.set_ylim(0, 1)
    # ax4.plot(range(len(confidence)), confidence)
    # ax4.plot(range(len(confidence)), [0.3] * len(confidence), color='red', label='0.5')

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()

def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content

def segment_bars_YHY(outdir_save, resnames, res_package, gt_package, checkmodel):
    mapping_file = './dataloaders/mapping.txt'
    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])
    label2index = dict()
    for k, v in actions_dict.items():
        label2index[k] = v
    num_classes = len(actions_dict)
    for vid in resnames:
        gt_file = gt_package + vid + '.txt'
        gt_content = read_file(gt_file).split('\n')[0:-1]
        recog_file = res_package + vid + '_' + checkmodel + '.csv'
        recog_content = np.loadtxt(open(recog_file), dtype=str, delimiter=',', usecols=2)
        math_gt = []
        math_rec = []
        for res_gt in gt_content:
            math_gt.append(label2index[res_gt])
        for res_rec in recog_content:
            math_rec.append(label2index[res_rec])
        savepath = outdir_save + "\\" + vid + '.png'
        plot_segment_bars_YHY(savepath, math_gt, math_rec)


def eval_asformer(resnames, res_package, gt_package, checkmodel):
    mapping_file = './dataloaders/mapping.txt'
    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
    correct = 0
    total = 0
    edit = 0
    ACC1211 = 0
    EDIT1211 = 0
    F11211 = np.array([0, 0, 0], dtype=float)

    for vid in resnames:
        gt_file = gt_package + vid + '.txt'
        gt_content = read_file(gt_file).split('\n')[0:-1]
        recog_file = res_package + vid + '_' + checkmodel + '.csv'
        recog_content = np.loadtxt(open(recog_file), dtype=str, delimiter=',', usecols=2)
        correct_i = 0
        for i in range(len(recog_content)):
            total += 1
            if gt_content[i] == "Preparation":
                # todo 如果人工标记的为Preparation，则不管识别结果为何，都认为其对，并修改其为Preparation
                recog_content[i] = "Preparation"
                correct += 1
                correct_i += 1
                # if recog_content[i] == "Preparation":
                #     correct += 1
                #     correct_i += 1
                # else:
                #     total -= 1
            else:
                if gt_content[i] == recog_content[i]:
                    correct += 1
                    correct_i += 1

        edit_i = edit_score(recog_content, gt_content)
        edit += edit_i
        tp_i, fp_i, fn_i = np.zeros(3), np.zeros(3), np.zeros(3)
        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s])
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1
            tp_i[s] += tp1
            fp_i[s] += fp1
            fn_i[s] += fn1

        acc_i = 100 * float(correct_i) / len(recog_content)
        f1s_i = np.array([0, 0, 0], dtype=float)
        for s in range(len(overlap)):
            precision_i = tp_i[s] / float(tp_i[s] + fp_i[s])
            recall_i = tp_i[s] / float(tp_i[s] + fn_i[s])
            f1_i = 2.0 * (precision_i * recall_i) / (precision_i + recall_i)
            f1_i = np.nan_to_num(f1_i) * 100
            f1s_i[s] = f1_i
            F11211[s] += f1_i
        print(vid.split('.')[0], " Acc, Acc_edit, F1@0.10,F1@0.25,F1@0.50: %0.2f, %0.2f, %0.2f, %0.2f, %0.2f" % (
            acc_i, edit_i, f1s_i[0], f1s_i[1], f1s_i[2]))
        ACC1211 += acc_i
        EDIT1211 += edit_i

    ACC1211 = float(ACC1211)/len(resnames)
    EDIT1211 = float(EDIT1211)/len(resnames)
    F11211[0] = float(F11211[0])/len(resnames)
    F11211[1] = float(F11211[1])/len(resnames)
    F11211[2] = float(F11211[2])/len(resnames)
    # print("Acc: %.4f  Edit: %4f  F1@10,25,50 " % (ACC1211, EDIT1211), F11211)

    acc = 100 * float(correct) / total
    edit = (1.0 * edit) / len(resnames)
    f1s = np.array([0, 0, 0], dtype=float)
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s] + fp[s])
        recall = tp[s] / float(tp[s] + fn[s])
        f1 = 2.0 * (precision * recall) / (precision + recall)
        f1 = np.nan_to_num(f1) * 100
        f1s[s] = f1
    return acc, edit, f1s, ACC1211,EDIT1211,F11211


if __name__ == '__main__':
    oriDataPackage = './Demonstration_data/endtest/0912all/'
    outpackage_4 = './Transit_area//'
    labelpath = './dataloaders/0504_VideoandMocapData_lables.txt'
    stage = 2
    thr = 4  
    de_sensitivity_ornot = True 

    day = '0923'
    modeltarPath = r'C:\Users\G314-Optitrack\Desktop\ActionRec\run\run14_4\models\C3D_MoCapLvarLstm2-ContiACT_epoch-179.pth.tar'
    modelRunName = 'R14_4'
    network = 'MultiFus_RGB_Motion2_LVAR_lstm2'

    if stage == 1:
        for file in os.listdir(oriDataPackage):
            if file.split('.')[-1] == 'avi':
                fileName = file.split('.')[0]
                rec_save(fileName, modeltarPath, modelRunName, oriDataPackage, model=network,
                             outpackage_ori=outpackage_4, lablestxt=labelpath, gt_out=True, writeft_out=True)
    elif stage == 2:
        tmp_Re = []
        recPackage = 'C:\\YHY\\Python_code\\HY_ActionRec\\Transit_area\\' + day + '\\' + modelRunName + '\\'  
        fileNs = os.listdir(recPackage)
        for fileN in fileNs:
            print(fileN)
            # if fileN.split('.')[0][1] == 'C':
            #     continue
            tmp_Re.append(read_recres(fileN, recPackage, lablestxt=labelpath, de_sensitivity=de_sensitivity_ornot))
        print("Num of videos_eval:", len(tmp_Re))
        eval(tmp_Re, fileNs, thr=thr)
    elif stage == 3:
        tmp_Re = []
        recPackage = './res/' + day + '/' + modelRunName + '/' 
        fileNs = os.listdir(recPackage)
        for fileN in fileNs:
            # print(fileN)
            tmp_Re.append(read_recres(fileN, recPackage, lablestxt=labelpath, de_sensitivity=de_sensitivity_ornot))
        print("Num of videos_eval:", len(tmp_Re))
        # color_traj顺序为(r,g,b)，大小0-1，也可 color_traj = ['b','g','r','c','m','y','k']
        color_traj = [[28 / 255, 207 / 255, 199 / 255], [251 / 255, 199 / 255, 141 / 255],
                      [167 / 255, 214 / 255, 118 / 255], [184 / 255, 161 / 255, 152 / 255],
                      [168 / 255, 222 / 255, 224 / 255], [205 / 255, 104 / 255, 57 / 255]]
        with open(labelpath, 'r') as f:
            labels_all = f.readlines()  
            f.close()
        for i in range(len(tmp_Re)):
            rec_hand_i = tmp_Re[i][0]
            rec_model_i = tmp_Re[i][1]
            rec_model_pro_i = tmp_Re[i][2]
            # figi, (axi,axi2) = plt.subplots(2,1)
            figi, axi = plt.subplots(figsize=(18, 8))

            axi.set_title('Probability Graph-' + str(fileNs[i].split('_')[0]) + '-' + str(fileNs[i].split('_')[1]),
                          fontsize=15)
            # axi.set_title('MMC3D', fontsize=15)

            plt.xticks(fontsize=15) 
            plt.yticks(fontsize=15)
            axi.set_xlabel(..., fontsize=18)  # 设置轴标题字体大小
            axi.set_ylabel(..., fontsize=18)
            axi.set_xlabel('t')
            axi.set_ylabel('P')
            t = np.linspace(0, len(rec_model_pro_i), len(rec_model_pro_i))
            for j in range(len(rec_model_pro_i[0, :])):
                axi.plot(t, rec_model_pro_i[:, j], color=color_traj[j], label=str(labels_all[j]))
            for modelRecj in range(len(rec_model_i)):
                axi.plot([modelRecj - 0.5, modelRecj + 0.5], [1.05, 1.05], color=color_traj[rec_model_i[modelRecj]],
                         linewidth=5)
            for handj in range(len(rec_hand_i)):
                axi.plot([handj - 0.5, handj + 0.5], [1.1, 1.1], color=color_traj[rec_hand_i[handj]], linewidth=5)

            axi.legend()
            plt.show()  # 图形可视化
    elif stage == 4:
        outdir_save = './Transit_area/0923/res_ASF1211'
        # datanames = ['cr5_3', 'cr5_4', 'ct10_00', 'ct9_4', 'cm5_0', 'cm11_6']
        # datanames = ['cr7_0']
        datanames = []
        npy_time = "0923"
        ori_npy_package = outpackage_4 + npy_time + "\\" + modelRunName + "_feature\\"
        for file in os.listdir(ori_npy_package):
            if file.split('.')[-1] == 'npy':
                datanames.append(file.split('.')[0])

        if not os.path.exists(outdir_save):
            os.mkdir(outdir_save)
        inputfeatures_path_save = 'Transit_area\\' + npy_time + "\\" + modelRunName + "_feature\\"
        # ASFmodelpath = r'C:\YHY\Python_code\HY_ActionRec\ASFormer_main\models\Assembly0728\split_1\epoch-50.model'
        ASFmodelpath = './ASFormer_main\models\Assembly0728\split_2\epoch-30.model'
        ASFormer_predict(datanames, outdir_save, inputfeatures_path_save, ASFmodelpath, mapping_file='./dataloaders/mapping.txt')
    elif stage == 5:
        # labelpath_Chi = r'C:\YHY\Python_code\HY_ActionRec\finalmapping_chinese.txt'
        outdir_save = './Transit_area/0923/res_ASF1'
        # datanames = ['cr5_3', 'cr5_4', 'ct10_00', 'ct9_4', 'cm5_0', 'cm11_6']
        # datanames = ['cr17_2'，'cTT6_3', 'cr11_3', 'cCam6_4']
        datanames = [ 'cTT2_2', 'cr13_3', 'cCam6_7']
        # todo
        res_package = outdir_save
        for k in range(len(datanames)):
            filename = datanames[k]
            res_csv = res_package+"\\"+ filename +"_3.csv"
            ori_csv = oriDataPackage + filename+".csv"
            ori_video = oriDataPackage + filename+".avi"
            rec_model_res = np.loadtxt(res_csv, dtype=np.int, delimiter=',', usecols=1)  
            rec_model_probs = np.loadtxt(res_csv, dtype=np.float32, delimiter=',', usecols=0)
            Traj_qp = np.loadtxt(ori_csv, dtype=np.float32, delimiter=',', usecols=(1, 2, 3, 4,  5, 6, 7))
            post_show(filename, rec_model_res, rec_model_probs, Traj_qp, ori_video, CM_ornot = True)
    elif stage == 6:
        res_time = "0923"
        gt_package = outpackage_4 + res_time + "\\" + modelRunName + "_groundtruth\\"
        # gt_package = outpackage_4 + res_time + "\\" + modelRunName + "_groundtruth_zhao\\"
        res_package = outpackage_4 + res_time + "\\res_ASF1\\"
        # res_package = outpackage_4 + res_time + "\\1210final_changedself\\YH\\"
        checkmodel = '3' 
        resnames = []
        size = 0
        for file in os.listdir(gt_package):
            # name = 'T'
            # name = 'r'
            # name = 'C'
            # if file.split('.')[0][1] != name:
            #     continue
            resnames.append(file.split('.')[0])
            size = size + 1
        acc_all, edit_all, f1s_all, ACC1211, EDIT1211, F11211 = eval_asformer(resnames, res_package, gt_package, checkmodel)
        # print("Acc: %.4f  Edit: %4f  F1@10,25,50 " % (acc_all, edit_all), f1s_all)
        print("Acc: %.4f  Edit: %4f  F1@10,25,50 " % (ACC1211, EDIT1211), F11211)
        print(size)


