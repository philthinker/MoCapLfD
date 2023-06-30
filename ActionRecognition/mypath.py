class Path(object):
    @staticmethod
    def db_dir(database):
        if database == '0330_VideoandMocapData':
            # folder that contains class labels
            root_dir = 'C:\\Users\\G314-Optitrack\\Desktop\\Demonstration_data\\all_Video_MoCap_gloves2'
            output_dir = 'C:\\YHY\\Python_code\\HY_ActionRec\\MultiFus_RGB_Motion2\\data_processed\\0330_VideoandMocapData'
            return root_dir, output_dir
        elif database == '0504_VideoandMocapData':
            # folder that contains class labels
            root_dir = 'C:\\Users\\G314-Optitrack\\Desktop\\Demonstration_data\\all_Video_MoCap_gloves3'
            output_dir = 'C:\\YHY\\Python_code\\HY_ActionRec\\MultiFus_RGB_Motion2\\data_processed\\0504_VideoandMocapData'
            return root_dir, output_dir
        elif database == 'ContiACT':
            root_dir = 'C:\\Users\\G314-Optitrack\\Desktop\\Demonstration_data\\ContiACT'
            output_dir = 'C:\\YHY\\Python_code\\HY_ActionRec\\MultiFus_RGB_Motion2\\data_processed\\ContinuousAct'
            return root_dir, output_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def C3D_model_dir():
        return r'C:\YHY\Python_code\HY_ActionRec\MultiFus_RGB_Motion2\run\run16_10\models\C3D-0504_VideoandMocapData_epoch-29.pth.tar'

    @staticmethod
    def LSTM_model_dir():
        return r'C:\YHY\Python_code\HY_ActionRec\MultiFus_RGB_Motion2\run\run15_4\models\Lstm-0504_VideoandMocapData_epoch-59.pth.tar'

    @staticmethod
    def BiLSTM_model_dir():
        return r'C:\YHY\Python_code\HY_ActionRec\MultiFus_RGB_Motion2\run\run15_9\models\BiLstm-0504_VideoandMocapData_epoch-95.pth.tar'
