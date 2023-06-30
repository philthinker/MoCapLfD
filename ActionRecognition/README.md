1. Enviroment
   Pytorch == 1.1.0, torchvision == 0.3.0, python == 3.6, CUDA=10.1
2. Train：
   You can retrain the 'C3D_LVAR_LSTM' by yourself with train_LVAR_lstm_Continuous.py.(\ActionRec\run\run14_4\models)
   Please refer to https://github.com/ChinaYi/ASFormer to train ASFormer.(\ActionRec\ASFormer_main\models)
2. Test：
    You can use prog_all.py in which:
    Stage 1 is used for inference, stage 2 is used for evaluation, stage 3 is used for probability change display,
    Perform ASFormer reprediction when stage is 4.
    In stage 5, a video plus trajectory demonstration is performed.
    And for stage 6, an asformer result evaluation is performed.