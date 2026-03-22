import argparse



def config(epochs=100,img_size=224,batch_size=10,num_gpus=1,version="v0",decay_epoch=30,lr=1e-4,is_Test=True,validate_dateset="NLPR"):
    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--Training', default=False, type=bool, help='Training or not')
    parser.add_argument('--save_edge_flag', default=False, type=bool, help='Training or not')
    parser.add_argument('--save_depth_quality_flag', default=False, type=bool, help='Training or not')
    parser.add_argument('--data_root', default='../Data/', type=str, help='data path')

    parser.add_argument('--rgb_label_root', type=str, default= '../Data/TrainDataset/RGB/',
                        help='the training rgb images root')
    parser.add_argument('--depth_label_root', type=str, default=  '../Data/TrainDataset/depth/',
                        help='the training depth images root')
    parser.add_argument('--depth_quality_label_root', type=str, default= '../Data/TrainDataset/depth_quality_pseudo_label/',
                        help='the training depth images root')
    parser.add_argument('--gt_label_root', type=str, default= '../Data/TrainDataset/GT/',
                        help='the training gt images root')

    parser.add_argument('--test_path', type=str, default='../Data/TestDataset/',
                        help='test dataset path')

    parser.add_argument('--validate_dir_img', type=str, default='../TrainDataset/GT/',
                        help='the training gt images root')
    parser.add_argument('--validate_dir_gt', type=str, default='../TrainDataset/GT/',
                        help='the training gt images root')

    parser.add_argument('--img_size', default=img_size, type=int, help='network input size')
    parser.add_argument('--top_num', default=3, type=int, help='save top num  model')
    parser.add_argument('--batch_size', default=batch_size, type=int, help='batch_size')
    parser.add_argument('--is_load_model', default=False, type=bool, help='load model')

    # training dataset
    parser.add_argument('--save_model_dir', default='./checkpoint/' + version, type=str, help='save model path')
    parser.add_argument('--save_best_mae_model_dir', default='checkpoint/' + version + "/mae/", type=str, help='save model path')
    parser.add_argument('--save_test_mae_model_dir', default='checkpoint/' + version + "/mae_test/", type=str, help='save model path')
    parser.add_argument('--project_dir', default="", type=str, help='save model path')

    parser.add_argument('--gpu_id', type=str, default=num_gpus, help='train use gpu')
    parser.add_argument('--lr', default=lr, type=int, help='learning rate')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_epoch', default=decay_epoch, type=int, help='train_steps')
    parser.add_argument('--is_pass_test', type=bool, default=True)

    parser.add_argument('--trainset', default='NJUD+NLPR', type=str, help='Trainging set')
    parser.add_argument('--epochs', default=epochs, type=int, help='epochs')
    parser.add_argument('--test_paths', type=str, default='NLPR+NJUD+STERE+SIP')

    # test
    parser.add_argument('--Testing', default=False, type=bool, help='Testing or not')
    parser.add_argument('--save_test_path_root', default='preds/' + version + "/", type=str,
                        help='save saliency maps path')

    # evaluation
    parser.add_argument('--Evaluation', default=False, type=bool, help='Evaluation or not')
    parser.add_argument('--methods', type=str, default='RGBD_VST', help='evaluated method name')
    parser.add_argument('--save_dir', type=str, default='./log/' + version + "/", help='path for saving result.txt')
    parser.add_argument('--cuda', default=True, type=bool, help='use cuda or not')
    parser.add_argument('--version', type=str, default=version, help="code version")
    parser.add_argument('--pre_model_path', type=str, default="./pretrain/mae_pretrain_vit_large.pth",
                        help="code version")
    parser.add_argument('--tensorboard_log_path', type=str, default="/root/tf-logs/VST/" + version + "/",
                        help="code version")
    args = parser.parse_args()
    return args




