import os
import argparse

import torch
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Normalize, RandomResizedCrop, RandomApply, Resize, CenterCrop, RandomAffine

from transforms import *
from dataloader import *
from models import *
from utils import seed_everything, find_th

import nsml
from nsml.constants import DATASET_PATH, GPU_NUM

import warnings
warnings.filterwarnings('ignore')

model_list = (
        ('KHD024/Breast_Pathology/565', 'best_score_fold0'),
        ('KHD024/Breast_Pathology/554', 'best_score_fold1'),
        ('KHD024/Breast_Pathology/565', 'best_score_fold2'),
        ('KHD024/Breast_Pathology/572', 'best_score_fold3'),
        ('KHD024/Breast_Pathology/572', 'best_score_fold4'),
    )

def model_infer(image_path, args):

    # fix seed for train reproduction
    seed_everything(args.SEED)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    default_transforms = transforms.Compose([transforms.Resize(args.input_size)])
    test_tranfsorms = get_transform(target_size=(args.input_size, args.input_size),
                                        transform_list=args.valid_augments,
                                        augment_ratio=args.augment_ratio,
                                        is_train=False)  
    
    test_dataset = PathDataset(image_paths=image_path, 
                                labels=None, 
                                default_transforms=default_transforms, 
                                transforms=test_tranfsorms, 
                                is_test=True)

    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        
    total_pred = []

    # ensemble models
    for i, (load_session, load_checkpoint) in enumerate(model_list):
        try:
            nsml.load(checkpoint=load_checkpoint, session=load_session)
            print(f'{i}th model loaded {load_session} {load_checkpoint}')
        except:
            print(f'{i}th model load cancel')
        
        model.to(device)
        model.eval()

        fold_pred = []
        
        # test time augmentation
        for _ in range(args.tta):
            tta_pred = []
            with torch.no_grad():
                for i, images in enumerate(test_loader):
                    output = torch.sigmoid(model(images.to(device))).cpu().detach().numpy()
                    tta_pred.append(output)
            tta_pred = np.concatenate(tta_pred).flatten()
            fold_pred.append(tta_pred) 
        total_pred.append(np.array(fold_pred)) 

    total_pred = np.concatenate(total_pred) 
    total_pred = np.mean(total_pred**args.power, axis=0)

    threshold = 0.5
    total_pred = np.where(np.array(total_pred) >= threshold, 1, 0)
    total_pred = total_pred.astype(np.int64)

    return total_pred

    ########################################################################################################


def bind_model(model, args):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        torch.save(model.state_dict(),os.path.join(dir_name, 'best_model'))
        print('model saved!')

    def load(dir_name):
        model.load_state_dict(torch.load(os.path.join(dir_name, 'best_model')))
        print('model loaded!')

    def infer(image_path):
        pred = model_infer(image_path, args)
        return pred

    nsml.bind(save=save, load=load, infer=infer)


if __name__ == "__main__":

    ########## ENVIRONMENT SETUP ############
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument

    arg('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    arg('--iteration', type=str, default='0', help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    arg('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')

    arg('--SEED', type=int, default=43)
    arg('--model', type=str, default='efficientnet_b3')
    arg('--input_size', type=int, default=512)
    arg('--batch_size', type=int, default=8)
    arg('--num_workers', type=int, default=4)
    arg('--valid_augments', default='horizontal_flip, random_rotate', type=str)
    arg('--augment_ratio', default=0.5, type=float, help='probability of implementing transforms')
    arg('--tta', type=int, default=1, help='test time augmentation')
    arg('--pretrained', default=False, type=bool, help='download pretrained model')
    arg('--num_classes', type=int, default=1)
    arg('--power', type=int, default=1)

    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("device", device)

    global model
    model = build_model(args, device)
    bind_model(model, args)

    if args.mode == 'train':
        nsml.save('sillim')

    if args.pause: 
        print('Inferring Start...')
        nsml.paused(scope=locals())