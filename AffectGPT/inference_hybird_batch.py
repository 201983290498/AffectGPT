import os
import glob
import argparse
from llamafactory import data
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
from datetime import datetime
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

import decord
decord.bridge.set_bridge('torch')

from my_affectgpt.tasks import *
from my_affectgpt.models import *
from my_affectgpt.runners import *
from my_affectgpt.processors import *
from my_affectgpt.datasets.builders import *
from my_affectgpt.common.config import Config
from my_affectgpt.common.dist_utils import get_rank
from my_affectgpt.common.registry import registry
from my_affectgpt.conversation.conversation_video import Chat
from my_affectgpt.datasets.builders.image_text_pair_builder import * # 加载所有dataset cls
from torch.utils.data import DataLoader

import config
from toolkit.utils.read_files import *
from tqdm.auto import tqdm

# 采用的是这个文件下存储数量最多的 root
def search_for_ckpt_root(root_candidates):
    if len(root_candidates) == 0:
        return ''
    
    # 找到 files 最多的 root
    maxcount = 0
    targetroot = ''
    for root in root_candidates:
        count = len([path for path in os.listdir(root) if path.startswith('checkpoint_')])
        print (root, '==>', count)
        if count > maxcount:
            maxcount = count
            targetroot = root
    print ('================================================')
    print (f'Targetroot: epoch range: 0-{maxcount-1}')
    
    # 打印最后一个文件的创建时间 for targetroot
    last_file = sorted(glob.glob(targetroot + '/checkpoint*'))[-1]
    file_stat = Path(last_file).stat()
    creation_time = file_stat.st_ctime
    print("Targetroot: Last ckpt creation time:", datetime.fromtimestamp(creation_time))
    print ('================================================')
    return targetroot


# case1: 默认 => last epoch
# case2: 指定 inference_cfg.test_epoch == a; 那就只跑这个 epoch 下的结果
# case3: 指定 inference_cfg.test_epochs == a-b; 跑最后一个
def get_ckpt3_candidates(ckpt3_root, inference_cfg):
    if inference_cfg.test_epoch != 'xxx':
        cur_epoch = inference_cfg.test_epoch
        ckpts = glob.glob("%s/*%06d*.pth" %(ckpt3_root, int(cur_epoch)))
        assert len(ckpts) == 1, 'Error: (ckpt, epoch) combination is not exists or contain multiple candidates!'
        return [ckpts[0]]
    
    elif inference_cfg.test_epochs == 'xxx-xxx':
        last_ckpt = sorted(glob.glob("%s/*.pth" %(ckpt3_root)))[-1]
        last_epoch=  int(last_ckpt.split('_')[-3])
        assert last_epoch > 10, f'Error: too less training time to conduct automatic inference!'
        return [last_ckpt]
    
    else:
        start_epoch, end_epoch = inference_cfg.test_epochs.split('-')
        skip_epoch = int(inference_cfg.skip_epoch) 
        whole_ckpts = []
        for cur_epoch in range(int(start_epoch), int(end_epoch)+1):
            if cur_epoch % skip_epoch == 0:
                ckpts = glob.glob("%s/*%06d*.pth" %(ckpt3_root, int(cur_epoch)))
                assert len(ckpts) == 1, 'Error: (ckpt, epoch) combination is not exists or contain multiple candidates!'
                whole_ckpts.append(ckpts[0])
        return whole_ckpts


# 因为我们目前只处理 merbench，这些是 video 的，需要和原始训练数据中的 video 数据对应的 face_or_frame 一致
def get_face_or_frame(datasets_cfg, outside_face_or_frame):
    if outside_face_or_frame is not None:
        return outside_face_or_frame
    
    face_or_frame_candidates = []
    if 'mercaptionplus' in datasets_cfg:
        face_or_frame_candidates.append(datasets_cfg['mercaptionplus'].face_or_frame)
    if 'ovmerd' in datasets_cfg:
        face_or_frame_candidates.append(datasets_cfg['ovmerd'].face_or_frame)
    assert len(set(face_or_frame_candidates)) == 1, f'must has the unified face_or_frame type'
    face_or_frame = list(set(face_or_frame_candidates))[0]
    return face_or_frame


def get_name2cls(dataset, dataset_cfg=None, inference_cfg=None, model_cfg=None):
    vis_processor = BaseProcessor()
    img_processor = BaseProcessor()
    vis_processor_cfg = inference_cfg.get("vis_processor") # read vis processor
    img_processor_cfg = inference_cfg.get("img_processor") # read img processor
    if vis_processor_cfg is not None:
        vis_processor = registry.get_processor_class(vis_processor_cfg.train.name).from_config(vis_processor_cfg.train)
    if img_processor_cfg is not None:
        img_processor = registry.get_processor_class(img_processor_cfg.train.name).from_config(img_processor_cfg.train)
    kwargs = {'vis_processor': vis_processor, 'img_processor': img_processor, "dataset_cfg": dataset_cfg, "model_cfg": model_cfg}
    if dataset == 'MER2023':          dataset_cls = MER2023_Dataset(**kwargs)
    if dataset == 'MER2024':          dataset_cls = MER2024_Dataset(**kwargs)
    if dataset == 'MELD':             dataset_cls = MELD_Dataset(**kwargs)
    if dataset == 'IEMOCAPFour':      dataset_cls = IEMOCAPFour_Dataset(**kwargs)
    if dataset == 'CMUMOSI':          dataset_cls = CMUMOSI_Dataset(**kwargs)
    if dataset == 'CMUMOSEI':         dataset_cls = CMUMOSEI_Dataset(**kwargs)
    if dataset == 'SIMS':             dataset_cls = SIMS_Dataset(**kwargs)
    if dataset == 'SIMSv2':           dataset_cls = SIMSv2_Dataset(**kwargs)
    if dataset == 'MER2025OV':        dataset_cls = MER2025OV_Dataset(**kwargs)
    if dataset == 'OVMERDPlus':       dataset_cls = OVMERDPlus_Dataset(**kwargs)
    dataset_cls.n_frms = model_cfg.vis_processor.train.n_frms
    print ('dataset cls not provided!')
    return dataset_cls


# 优先级：zeroshot > dataset specific
def set_config_messages(zeroshot, outside_user_message):
    # 设置全局的inference user message
    if outside_user_message is not None:
        config.USER_MESSAGES = outside_user_message
    elif zeroshot: # predict ov labels
        config.USER_MESSAGES = "Please recognize all possible emotional states of the character."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AffectGPT Inference Process")
    parser.add_argument("--cfg-path", default='/data/testmllm/project/AffectGPT/AffectGPT/train_configs/emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz.yaml', help="path to configuration file.")
    parser.add_argument("--options",  nargs="+", help="override some settings in the used config, format: --option xx=xx yy=yy zz=zz")
    parser.add_argument("--dataset", default='merbench', help="evaluate dataset")
    # parser.add_argument("--dataset", default='inferenceData', help="evaluate dataset")
    parser.add_argument("--batch_size", default=128, type=int, help="batch size for inference")
    parser.add_argument('--zeroshot', action='store_true', default=False, help='whether testing on zeroshot performance?')
    # parser.add_argument('--outside_user_message',  default="Please infer the person's emotional state and provide your reasoning process.", help="we use the outside user message, rather than dataset dependent.")
    parser.add_argument('--outside_user_message',  default=None, help="we use the outside user message, rather than dataset dependent.")
    parser.add_argument('--outside_face_or_frame', default=None, help="we use the outside face_or_frame, rather than dataset dependent.")
    args = parser.parse_args()
    cfg = Config(args)
    model_cfg = cfg.model_cfg
    datasets_cfg = cfg.datasets_cfg
    inference_cfg = cfg.inference_cfg
    device = 'cuda:{}'.format(inference_cfg.gpu)
    # inference_datasets = ['MER2023', 'MER2024', 'MELD', 'IEMOCAPFour', 'CMUMOSI', 'CMUMOSEI', 'SIMS', 'SIMSv2', 'OVMERDPlus']
    inference_datasets = ['CMUMOSEI', 'MER2024', 'IEMOCAPFour', 'SIMSv2']
    set_config_messages(args.zeroshot, args.outside_user_message)

    print ('======== Step1: cfg pre-analysis ========')
    # 支持 ckpt_root / ckpt_name 两种类型输入 => (ckpt3_root)
    # 默认情况是依据 os.path.basename(args.cfg_path) 找到 => (ckpt3_root)
    if inference_cfg.ckpt_root not in ['', 'xxx']:
        ckpt3_root = inference_cfg.ckpt_root
    elif inference_cfg.ckpt_name not in ['', 'xxx']:
        cfg_name = os.path.basename(args.cfg_path)[:-len('.yaml')]
        ckpt3_root = os.path.join('output', cfg_name, inference_cfg.ckpt_name)
        assert inference_cfg.ckpt_name.startswith(cfg_name) # 这块和 train 部分是相互配合下的结果
    else:
        print ('strat searching for suitable ckpt_root')
        cfg_name = os.path.basename(args.cfg_path)[:-len('.yaml')]
        root_candidates = glob.glob(os.path.join('/data/testmllm/models/AffectGPT/', cfg_name+'*'))
        ckpt3_root = search_for_ckpt_root(root_candidates)
    print ('processed ckpt3 root:')
    print (ckpt3_root)

    # (ckpt3_root) => processed epochs
    print ('processed ckpt3 epochs:')
    whole_ckpt3s = get_ckpt3_candidates(ckpt3_root, inference_cfg)
    for item in whole_ckpt3s: print (os.path.basename(item))

    # => (face_or_frame) (这个需要与训练数据采用的 face_or_frame 相同)
    face_or_frame = get_face_or_frame(datasets_cfg, args.outside_face_or_frame)
    print (f'Read data type: {face_or_frame}')
    print ('=======================================')


    ## main process for each ckpt3 candidates
    for ii, ckpt_3 in enumerate(whole_ckpt3s):

        ##############################################################
        print (f'======== Step2: initial model; using ckpt_3: {os.path.basename(ckpt_3)} ========')
        model_cfg.ckpt_3 = ckpt_3 # ckpt_3 has the highest priority
        if ii == 0: # first-round: initialize models
            model_cls = registry.get_model_class(model_cfg.arch) # affectgpt
            model = model_cls.from_config(model_cfg)
        if ii > 0:  # second-round: update trainable params (用新的 ckpt_3 参数覆盖)
            ckpt = torch.load(model_cfg.ckpt_3, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt['model'], strict=False)
        model = model.to(device).eval() # !! reduce randomness during the inference
        chat = Chat(model, model_cfg, device=device)
        ##############################################################


        print ('======== Step3: Inferece ========')
        if args.dataset == 'inferenceData':
            process_datasets = inference_datasets
        else:
            names = args.dataset.split(',')
            process_datasets = names
        print ('process datasets: ', process_datasets)

        ## for each dataset
        for dataset in process_datasets:
            print (f'current dataset: {dataset}')
            ## 定义结果存储位置，如果存在相应路径直接跳过
            save_root = os.path.join(inference_cfg.base_root + f'-{dataset.lower()}', # output/results-{dataset}/ckpt3_name
                                    os.path.basename(ckpt3_root)) 
            if not os.path.exists(save_root): os.makedirs(save_root)
            epoch = os.path.basename(cfg.model_cfg.ckpt_3)[:-4]
            save_path = '%s/%s.npz' %(save_root, epoch) # output/result-{dataset}/ckpt3_name/epochname
            if os.path.exists(save_path): continue
            
            ## dataset_cls 内部在 train / inference 内部的更新
            dataset_cls = get_name2cls(dataset, 
                            dataset_cfg=OmegaConf.create({"face_or_frame": face_or_frame, "label_type": "hybird"}),
                            inference_cfg=inference_cfg,
                            model_cfg=model_cfg)
            dataset_loader = DataLoader(dataset_cls, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=dataset_cls.collater)  # batch_size 可根据显存调整
            len_dataloader = len(dataset_loader)
            ## 主要处理函数 【费时的主要在这个部分】
            name2reason = {}
            for ii, batch in enumerate(tqdm(dataset_loader, total=len_dataloader)):
                responses = chat.answer_batch(samples=batch, num_beams=1, temperature=1, do_sample=True, top_p=0.9, 
                                            max_new_tokens=1024, max_length=2000)
                for name,response in zip(batch['sample_name'], responses):
                    name2reason[name] = response
                    print (f'{name}: {response}')
                
            print ('save results')
            np.savez_compressed(save_path, name2reason=name2reason)
