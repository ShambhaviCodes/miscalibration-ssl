# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import matplotlib.pyplot as plt
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from semilearn.core.utils import get_net_builder, get_dataset
from metrics import expected_calibration_error, ECELoss, ClasswiseECELoss, BrierScore, AdaptiveECELoss
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--load_path', type=str, required=True)

    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='vit_small_patch2_32')
    parser.add_argument('--net_from_name', type=bool, default=False)

    '''
    Data Configurations
    '''
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--crop_ratio', type=int, default=0.875)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max_length_seconds', type=float, default=4.0)
    parser.add_argument('--sample_rate', type=int, default=16000)

    args = parser.parse_args()
    
    checkpoint_path = os.path.join(args.load_path)
    checkpoint = torch.load(checkpoint_path)
    load_model = checkpoint['ema_model']
    load_state_dict = {}
    for key, item in load_model.items():
        if key.startswith('module'):
            new_key = '.'.join(key.split('.')[1:])
            load_state_dict[new_key] = item
        else:
            load_state_dict[key] = item
    save_dir = '/'.join(checkpoint_path.split('/')[:-1])
    args.save_dir = save_dir
    args.save_name = ''
    
    net = get_net_builder(args.net, args.net_from_name)(num_classes=args.num_classes)
    keys = net.load_state_dict(load_state_dict)
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    
    # specify these arguments manually 
    args.num_labels = 200
    args.ulb_num_labels = 49600
    args.lb_imb_ratio = 1
    args.ulb_imb_ratio = 1
    args.seed = 0
    args.epoch = 1
    args.num_train_iter = 1024
    dataset_dict = get_dataset(args, 'fixmatch', args.dataset, args.num_labels, args.num_classes, args.data_dir, False)
    eval_dset = dataset_dict['eval']

    eval_loader = DataLoader(eval_dset, batch_size=args.batch_size, drop_last=False, shuffle=False, num_workers=4)
    print('LENGTH OF TEST SET : ', len(eval_dset))
    acc = 0.0
    ece = 0.0
    aece = 0.0
    cece = 0.0
    brier = 0.0
    test_feats = []
    test_preds = []
    test_probs = []
    test_labels = []

    metrics_dict = {}
    checkpoint_file_name = os.path.basename(args.load_path)
    checkpoint_directory = os.path.dirname(args.load_path)

    with torch.no_grad():
        ece_criterion = ECELoss(15).cuda()
        classwise_ece_loss = ClasswiseECELoss(n_bins=15).cuda()
        adaptive_ece_loss = AdaptiveECELoss(n_bins=15).cuda()
        brier_loss = BrierScore()
        accuracy_values = []  # List to store accuracy values (0 or 1)
        confidence_values = []  # List to store confidence (probability) values

        for data in eval_loader:
            image = data['x_lb']
            target = data['y_lb'].cpu()  # Move target to CPU

            image = image.type(torch.FloatTensor).cuda()
            feat = net(image, only_feat=True)
            logit = net(feat, only_fc=True)
            prob = logit.softmax(dim=-1)
            pred = prob.argmax(1)

            accuracy = pred.cpu().eq(target).numpy()
            acc += accuracy.sum() 
            confidence = prob.cpu().numpy()
            accuracy_values.extend(accuracy)
            confidence_values.extend(confidence)

            test_feats.append(feat.cpu().numpy())
            test_preds.append(pred.cpu().numpy())
            test_probs.append(logit.cpu())
            test_labels.append(target)

    test_feats = np.concatenate(test_feats)
    test_preds = np.concatenate(test_preds)
    test_probs = np.concatenate(test_probs)
    test_labels = np.concatenate(test_labels)

    ece = ece_criterion(torch.from_numpy(test_probs), torch.from_numpy(test_labels))
    aece = adaptive_ece_loss(torch.from_numpy(test_probs), torch.from_numpy(test_labels))
    cece = classwise_ece_loss(torch.from_numpy(test_probs), torch.from_numpy(test_labels))
    brier = brier_loss(torch.from_numpy(test_probs), torch.from_numpy(test_labels))

    metrics_dict['Test ECE'] = ece 
    metrics_dict['Test CECE'] = cece 
    metrics_dict['Test AECE'] = aece
    metrics_dict['Test Brier'] = brier 
    metrics_dict['Test Accuracy'] = acc / len(eval_dset)
    metrics_dict['Test Error'] = 1 - acc / len(eval_dset)

    print("Metrics Dictionary:", metrics_dict)

    # Convert tensor types to lists or numbers
    converted_dict = {}
    for key, value in metrics_dict.items():
        if isinstance(value, torch.Tensor):
            if value.dim() == 0:
                converted_dict[key] = value.item()  # Convert single value tensor to number
            else:
                converted_dict[key] = value.tolist()  # Convert tensor to a list

    checkpoint_file_name = os.path.basename(args.load_path)
    checkpoint_directory = os.path.dirname(args.load_path)

    metrics_file_name = os.path.splitext(checkpoint_file_name)[0] + "_metrics.json"
    metrics_file_path = os.path.join(checkpoint_directory, metrics_file_name)

    with open(metrics_file_path, 'w') as metrics_file:
        json.dump(converted_dict, metrics_file)

    print(f"Metrics saved to: {metrics_file_path}")