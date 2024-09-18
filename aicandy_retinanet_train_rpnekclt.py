"""

@author:  AIcandy 
@website: aicandy.vn

"""

import argparse
import collections
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from aicandy_utils_src_obilenxc import model
from aicandy_utils_src_obilenxc.dataloader import CocoDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer
from torch.utils.data import DataLoader
from aicandy_utils_src_obilenxc import evaluate

# python aicandy_retinanet_train_rpnekclt.py --train_dir /aicandy/datasets/aicandy_motorcycle_humukdiy --num_epochs 100 --batch_size 4 --model_path 'aicandy_output_ntroyvui/aicandy_model_retina_lgkrymnl.pth' 


def train(train_dir, epochs, batch_size, model_path):
    if train_dir is None:
        raise ValueError('Must provide --train_dir when training')

    dataset_train = CocoDataset(train_dir, set_name='train2017',
                                transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    dataset_val = CocoDataset(train_dir, set_name='val2017',
                              transform=transforms.Compose([Normalizer(), Resizer()]))

    sampler_train = AspectRatioBasedSampler(dataset_train, batch_size=batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler_train)

    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=batch_size, drop_last=False)
    dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    retinanet = model.create_base_resnet(num_classes=dataset_train.num_classes())
    print('num_classes: ', dataset_train.num_classes())

    # Use GPU if available, otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    retinanet = retinanet.to(device)

    retinanet.training = True
    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    running_loss_best = float('inf')

    for epoch_num in range(epochs):

        retinanet.train()
        retinanet.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()

                # Move data to the selected device
                images = data['img'].to(device).float()
                annotations = data['annot'].to(device)

                classification_loss, regression_loss = retinanet([images, annotations])
                
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
                optimizer.step()
                loss_hist.append(float(loss))
                epoch_loss.append(float(loss))
                running_loss = np.mean(loss_hist)

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        evaluate.evaluate_coco(dataset_val, retinanet)

        scheduler.step(np.mean(epoch_loss))
        print('Epoch: {}/{} Running loss: {:1.5f}'.format(epoch_num + 1, epochs, running_loss))
        
        if running_loss < running_loss_best:
            running_loss_best = running_loss
            torch.save(retinanet.state_dict(), model_path)
            print('Saved model with loss: ', running_loss_best)

    retinanet.eval()

if __name__ == '__main__':    
    import sys
    import argparse
    parser = argparse.ArgumentParser(description='AIcandy.vn')
    parser.add_argument('--train_dir', help='Path to dataset directory')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--model_path', help='Path to save the trained model', default='aicandy_output_ntroyvui/aicandy_model_retina_lgkrymnl.pth')

    args = parser.parse_args()

    train(args.train_dir, args.num_epochs, args.batch_size, args.model_path)
