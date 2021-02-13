import argparse
import logging
import os
import time

import numpy as np
import torch
# torch.multiprocessing.set_start_method('spawn', force=True)

import torch.optim
from torch.utils.data import DataLoader, dataset
from vgg_nets import EmbeddingModel
from loss import AAV_loss, AV_loss
from dataset import AudioVisualDataset


SEED = 2
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

# if torch.cuda.is_available():
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')


def get_data_loader(args, logger):
    train_folder = './data/LRS2/pretrain/'
    dev_folder = './data/LRS2/val/'
    vox_tr = AudioVisualDataset(video_scp=train_folder + 'video.scp', sec=2.4, rand_clip=True)
    vox_dev = AudioVisualDataset(video_scp=dev_folder + 'video.scp', sec=2.4, rand_clip=False)
    data_loader_tr = DataLoader(vox_tr, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)
    data_loader_dev = DataLoader(vox_dev, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)
    return {'dev': data_loader_dev, 'tr': data_loader_tr}


def reload_model(model, logger, device, args):
    if not bool(args.model_path):
        logger.info('train from scratch')
    else:
        logger.info('loading model from {}'.format(args.model_path))
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.model_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           k in model_dict and v.size() == model_dict[k].size()
                           }
        # and (not 'phasenet' in k)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.info('*** Model has been successfully loaded! {} parameters loaded***'.format(len(pretrained_dict)))

    if torch.cuda.device_count() >= 1:
        logger.info('Use {} GPUs'.format(torch.torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    model.to(device)

    return model



def showLR(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr



def save_model(model, args, epoch):
    save_path = args.exp_dir + '/models'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    try:
        state_dict = model.module.state_dict()
    except AttributeError:
        state_dict = model.state_dict()
    torch.save(state_dict, save_path + '/' + str(epoch + 1) + '.pt')



def train(model, dataloaders, epoch, optimizer, args, logger):
    model.train()

    dataloader = dataloaders['tr']
    logger.info("-" * 10)
    logger.info('Epoch {}/{}'.format(epoch, args.epochs - 1))
    logger.info("Current LR: {}".format(showLR(optimizer)))

    runningloss = 0.0
    runningall = 0

    for batch_idx, (audio, video) in enumerate(dataloader):
        audio = audio.cuda()
        video = video.cuda()
        a_emb, v_emb, loss = model.forward(audio, video)
        optimizer.zero_grad()
        loss = loss.mean()
        loss.backward()
        optimizer.step()

        runningall += 1
        runningloss += float(loss) 

        if batch_idx == 0:
            since = time.time()
        elif batch_idx % args.interval == 0 or (batch_idx == len(dataloader) - 1):
            logger.info(
                'Process: [{:5.0f}/{:5.0f} ({:.0f}%)]\tLoss: {:.4f}\tCost time:{:5.0f}s\tEstimated time:{:5.0f}s\r'.format(
                    runningall,
                    len(dataloader.dataset),
                    100. * batch_idx / (len(dataloader) - 1),
                    runningloss / runningall,
                    time.time() - since,
                    (time.time() - since) * (len(dataloader) - 1) / batch_idx - (time.time() - since)
                ))

    logger.info('Train Epoch:\t{:2}\tLoss: {:.4f}'.format(
        epoch,
        runningloss / len(dataloader.dataset) 
    ))

    save_model(model, args, epoch)
    return model


def test(model, dataloaders, epoch, args, logger):
    model.eval()

    dataloader = dataloaders['dev']
    logger.info("-" * 10)
    logger.info('Dev for Epoch {}/{}'.format(epoch, args.epochs - 1))

    runningloss = 0.0
    runningall = 0

    for batch_idx, (audio, video) in enumerate(dataloader):
        audio = audio.cuda()
        video = video.cuda()

        with  torch.no_grad():
            a_emb, v_emb, loss = model.forward(audio, video)
            loss = loss.mean()
        runningall += 1
        runningloss += float(loss) 

        if batch_idx == 0:
            since = time.time()
        elif batch_idx % args.interval == 0 or (batch_idx == len(dataloader) - 1):
            logger.info(
                'Process: [{:5.0f}/{:5.0f} ({:.0f}%)]\tLoss: {:.4f}\tCost time:{:5.0f}s\tEstimated time:{:5.0f}s\r'.format(
                    runningall,
                    len(dataloader.dataset),
                    100. * batch_idx / (len(dataloader) - 1),
                    runningloss / runningall,
                    time.time() - since,
                    (time.time() - since) * (len(dataloader) - 1) / batch_idx - (time.time() - since)
                ))

    logger.info('Valid Epoch:\t{:2}\tLoss: {:.4f}'.format(
        epoch,
        runningloss / len(dataloader.dataset)
    ))

    save_model(model, args, epoch)
    return model

def get_logger(args):
    if not os.path.exists(args.exp_dir):
        os.mkdir(args.exp_dir)
    log_path = args.exp_dir + '/log'
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    log_path += '/log.txt'
    logger = logging.getLogger('mylog')
    logger.setLevel("INFO")

    fh = logging.FileHandler(log_path, mode='a')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    dfh = logging.FileHandler(log_path + '.debug', mode='a')
    dfh.setLevel(logging.DEBUG)
    logger.addHandler(dfh)

    logger.addHandler(logging.StreamHandler())

    return logger


def schedule(args):
    logger = get_logger(args)
    logger.info("RUN with parameters: " + str(vars(args)))

    model= EmbeddingModel()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = reload_model(model, logger, device, args)

    dataloaders = get_data_loader(args, logger)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=2, gamma=0.9)

    test(model, dataloaders, -1, args, logger)
    for epoch in range(args.epochs):
        model = train(model=model, dataloaders=dataloaders, epoch=epoch, optimizer=optimizer, args=args,
                        logger=logger)
        test(model, dataloaders, epoch, args, logger, )
        scheduler.step()



def main():
    parser = argparse.ArgumentParser(description="SPEECH SPLIT")
    parser.add_argument('--model_path', default='', help='path to pretrained model')
    parser.add_argument('--exp-dir', default='./exp', help='path to exp dir')
    parser.add_argument('--batch-size', default=12, type=int, help='mini-batch size (default: 36)')
    parser.add_argument('--workers', default=14, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=30, type=int, help='number of total epochs')
    parser.add_argument('--lr', default=1e-3, type=float, help='False for test model')
    parser.add_argument('--interval', default=10, type=int, help='display interval')
    args = parser.parse_args()

    schedule(args)


if __name__ == '__main__':
    main()
