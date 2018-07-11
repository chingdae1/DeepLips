from dataloader import get_data_loader
import os
import torch
from torch import nn, optim
import argparse
import yaml
import nsml
import importlib

from nsml import DATASET_PATH, GPU_NUM, HAS_DATASET, IS_ON_NSML

torch.backends.cudnn.enabled = True

if HAS_DATASET and IS_ON_NSML:
    DATASET_PATH = os.path.join(DATASET_PATH, 'train', '0')
    print("Use NSML dataset: {}".format(DATASET_PATH))
else:
    DATASET_PATH = "./data_categorized/0"
    print("Use local dataset: {}".format(DATASET_PATH))



def train_iter(model, data, criterion, optimizer, label, input):
    # label 앞뒤로 몇프레임 예측할지
    # input 인풋 프레임
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = data.size(1) - input - label * 2 + 1
    result_loss = 0.0

    for i in range(size):
        loss = 0.0
        optimizer.zero_grad()
        part = data[:, i:i+(label*2 + input), :, :].to(device)
        output = model(part[:, label:input+label, :, :], False)
        loss += criterion(output, part.div(255))

        loss.to(device)
        loss.backward()
        optimizer.step()
        result_loss += loss.item()

    return result_loss / size


def eval_iter(model, data, criterion, label, input):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = data.size(1) - input - label * 2 + 1
    loss = 0.0

    for i in range(size):
        part = data[:, i:i+(label*2 + input), :, :].to(device)
        output = model(part[:, label:input+label, :, :], False)
        loss += criterion(output, part.div(255))
        del part

    return loss.item() / size


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, help='start iteration')
    parser.add_argument('--load', help='load trained model')
    parser.add_argument('--target', type=int,help='target size')
    parser.add_argument('--input', type=int, help='input size')
    parser.add_argument('--model', type=str, help='model to train')

    return parser.parse_args()


def train(configs, args):

    if args.load:
        model = torch.load(args.load)
    else:
        model = importlib.import_module("model.{}".format(args.model))
        model = model.CAE()

    bind_model(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = nn.DataParallel(model).to(device)

    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(model.parameters(),
               lr=configs['LEARNING_RATE'], weight_decay=configs['WEIGHT_DECAY'])

    train_dataset, val_dataset = get_data_loader(DATASET_PATH, configs['BS'], configs['RATIO'], configs['VMAX'])

    train_size = len(train_dataset)
    val_size = len(val_dataset)

    for epoch in range(int(args.start) if args.start else 0, configs['ITER']):
        avg_loss = 0.0
        avg_eval_loss = 0.0

        for i, data in enumerate(train_dataset):
            loss = train_iter(model, data, criterion, optimizer, args.target, args.input)
            avg_loss += loss
            print('Epoch :', epoch, ', Batch : ', i + 1, '/', train_size, ', ERROR in this minibatch: ', loss)
        
        if epoch % configs['SAVE_EVERY'] == 0 and epoch != 0:
            nsml.save(epoch)

        with torch.no_grad():
            for i, data in enumerate(val_dataset):
                loss = eval_iter(model, data, criterion, args.target, args.input)
                avg_eval_loss += loss
                print('Batch : ', i + 1, '/', val_size, ', Validation ERROR in this minibatch: ', loss)

        print('epoch:', epoch, ' train_loss:', float(avg_loss / train_size))
        # print('epoch:', epoch, ' Average CER:', float(avg_cer/total_batch))
        print('epoch:', epoch, ' Validation_loss:', float(avg_eval_loss / val_size))
        # print('epoch:', epoch, ' Average CER:', float(avg_eval_cer/total_eval_batch))
        
def bind_model(model):
    def save(filename, *args):
        checkpoint = {
            'CAE': model.state_dict()
        }

        os.makedirs(filename, exist_ok=True)
        path = os.path.join(filename, "model.pt")
        torch.save(checkpoint, path)

    nsml.bind(save=save)

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.load(f)
    train(config['CONFIG'], arg_parser())
