import torch
from torch import nn, optim
from dataLoader import get_dataloaders
from model import Watch, Spell
from torch.utils.data.sampler import SubsetRandomSampler
from charSet import CharSet
import numpy as np
import sys
import argparse

torch.backends.cudnn.enabled = True

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter', type=int, help='the number of iterations')
    parser.add_argument('--vmax', type=int, help='video tensor max length')
    parser.add_argument('--tmax', type=int, help='text tensor max length')
    parser.add_argument('--path', type=str, help='data path')
    parser.add_argument('--bs', type=int, help='batch size', default=2)
    parser.add_argument('--worker', type=int, help='the number of data loader worker', default=0)
    parser.add_argument('--validation_ratio', type=float, help='validation ratio', default=0.0001)
    parser.add_argument('--learning_rate_decay_epoch', type=int, help='learning rate decay per certain epoch')
    parser.add_argument('--learning_rate_decay_ratio', type=float, help='learning rate decay ratio')
    parser.add_argument('--save_every', type=int, help='save every certain epoch', default=30)
    parser.add_argument('--learning_rate', type=float, help='learning rate')
    parser.add_argument('--language', type=str, help='language')
    parser.add_argument('--hidden_size', type=int, help='hidden dimension size', default=512)
    parser.add_argument('--layer_size', type=int, help='layer size', default=3)

    return parser.parse_args()

def train(watch_input_tensor, target_tensor,
        watch, spell, 
        watch_optimizer, spell_optimizer, 
        criterion, is_train, charSet):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    watch = watch.to(device)
    spell = watch.to(device)
    watch_input_tensor = watch_input_tensor.to(device)
    target_tensor = target_tensor.to(device)
    spell_input = torch.tensor([[charSet.get_index_of('<sos>')]]).repeat(watch_outputs.size(0), 1).to(device)
    spell_hidden = watch_state.to(device)
    cell_state = torch.zeros_like(spell_hidden).to(device)
    context = torch.zeros(watch_outputs.size(0), 1, spell_hidden.size(2)).to(device)

    print('train step')
    watch_optimizer.zero_grad()
    spell_optimizer.zero_grad()

    target_length = target_tensor.size(1)

    loss = 0

    watch_outputs, watch_state = watch(watch_input_tensor)

    if is_train:
        for di in range(target_length):
            spell_output, spell_hidden, cell_state, context = spell(
                spell_input, spell_hidden, cell_state, watch_outputs, context)
            topv, topi = spell_output.topk(1, dim=2)
            spell_input = target_tensor[:, di].long().unsqueeze(1)
            
            loss += criterion(spell_output.squeeze(1), target_tensor[:, di].long())
        loss = loss.to(device)
        loss.backward()

        watch_optimizer.step()
        spell_optimizer.step()
    else:
        for di in range(target_length):
            spell_output, spell_hidden, cell_state, context = spell(
                spell_input, spell_hidden, cell_state, watch_outputs, context)
            topv, topi = spell_output.topk(1, dim=2)
            spell_input = topi.squeeze(1).detach()
            
            if int(target_tensor[0, di]) != charSet.get_index_of('<pad>'):
                print('output : ', charSet.get_char_of(int(topi.squeeze(1)[0])), 'label : ', charSet.get_char_of(int(target_tensor[0, di])))

            # loss += criterion(spell_output.squeeze(1), target_tensor[:, di].long())

    return loss.item() / target_length

def trainIters(args):
    charSet = CharSet(args.language)

    watch = Watch(args.layer_size, args.hidden_size, args.hidden_size)
    spell = Spell(args.layer_size, charSet.get_total_num(), args.hidden_size)
    
    watch = nn.DataParallel(watch)
    spell = nn.DataParallel(spell)

    
    watch_optimizer = optim.Adam(watch.parameters(),
                    lr=args.learning_rate)
    spell_optimizer = optim.Adam(spell.parameters(),
                    lr=args.learning_rate)
    watch_scheduler = optim.lr_scheduler.StepLR(watch_optimizer, step_size=args.learning_rate_decay_epoch, gamma=args.learning_rate_decay_ratio)
    spell_scheduler = optim.lr_scheduler.StepLR(spell_optimizer, step_size=args.learning_rate_decay_epoch, gamma=args.learning_rate_decay_ratio)
    criterion = nn.CrossEntropyLoss(charSet.get_index_of('<pad>'))

    train_loader, eval_loader = get_dataloaders(args.path, args.bs, args.vmax, args.tmax, args.worker, args.validation_ratio)
    # train_loader = DataLoader(dataset=dataset,
    #                     batch_size=batch_size,
    #                     shuffle=True)
    total_batch = len(train_loader)
    total_eval_batch = len(eval_loader)

    for epoch in range(args.iter):
        avg_loss = 0.0
        avg_eval_loss = 0.0
        watch_scheduler.step()
        spell_scheduler.step()

        watch = watch.train()
        spell = spell.train()

        for i, (data, labels) in enumerate(train_loader):

            loss = train(data, labels,
                        watch, spell,
                        watch_optimizer, spell_optimizer,
                        criterion, True, charSet)
            avg_loss += loss
        
        watch = watch.eval()
        spell = spell.eval()

        for k, (data, labels) in enumerate(eval_loader):
            loss = train(data, labels, watch, spell, watch_optimizer, spell_optimizer, criterion, False, charSet)
            avg_eval_loss += loss
        
        print('epoch:', epoch, ' train_loss:', float(avg_loss/total_batch))
        print('epoch:', epoch, ' eval_loss:', float(avg_eval_loss/total_eval_batch))
        if epoch % save_every == 0 and epoch != 0:
            torch.save(watch, 'watch{}.pt'.format(epoch))
            torch.save(spell, 'spell{}.pt'.format(epoch))

if __name__ == '__main__':
    args = arg_parser()
    trainIters(args)