import torch
from torch import nn, optim
from dataLoader import to_char, videoDataset, one_hot
from model import Watch, Spell
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import sys

torch.backends.cudnn.enabled = True

def get_dataloaders(path, batch_size, videomax, txtmax, worker, ratio_of_validation=0.0001):
    num_workers = worker # num of threads to load data, default is 0. if you use thread(>1), don't confuse evenif debug messages are reported asynchronously.
    train_movie_dataset = videoDataset(path, videomax, txtmax)
    num_train = len(train_movie_dataset)
    split_point = int(ratio_of_validation*num_train)

    indices = list(range(num_train))

    np.random.shuffle(indices)
    train_idx, val_idx = indices[split_point : ], indices[ : split_point]

    train_sampler = SubsetRandomSampler(train_idx) # Random sampling at every epoch without replacement in given indices
    val_sampler = SubsetRandomSampler(val_idx)
    train_loader = torch.utils.data.DataLoader(
        train_movie_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers,
        pin_memory=True)
        # 'pin_memory=True' allows for you to use fast memory buffer with way of calling '.cuda(async=True)' function.

    val_loader = torch.utils.data.DataLoader(
        train_movie_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers,
        pin_memory=True)
    return [train_loader, val_loader]

def train(watch_input_tensor, target_tensor,
        watch, spell, 
        watch_optimizer, spell_optimizer, 
        criterion, train):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('train step')
    watch_optimizer.zero_grad()
    spell_optimizer.zero_grad()

    target_length = target_tensor.size(1)

    loss = 0

    watch_outputs, watch_state = watch(watch_input_tensor)
    #sos token
    spell_input = torch.tensor([[one_hot['<sos>']]]).repeat(watch_outputs.size(0), 1)
    spell_hidden = watch_state.to(device)
    cell_state = torch.zeros_like(spell_hidden).to(device)
    context = torch.zeros(watch_outputs.size(0), 1, spell_hidden.size(2)).to(device)

    # test = [spell_hidden]
    # Without teacher forcing: use its own predictions as the next input
    if train:
        for di in range(target_length):
            spell_output, spell_hidden, cell_state, context = spell(
                spell_input.to(device), spell_hidden, cell_state, watch_outputs, context)
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
            
            # if int(target_tensor[0, di]) != 38:
            #     print('output : ', to_char[int(topi.squeeze(1)[0])], 'label : ', to_char[int(target_tensor[0, di])])

            loss += criterion(spell_output.squeeze(1), target_tensor[:, di].long())

    return loss.item() / target_length

def trainIters(n_iters, videomax, txtmax, data_path, batch_size, worker, ratio_of_validation=0.0001, learning_rate_decay=2000, save_every=30, learning_rate=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    watch = Watch(3, 512, 512)
    spell = Spell(num_layers=3, output_size=40, hidden_size=512)
    
    watch = nn.DataParallel(watch).to(device)
    spell = nn.DataParallel(spell).to(device)

    watch_optimizer = optim.Adam(watch.parameters(),
                    lr=learning_rate)
    spell_optimizer = optim.Adam(spell.parameters(),
                    lr=learning_rate)
    watch_scheduler = optim.lr_scheduler.StepLR(watch_optimizer, step_size=learning_rate_decay, gamma=0.1)
    spell_scheduler = optim.lr_scheduler.StepLR(spell_optimizer, step_size=learning_rate_decay, gamma=0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=38)

    train_loader, eval_loader = get_dataloaders(data_path, batch_size, videomax, txtmax, worker, ratio_of_validation=ratio_of_validation)
    # train_loader = DataLoader(dataset=dataset,
    #                     batch_size=batch_size,
    #                     shuffle=True)
    total_batch = len(train_loader)
    total_eval_batch = len(eval_loader)

    for epoch in range(n_iters):
        avg_loss = 0.0
        avg_eval_loss = 0.0
        watch_scheduler.step()
        spell_scheduler.step()

        watch = watch.train()
        spell = spell.train()

        for i, (data, labels) in enumerate(train_loader):

            loss = train(data.to(device), labels.to(device),
                        watch, spell,
                        watch_optimizer, spell_optimizer,
                        criterion, True)
            avg_loss += loss
            
            del data, labels, loss
        
        watch = watch.eval()
        spell = spell.eval()

        for k, (data, labels) in enumerate(eval_loader):
            loss = train(data.to(device), labels.to(device), watch, spell, watch_optimizer, spell_optimizer, criterion, False)
            avg_eval_loss += loss

            del data, labels, loss
        
        print('epoch:', epoch, ' train_loss:', float(avg_loss/total_batch))
        print('epoch:', epoch, ' eval_loss:', float(avg_eval_loss/total_eval_batch))
        if epoch % save_every == 0 and epoch != 0:
            torch.save(watch, 'watch{}.pt'.format(epoch))
            torch.save(spell, 'spell{}.pt'.format(epoch))

if __name__ == '__main__':
    num_iterates = int(sys.argv[1])
    videomax = int(sys.argv[2])
    txtmax = int(sys.argv[3])
    data_path = sys.argv[4]
    batch_size = int(sys.argv[5])
    worker = int(sys.argv[6])
    ratio_of_validation = float(sys.argv[7])
    learning_rate_decay = int(sys.argv[8])
    save_every = int(sys.argv[9])
    learning_rate = float(sys.argv[10])
    trainIters(num_iterates, videomax, txtmax, data_path, batch_size, worker, ratio_of_validation, learning_rate_decay, save_every, learning_rate)