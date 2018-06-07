import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint_sequential

class Watch(nn.Module):
    '''
    layer size 3
    cell size 256
    '''
    def __init__(self, num_layers, input_size, hidden_size):
        super(Watch, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.encoder = Encoder()

    def forward(self, x):
        '''
        Parameters
        ----------
        x : 4-D torch Tensor
            (batch_size, time_sequence, 120, 120) 
        '''
        size = list(x.size())
        
        # assert len(size) == 4, 'video input size is wrong'
        # assert size[2:] == [120, 120], 'image size should 120 * 120'

        outputs = []
        for i in range(size[1] - 4):
            outputs.append(self.encoder(x[:, i:i+5, :, :]).unsqueeze(1))
        outputs.reverse()
        x = torch.cat(outputs, dim=1)
        self.flatten_parameters()
        outputs, states = self.lstm(x)

        return (outputs, states[0])

class Listen(nn.Module):
    '''
    layer size 3
    cell size 256
    '''
    def __init__(self, num_layers, hidden_size):
        super(Listen, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(13, hidden_size, num_layers, batch_first=True)
        
    def forward(self, x):
        '''
        Parameters
        ----------
        x : 3-D torch Tensor
            (batch_size, sequence, features) 
        '''
        outputs, states = self.lstm(x)

        return (outputs, states[0])

class Spell(nn.Module):
    def __init__(self, num_layers=3, output_size=40, hidden_size=512):
        super(Spell, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.embedded = nn.Embedding(self.output_size, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size*2, self.hidden_size, self.num_layers, batch_first=True)
        self.attentionVideo = Attention(hidden_size, hidden_size)
        self.mlp = nn.Linear(hidden_size*2, output_size)
    
    def forward(self, input, hidden_state, cell_state, watch_outputs, context):
        '''
        1. embedding
            input : (batch_size, 1, hidden_size)

        2. concatenate intput and context
            concatenated : (batch_size, 1, hidden_size + encoder_hidden_size * 2)
        
        3. lstm
            output : (batch_size, 1, hidden_size)
            cell_state, hidden_state : (num_layers, batch_size, hidden_size)
            
        4. compute attention
            video_context, audio_context : (batch_size, 1, encoder_hidden_size(= hidden_size/2))

        5. concatenate two seperate context
            context : (batch_size, 1, encoder_hidden_size * 2)

        6. through mlp layer [output, context]
            output : (batch_size, 1, output_size)

        Parameters
        ----------
        input : 2-D torch Tensor
            size (batch_size, 1)

        hidden_state : 3-D torch Tensor
            size (num_layers, batch_size, hidden_size)

        cell_state : 3-D torch Tensor
            size (num_layers, batch_size, hidden_size)

        watch_outputs : 3-D torch Tensor
            size (batch_size, watch time sequence, encoder_hidden_size)

        listen_outputs : 3-D torch Tensor
            size (batch_size, listen time sequence, encoder_hidden_size)

        context : 3-D torch Tensor
            size (batch_size, 1, encoder_hidden_size*2)

        Returns
        -------
        output : 3-D torch Tensor
            size (batch size, 1, output_size)

        hidden_state : 3-D torch Tensor
            size (num_layers, batch_size, hidden_size)
        '''
        input = self.embedded(input)
        concatenated = torch.cat([input, context], dim=2)
        self.flatten_parameters()
        output, (hidden_state, cell_state) = self.lstm(concatenated, (hidden_state, cell_state))
        context = self.attentionVideo(hidden_state[-1], watch_outputs)
        
        output = self.mlp(torch.cat([output, context], dim=2))
        
        return output, hidden_state, cell_state, context

class Attention(nn.Module):
    '''
    from https://arxiv.org/pdf/1409.0473.pdf
    Bahdanau attention
    https://machinelearningmastery.com/how-does-attention-work-in-encoder-decoder-recurrent-neural-networks/
    simple image
    https://images2015.cnblogs.com/blog/670089/201610/670089-20161012111504671-910168246.png
    '''
    def __init__(self, hidden_size, annotation_size):
        super(Attention, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(hidden_size+annotation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, prev_hidden_state, annotations):
        '''

        1. expand prev_hidden_state dimension and transpose
            prev_hidden_state : (batch_size, sequence_length, feature dimension(512)) 
        
        2. concatenate
            concatenated : size (batch_size, sequence_length, encoder_hidden_size + hidden_size)
        
        3. dense and squeeze
            energy : size (batch_size, sequence_length)

        4. softmax to compute weight alpha
            alpha : (batch_size, 1, sequence_length)
        
        5. weighting annotations
            context : (batch_size, 1, encoder_hidden_size(256))

        Parameters
        ----------
        prev_hidden_state : 3-D torch Tensor
            (batch_size, 1, hidden_size(default 512))
        
        annotations : 3-D torch Tensor
            (batch_size, sequence_length, encoder_hidden_size(256))

        Returns
        -------
        context : 3-D torch Tensor
            (batch_size, 1, encoder_hidden_size(256))
        '''
        batch_size, sequence_length, _ = annotations.size()

        prev_hidden_state = prev_hidden_state.repeat(sequence_length, 1, 1).transpose(0, 1)

        concatenated = torch.cat([prev_hidden_state, annotations], dim=2)
        attn_energies = self.dense(concatenated).squeeze(2)
        alpha = F.softmax(attn_energies).unsqueeze(1)
        context = alpha.bmm(annotations)

        return context

class Encoder(nn.Module):
    '''modified VGG-M
    '''
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 96, (7, 7), (2, 2)),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True),
            nn.Conv2d(96, 256, (5, 5), (2, 2), (1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((3, 3),(2, 2),(0, 0), ceil_mode=True)
        )
        
        self.fc = nn.Linear(4608, 512)

    def forward(self, x):
        return self.fc(checkpoint_sequential(self.encoder, len(self.encoder), x).view(x.size(0), -1))