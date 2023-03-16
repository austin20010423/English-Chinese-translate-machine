import torch
from torch import nn
from read_data import *
import random
from torch import optim

MAX_LENGTH = 10
device = torch.device('cpu')


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size, device=device)
        self.gru = nn.GRU(hidden_size, hidden_size, device=device)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(
            self.output_size, self.hidden_size, device=device)
        self.attn = nn.Linear(self.hidden_size*2,
                              self.max_length, device=device)
        self.attn_combine = nn.Linear(
            self.hidden_size*2, self.hidden_size, device=device)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, device=device)
        self.out = nn.Linear(self.hidden_size, self.output_size, device=device)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = torch.nn.functional.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(
            0), encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = torch.nn.functional.relu(output)
        output, hidden = self.gru(output, hidden)
        output = torch.nn.functional.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def optimizer():
    # optimize and compile
    input_lang, output_lang, pairs = readLangs('eng', 'cha', reverse=False)

    # SGD optimizer
    encoder = EncoderRNN(input_lang.n_words, 256)
    decoder = AttnDecoderRNN(256, output_lang.n_words)

    learning_rate = 0.001
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    return encoder_optimizer, decoder_optimizer, criterion, encoder, decoder


teacher_forcing_ratio = 0.5
SOS_token = 0
EOS_token = 1


def train(input_tensor, target_tensor):
    encoder_optimizer, decoder_optimizer, criterion, encoder, decoder = optimizer()
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size()[0]
    target_length = target_tensor.size()[0]

    loss = 0

    encoder_outputs = torch.zeros(
        MAX_LENGTH, encoder.hidden_size, device=device)

    # encoder in
    for ei in range(input_length):
        _, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

    # decoder calculate
    decoder_input = torch.tensor([SOS_token], device=device)
    decoder_hidden = encoder_hidden

    # set teacher forcing
    if random.random() < teacher_forcing_ratio:
        use_teacher_forcing = True
    else:
        use_teacher_forcing = False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)

            decoder_input = topi.squeeze().detach()
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    # back propagation
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()/target_length


if __name__ == '__main__':
    in_lang, out_lang, pair = readLangs('eng', 'cha', reverse=False)

    for i in range(0, 2):
        pairs = tensorFromPair(pair)
        input = pairs[0]
        output = pairs[1]

        loss = train(input, output)
        print(loss)
