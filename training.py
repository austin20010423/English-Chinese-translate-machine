import numpy as np
import matplotlib.ticker as ticker
from model import *
from read_data import *
import random
import time
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
device = torch.device('cpu')
MAX_LENGTH = 10
'''
def timeSince(start):
    seconds = time.time()
    sub_time = time.ctime(seconds) - start

    return sub_time
'''


def training(encoder, decoder, n_iter, print_every):
    print_loss_total = 0
    plot_loss_total = 0
    iter_total = []
    plot_losses = []
    _, _, pairs = readLangs('eng', 'cha', reverse=False)
    training_pairs = [tensorFromPair(random.choice(pairs))
                      for i in range(n_iter)]

    for iter in range(1, n_iter+1):
        # random.shuffle(pairs)
        training_pair = training_pairs[n_iter-1]
        # print(training_pair)
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        loss = train(input_tensor, target_tensor)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            iter_total.append(iter)
            print_loss_avg = print_loss_total/print_every
            plot_losses.append(print_loss_avg)
            print_loss_total = 0
            seconds = time.time()
            start = time.ctime(seconds)
            print('iter:\t%d\tpercentage:\t%d%%\tloss:\t%.8f' %
                  (iter, iter/n_iter*100, print_loss_avg))

    torch.save(encoder, 'dataset/model/encoder.pt')
    torch.save(decoder, 'dataset/model/decoder.pt')

    return plot_losses, iter_total


def showPlot(iter, points):
    plt.plot(iter, points)
    plt.title("Loss Chart")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()

    # plt.legend(loc=0.7)


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):

    in_lang, out_lang, _ = readLangs('eng', 'cha', reverse=False)
    eng = sentence[0]
    inputs = tensorFromSentence(in_lang, eng)
    input_length = inputs.size()[0]

    encoder_hidden = encoder.initHidden()
    encoder_outputs = torch.zeros(
        max_length, encoder.hidden_size, device=device)

    decoder_input = torch.tensor([SOS_token], device=device)
    decoder_hidden = encoder_hidden
    decoder_attentions = torch.zeros(max_length, max_length, device=device)
    decoder_words = []

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0].item()

        if ni == EOS_token:
            decoder_words.append('<EOS>')
            break
        else:
            decoder_words.append(out_lang.index2word[topi.item()])
        decoder_input = topi.squeeze().detach()

    return decoder_words


def evaluateRandomly(encoder, decoder, n):
    _, _, pairs = readLangs('eng', 'cha', reverse=False)
    for i in range(0, n):
        pair = random.choice(pairs)
        print('source: ', pair[0])
        print('target: ', pair[1])
        output_word = evaluate(encoder, decoder, pair)
        output_word = ' '.join(output_word)
        print('predict: ', output_word)
        print()


if __name__ == '__main__':
    _, _, _, encoder, decoder = optimizer()
    encoder = torch.load('dataset/model/encoder.pt')
    decoder = torch.load('dataset/model/decoder.pt')
    plot_loss, iter = training(encoder, decoder, n_iter=100, print_every=10)
    encoder = torch.load('dataset/model/encoder.pt')
    decoder = torch.load('dataset/model/decoder.pt')
    evaluateRandomly(encoder, decoder, 1)
    showPlot(iter, plot_loss)
