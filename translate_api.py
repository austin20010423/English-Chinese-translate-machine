import training


def response():
    input_sentence = str(input('enter words: '))
    output_sentence = training.translate_api(input_sentence)
    print('output translate: ', output_sentence)


if __name__ == '__main__':

    response()
