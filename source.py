# -*- coding: utf-8 -*-
import re
import collections
import math
import os
import numpy as np
import operator as op

np.set_printoptions(precision=5, suppress=True)


def my_cmp(a, b, eps, op):
    if math.isclose(a, b, abs_tol=eps): return True
    return op(a, b)


def tokenize(s):
    tokens = re.split(r"[^0-9A-Za-zа-яА-Я\-'_]+", s)
    return tokens


def check_date(check_folder, match_folder):
    check_files = os.listdir(check_folder)
    match_files = os.listdir(match_folder)
    for it in check_files:
        sys = open(str(check_folder + it))
        text = "".join(sys.readlines())
        text = " ".join(text.split("\n"))
        words = tokenize(text)
        for jt in match_files:
            if str(check_folder + it) != str(match_folder + jt):
                sys1 = open(str(match_folder + jt))
                text1 = "".join(sys1.readlines())
                text1 = " ".join(text1.split("\n"))
                words1 = tokenize(text1)
                if set(words) == set(words1):
                    print('Файл есть в тестовой выборке: ' + str(it) + ' ' + str(jt))


def get_yules(s, ResArray):
    """
    Returns a tuple with Yule's K and Yule's I.
    (cf. Oakes, M.P. 1998. Statistics for Corpus Linguistics.
    International Journal of Applied Linguistics, Vol 10 Issue 2)
    In production this needs exception handling.
    """
    tokens = tokenize(s)
    token_counter = collections.Counter(tok.upper() for tok in tokens)
    m1 = sum(token_counter.values())
    m2 = sum([freq ** 2 for freq in token_counter.values()])
    i = (m1 * m1) / (m2 - m1)
    k = 1 / i * 10000
    # filename.write("Yule K метрика: " + str(k) + "\n")
    ResArray[6] = k
    return k


def Hapax(s, ResArray):
    words = tokenize(s)
    freqs = {key: 0 for key in words}
    cnthapax = 0.0
    cnttokens = 0.0
    for word in words:
        freqs[word] += 1
        cnttokens += 1

    for word in freqs:
        if freqs[word] == 1: cnthapax += 1

    # filename.write("Hapax: " + str(cnthapax / cnttokens) + "\n")
    ResArray[5] = cnthapax / cnttokens


def safeSplit(regex, text):
    res = []
    sear = regex.search(text)
    while sear:
        res.append(text[:sear.end()])
        text = text[sear.end():]
        sear = regex.search(text)
    res.append(text)
    return res


def AvgSentLength(sentences):
    cnt = len(sentences) - 1
    # print(cnt, "\n")
    res = 0.0
    avg = 0.0

    for sent in sentences:
        # print(sent, "\n------------")
        words = sent.split()
        avg += len(words)
    return (avg / cnt)


def LexicalCharacterFeature(text, ResArray):
    i = 0
    Specsymbols = {'~', '@', '#', '№', '%', '^', '&', '*', '{', '}', '-', '+', '=', '/', '|', '[', ']', '<', '>', '(',
                   ')', '\\', '$'}
    Punctuations = {',', '.', '!', '?', ':', ';', "'", '"'}
    Vowels = {'у', 'е', 'а', 'о', 'э', 'я', 'и', 'ю', 'e', 'y', 'u', 'o', 'i', 'a'}
    countDigits = 0.0
    countLetters = 0.0
    countUppercaseLetters = 0.0
    countSpaces = 0.0
    countTabs = 0.0
    countSpecsymbols = 0.0
    countPunctuations = 0.0
    countVowels = 0.0
    lettermap = {}
    specsymbolmap = dict.fromkeys(Specsymbols, 0)
    punctuationmap = dict.fromkeys(Punctuations, 0)

    while i != len(text) - 1:
        if text[i].isdigit(): countDigits += 1
        if text[i].isalpha():
            countLetters += 1
            if text[i].lower() in Vowels: countVowels += 1
            if text[i].lower() not in lettermap.keys():
                lettermap[text[i].lower()] = 0
            lettermap[text[i].lower()] += 1
        if text[i].isupper(): countUppercaseLetters += 1
        if text[i] == ' ': countSpaces += 1
        if text[i] == '\t': countTabs += 1
        if text[i] in Specsymbols:
            countSpecsymbols += 1
            specsymbolmap[text[i]] += 1
        if text[i] in Punctuations:
            countPunctuations += 1
            punctuationmap[text[i]] += 1

        i += 1
        ResArray[7] = countLetters / len(text)
        # ResArray[8] = countDigits / len(text)
        ResArray[8] = countPunctuations / len(text)
        ResArray[9] = countSpecsymbols / len(text)
        ResArray[10] = countSpaces / len(text)
        # filename.write('Общее число символов(N): ' + str(len(text)) + "\n")
        # filename.write('Количество цифр/N: ' + str(countDigits / len(text)) + "\n")
        # filename.write('Количество букв/N: ' + str(countLetters / len(text)) + "\n")
        # filename.write('Количество заглавных букв/N: ' + str(countUppercaseLetters / len(text)) + "\n")
        # filename.write('Количество пробелов/N: ' + str(countSpaces / len(text)) + "\n")
        # filename.write('Количество табуляций/N: ' + str(countTabs / len(text)) + "\n")
        # filename.write('Количество спецсимволов/N: ' + str(countSpecsymbols / len(text)) + "\n")
        # filename.write('Количество пунктуационных символов/N: ' + str(countPunctuations / len(text)) + "\n")
        # filename.write('Счетчик букв:\n')
        # sortdict = lettermap.keys()
        # sortdict = list(sortdict)
        # sortdict.sort()
        # for letter in sortdict:
        #     filename.write(letter + ' ' + str((lettermap[letter])) + "\n")


def LexicalWordBasedFeature(sentences, ResArray):
    cntWords = 0.0
    cntSentences = len(sentences) - 1
    cntLenSentences = 0.0
    cntWordLenth = 0.0
    cntShortWords = 0.0
    cntLongWodrs = 0.0
    cntProperNames = 0.0
    firstWordinSentence = 1

    for sent in sentences:
        cntLenSentences += len(sent)
        words = tokenize(sent)
        cntWords += len(words)
        firstWordinSentence = 1
        for word in words:
            cntWordLenth += len(word)
            if len(word) < 4: cntShortWords += 1
            if firstWordinSentence == 0 and word.istitle(): cntProperNames += 1
            firstWordinSentence = 0

    avgWordLength = cntWordLenth / cntWords
    # filename.write("Общее количество слов(T): " + str(cntWords) + "\n")
    # filename.write("Общее количество предложений): " + str(cntSentences) + "\n")
    # filename.write("Средняя длина предложений(в символах):" + str(cntLenSentences / cntSentences) + "\n")
    # filename.write("Средняя длина слов: " + str(avgWordLength) + "\n")
    # filename.write("Короткие слова(1-3 символа)/T: " + str(cntShortWords / cntWords) + "\n")
    for sent in sentences:
        words = sent.split()
        for word in words:
            if len(word) >= avgWordLength: cntLongWodrs += 1

    # filename.write("Длинные слова(слова с длиной больше средней)/T: " + str(cntLongWodrs / cntWords) + "\n")
    # filename.write("Кооличество имен собственных/T: " + str(cntProperNames / cntWords) + "\n")
    ResArray[0] = cntLenSentences / cntSentences
    ResArray[1] = avgWordLength
    ResArray[2] = cntShortWords / cntWords
    ResArray[3] = cntLongWodrs / cntWords
    ResArray[4] = cntProperNames / cntWords


re1 = re.compile("""
    (?:
        (?:
            (?<!\\d(?:р|г|к))
            (?<!и\\.т\\.(?:д|п))
            (?<!и(?=\\.т\\.(?:д|п)\\.))
            (?<!и\\.т(?=\\.(?:д|п)\\.))
            (?<!руб|коп)
        \\.) |
        [!?\\n]
    )+
    """, re.X)


def f1(Res):
    # old 15 files return -0.66531213 * Res[1] - 1.74932097 * Res[3] + 1.55590547 * Res[5] + 0.01443481 * Res[6] + 2.5513533
    # old 30 files return 3.87344423 * Res[2] + 2.09653228 * Res[4] + 0.00931351 * Res[6] - 2.12646138
    return 0.00114659 * Res[0] - 1.11005179 * Res[1] - 1.52572186 * Res[3] - 0.79935266 * Res[5] + 0.01124218 * Res[
        6] + 53.43603776 * Res[9] - 48.3560366 * Res[10] + 13.28955791


def f2(Res):
    # olf return -0.37162769 * Res[1] - 2.15999379 * Res[3] + 4.51894416 * Res[5] + 0.00976478 * Res[6] + 0.17700341
    # old 30 files return 2.71987327 * Res[2] - 1.60477892 * Res[3] + 2.88512435 * Res[4] + 3.09086243 * Res[5] + 0.00976104 * Res[6] - 2.49476899
    return -0.41670645 * Res[1] + 1.10474101 * Res[2] - 1.80510707 * Res[3] + 1.49446929 * Res[4] + 2.53937585 * Res[
        5] + 0.01083192 * Res[6] - 6.95972671 * Res[7] - 33.55335051 * Res[9] - 27.19890629 * Res[10] + 10.59148444


def f3(Res):
    # old return 0.0061358 * Res[0] - 0.89109292 * Res[1] + 3.34285494 * Res[4] + 0.01302892 * Res[6] + 2.88912527
    # return 0.00259444 * Res[0] - 0.59364292 * Res[1] - 2.88400263 * Res[2] - 2.40832936 * Res[3] - 0.98683601 * Res[5] + 0.01126026 * Res[6] + 5.32713185
    return 0.00334412 * Res[0] - 0.80241887 * Res[1] - 3.08843481 * Res[2] - 2.70885979 * Res[3] - 0.89942445 * Res[
        5] + 0.01230323 * Res[6] + 15.42688185 * Res[7] + 23.25931525 * Res[8] + 43.3122369 * Res[9] - 6.88875441


def f4(Res):
    # old return 0.0084909 * Res[0] - 0.84377454 * Res[1] + 3.81141367 * Res[4] + 3.99526148 * Res[5] + 0.00803095 * Res[6] + 0.56127883
    # return 0.0035173 * Res[0] - 0.43246268 * Res[1] - 1.95920243 * Res[2] - 2.77518217 * Res[3] + 2.64232624 * Res[4] + 2.28100837 * Res[5] + 0.01164255 * Res[6] + 2.29302797
    return 0.00305075 * Res[0] - 2.49774282 * Res[3] + 2.63514397 * Res[4] + 2.36636696 * Res[5] + 0.01122454 * Res[
        6] + 11.48356245 * Res[7] + 28.32561341 * Res[8] - 38.53047966 * Res[9] + 20.24281479 * Res[10] + -13.7654428


def f5(Res):
    # return -0.00487343 * Res[0] - 2.5557295 * Res[5] + 2.55706235
    return -0.00181438 * Res[0] - 1.12130947 * Res[1] - 2.32413906 * Res[4] - 3.27167373 * Res[5] - 12.35495529 * Res[
        7] - 33.08816484 * Res[8] + 91.17542537 * Res[9] - 71.12103751 * Res[10] + 29.86021335

    # old  return -0.00610943 * Res[0] + 3.05443594 * Res[2] - 2.86487486 * Res[5] + 1.59262578


def f6(Res):
    # return -0.00379366 * Res[0] + 0.3698661 * Res[1] + 3.75422388 * Res[2] + 1.06826961 * Res[3] + 0.90457188 * Res[5] - 3.16389093
    return -0.00240753 * Res[0] - 0.22231009 * Res[1] + 3.47015636 * Res[2] + 1.33443051 * Res[3] - 17.74857373 * Res[
        7] - 27.55552932 * Res[8] - 49.51993133 * Res[10] + 22.77590225

    # old  return -0.00653362 * Res[0] + 0.62713759 * Res[1] + 3.86184454 * Res[2] - 3.04419763


def f7(Res):
    return -3.6218862 * Res[5] + 76.37126 * Res[9] + -8.24426171 * Res[10] + 3.95940794
    # return -3.26575849 * Res[5] + 2.66313208
    # old return 2.56711228 * Res[2] - 3.50941948 * Res[5] + 1.74198514


def f1_mr5(Res):
    if my_cmp(Res[1], 4.76, 1e-2, op.le) and Res[6] > 49.272 and my_cmp(Res[0], 107.298, 1e-3, op.le):
        return -0.0005 * Res[0] - 0.0724 * Res[1] + 1.3805 * Res[2] + 0.0109 * Res[6] + 4.7121 * Res[7] + 93.8703 * Res[
            9] - 4.4647
    if Res[0] > 72.158:
        return 0
    if Res[0] > 63.332:
        return 0
    if my_cmp(Res[0], 53.361, 1e-3, op.le):
        return 0.051 * Res[0] - 2.2661
    return 1
    # swap 1 abd 3
    # if my_cmp(Res[0], 103.203, 1e-3, op.le):
    #     return 0.0007 * Res[0] + 1.691 * Res[10] - 0.3112
    # if Res[0] > 143.198:
    #     return 0.0009 * Res[0] + 15.6087 * Res[10] - 1.7429
    # if my_cmp(Res[10], 0.161, 1e-3, op.le):
    #     return 58.7483 * Res[10] - 9.037
    # return 1


def f2_mr5(Res):
    if Res[0] > 103.737:
        return -0.0003 * Res[0] + 0.9715 * Res[5] + 0.0036 * Res[6] + 7.8353 * Res[8] - 0.8106
    if Res[8] > 0.031:
        return -0.0035 * Res[0] + 0.9583 * Res[5] + 0.0042 * Res[6] + 0.3317
    if my_cmp(Res[6], 60.319, 1e-3, op.le) and my_cmp(Res[5], 0.568, 1e-3, op.le):
        return -0.0102 * Res[0] + 1.6117 * Res[5] + 0.004 * Res[6] - 9.9906 * Res[8] - 7.2324 * Res[10] + 1.3125
    if Res[6] > 50.73:
        return -0.005 * Res[0] + 0.8481 * Res[5] + 0.006 * Res[6] + 0.3952
    if my_cmp(Res[5], 0.708, 1e-3, op.le) and my_cmp(Res[7], 0.808, 1e-3, op.le):
        return -0.0088 * Res[0] + 1.5346 * Res[5] + 8.3927 * Res[7] - 6.8305
    if my_cmp(Res[0], 81.569, 1e-3, op.le):
        return -0.0171 * Res[0] + 2.1992
    return -0.0666 * Res[0] + 6.4226

    # swap 1 and 3 class
    # if my_cmp(Res[0], 143.198, 1e-3, op.le) and Res[5] > 0.613:
    #     return 0.0006 * Res[0] + 3.3781 * Res[5] - 112.9966 * Res[9] - 1.4452
    # if my_cmp(Res[0], 143.198, 1e-3, op.le) and Res[5] > 0.522:
    #     return 0.0015 * Res[0] - 0.3313 * Res[2] - 1.3677 * Res[5] + 0.8036
    # if Res[0] > 143.198:
    #     return 0.0005 * Res[0] - 1.6226 * Res[2] + 1.4543
    # if Res[2] > 0.416:
    #     return 0.0053 * Res[0] - 0.2839
    # return 1


def f3_mr5(Res):
    if Res[0] > 143.198:
        return 0.0014 * Res[0] - 0.2712 * Res[1] - 0.4851 * Res[5] + 0.0043 * Res[6] - 8.8953 * Res[10] + 3.3446
    if my_cmp(Res[1], 4.76, 1e-2, op.le) and Res[6] > 57.914 and my_cmp(Res[5], 0.618, 1e-3, op.le):
        return 0.0044 * Res[0] - 0.3368 * Res[1] + 1.1131 * Res[2] - 0.5995 * Res[5] + 0.0073 * Res[6] - 17.7425 * Res[
            10] + 4.0979
    if my_cmp(Res[1], 4.724, 1e-3, op.le) and my_cmp(Res[8], 0.031, 1e-3, op.le) and my_cmp(Res[1], 4.643, 1e-3, op.le):
        return 0.1379 * Res[1] + 1.1754 * Res[4] + 3.8883 * Res[8] + 31.4462 * Res[9] - 0.8696
    if Res[1] > 4.724:
        return 0.0031 * Res[0] - 0.7518 * Res[1] + 46.6735 * Res[9] - 28.6013 * Res[10] + 7.5957
    if my_cmp(Res[9], 0.003, 1e-3, op.le) and Res[10] > 0.154:
        return -0.4337 * Res[1] + 71.6989 * Res[9] - 35.2233 * Res[10] + 7.4651
    return -10.5233 * Res[3] + 6.0751


def f4_mr5(Res):
    if Res[6] > 65.145:
        return 0.001 * Res[0] + 0.7695 * Res[4] + 0.7132 * Res[5] + 0.0046 * Res[6] + 7.8849 * Res[8] - 17.5871 * Res[
            9] - 0.1451
    if my_cmp(Res[0], 143.198, 1e-3, op.le) and my_cmp(Res[8], 0.031, 1e-3, op.le) and my_cmp(Res[5], 0.606, 1e-3,
                                                                                              op.le) and my_cmp(Res[6],
                                                                                                                55.18,
                                                                                                                1e-2,
                                                                                                                op.le):
        return -0.0015 * Res[0] + 2.3558 * Res[4] + 1.5867 * Res[5] + 0.0196 * Res[6] + 9.506 * Res[8] - 45.1201 * Res[
            9] - 1.8841
    if my_cmp(Res[9], 0.002, 1e-3, op.le):
        return -16.3512 * Res[9] + 1.0125
    if my_cmp(Res[0], 75.434, 1e-3, op.le):
        return -0.4199 * Res[1] + 2.7752
    if Res[5] > 0.497 and my_cmp(Res[1], 4.68, 1e-2, op.le):
        return 0
    if my_cmp(Res[1], 4.823, 1e-3, op.le):
        return -1.9594 * Res[1] + 10.1071
    return 0


def f5_mr5(Res):
    if my_cmp(Res[0], 143.198, 1e-3, op.le) and Res[5] > 0.613:
        return -0.0006 * Res[0] - 3.3781 * Res[5] + 112.9966 * Res[9] + 2.4452
    if my_cmp(Res[0], 143.198, 1e-3, op.le) and Res[5] > 0.522:
        return -0.0015 * Res[0] + 0.3313 * Res[2] + 1.3677 * Res[5] + 0.1964
    if Res[0] > 143.198:
        return -0.0005 * Res[0] + 1.6226 * Res[2] - 0.4543
    if Res[2] > 0.416:
        return -0.0053 * Res[0] + 1.2839
    return 0
    # swap 1 and 3
    # if Res[0] > 103.737:
    #     return 0.0003 * Res[0] - 0.9715 * Res[5] - 0.0036 * Res[6] - 7.8353 * Res[8] + 1.8106
    # if Res[8] > 0.031:
    #     return 0.0035 * Res[0] - 0.9583 * Res[5] - 0.0042 * Res[6] + 0.6683
    # if my_cmp(Res[6], 60.319, 1e-3, op.le) and my_cmp(Res[5], 0.568, 1e-3, op.le):
    #     return 0.0102 * Res[0] - 1.6117 * Res[5] - 0.004 * Res[6] + 9.9906 * Res[8] + 7.2324 * Res[10] - 0.3125
    # if Res[6] > 50.73:
    #     return 0.005 * Res[0] - 0.8481 * Res[5] - 0.006 * Res[6] + 0.6048
    # if my_cmp(Res[5], 0.708, 1e-3, op.le) and my_cmp(Res[7], 0.809, 1e-3, op.le):
    #     return 0.0088 * Res[0] - 1.5346 * Res[5] - 8.3927 * Res[7] + 7.8305
    # if my_cmp(Res[0], 81.569, 1e-3, op.le):
    #     return 0.0171 * Res[0] - 1.1992
    # return 0.0666 * Res[0] - 5.4226


def f6_mr5(Res):
    if my_cmp(Res[0], 103.203, 1e-3, op.le):
        return -0.0007 * Res[0] - 1.691 * Res[10] + 1.3112
    if Res[0] > 143.198:
        return -0.0009 * Res[0] - 15.6087 * Res[10] + 2.7429
    if my_cmp(Res[10], 0.161, 1e-3, op.le):
        return -58.7483 * Res[10] + 10.037
    return 0
    # swap 1 and 3
    # if my_cmp(Res[1], 4.76, 1e-2, op.le) and Res[6] > 49.272 and my_cmp(Res[0], 107.298, 1e-3, op.le):
    #     return 0.0005 * Res[0] + 0.0724 * Res[1] - 1.3805 * Res[2] - 0.0109 * Res[6] - 4.7121 * Res[7] - 93.8703 * Res[
    #         9] + 5.4647
    # if Res[0] > 72.158:
    #     return 1
    # if Res[0] > 63.332:
    #     return 1
    # if my_cmp(Res[0], 53.361, 1e-3, op.le):
    #     -0.051 * Res[0] + 3.2661
    # return 0


def f7_mr5(Res):
    if my_cmp(Res[5], 0.613, 1e-3, op.le):
        return -0.5669 * Res[5] + 12.4745 * Res[9] + 1.2566
    return -3.7672 * Res[5] + 124.6323 * Res[9] + 2.6204


def makevec_mr5(Res):
    res_vec = [round(math.fabs(f1_mr5(Res))), round(math.fabs(f2_mr5(Res))), round(math.fabs(f3_mr5(Res))),
               round(math.fabs(f4_mr5(Res))), round(math.fabs(f5_mr5(Res))),
               round(math.fabs(f6_mr5(Res))), round(math.fabs(f7_mr5(Res)))]
    res = []
    for i in res_vec:
        if i > 1: i = 1
        res.append(i)

    return res


def makevec(Res):
    res_vec = [round(math.fabs(f1(Res))), round(math.fabs(f2(Res))), round(math.fabs(f3(Res))),
               round(math.fabs(f4(Res))), round(math.fabs(f5(Res))),
               round(math.fabs(f6(Res))), round(math.fabs(f7(Res)))]
    res = []
    for i in res_vec:
        if i > 1: i = 1
        res.append(i)

    return res



def getArray(src):
    sys = open(src, 'r')
    text = "".join(sys.readlines())
    text = " ".join(text.split("\n"))
    ResArray = np.zeros(11)
    sentences = [s for s in safeSplit(re1, text)]
    LexicalWordBasedFeature(sentences, ResArray)
    LexicalCharacterFeature(text, ResArray)
    Hapax(text, ResArray)
    get_yules(text, ResArray)
    return makevec(ResArray)
    # return makevec_mr5(ResArray)


def out_file(src, out):
    files = os.listdir(src)
    outfile = open(str('res' + src[-2] + '_' + str(out) + '.txt'), 'w')
    for file in files:
        sys = open(str(src + file), 'r')
        text = "".join(sys.readlines())
        text = " ".join(text.split("\n"))
        ResArray = np.zeros(11)
        sentences = [s for s in safeSplit(re1, text)]
        LexicalCharacterFeature(text, ResArray)
        LexicalWordBasedFeature(sentences, ResArray)
        Hapax(text, ResArray)
        get_yules(text, ResArray)
        outfile.write(','.join(map(str, ResArray)) + ',' + str(out) + '.0' + '\n')
        sys.close()
    outfile.close()


def Hamming_strong(list_of_classifiers, testvec):
    for resclass, classvec in enumerate(list_of_classifiers, 1):
        i = 0
        cnt = 0
        while i < len(classvec):
            if classvec[i] != testvec[i]: cnt += 1
            i += 1
        if cnt <= 1: return ([resclass, cnt])
    return ([0, cnt])


def Hamming_soft(list_of_classifiers, testvec):
    res = []
    mincnt = 0
    flag = False
    for resclass, classvec in enumerate(list_of_classifiers, 1):
        i = 0
        cnt = 0
        while i < len(classvec):
            if classvec[i] != testvec[i]: cnt += 1
            i += 1
        if flag == False and mincnt == 0:
            mincnt = cnt
            flag = True
        if flag == True and cnt == mincnt: res.append(resclass)
        if flag == True and cnt < mincnt:
            res.clear()
            mincnt = cnt
            res.append(resclass)
    return res


def classify_folder_strong(folder, list_of_classifiers):
    files = os.listdir(folder)
    cntSuccess = 0
    cntMisClasify = 0
    cntCorrection = 0
    classid = int(folder[-2])
    for file in files:
        class_num = Hamming_strong(list_of_classifiers, getArray(str(folder + file)))
        if int(class_num[0]) == classid:
            cntSuccess += 1
            if class_num[1] == 1: cntCorrection += 1
        if class_num[0] != classid and class_num[0] != 0: cntMisClasify += 1
    resvec = np.array([len(files), cntSuccess, cntMisClasify, len(files) - cntMisClasify - cntSuccess, cntCorrection])
    return resvec


def classify_folder_mild(folder, list_of_classifiers):
    files = os.listdir(folder)
    cntSuccess = 0
    cntFail = 0
    cntAmongSuccess = 0
    classid = int(folder[-2])

    for file in files:
        class_list = Hamming_soft(list_of_classifiers, getArray(str(folder + file)))
        if len(class_list) == 1:
            if class_list.pop() == classid:
                cntSuccess += 1
            else:
                cntFail += 1
        for i in class_list:
            if i == classid: cntAmongSuccess += 1
    resvec = np.array([len(files), cntSuccess, cntFail, cntAmongSuccess])
    return resvec


def print_res_all(seq):
    res_strong = np.zeros(5)
    res_mild = np.zeros(4)
    for i in seq:
        res_strong += classify_folder_strong(i, list_of_classifiers)
        res_mild += classify_folder_mild(i, list_of_classifiers)
    print('Жесткое декодирование')
    print('Всего было обработано {} файла, из них:'.format(int(res_strong[0])))
    print('Верно определен класс: ' + str(int(res_strong[1])))
    print('Из верно определенных было исправление: {}'.format(int(res_strong[4])))
    print('Класс определен, но не верно: ' + str(int(res_strong[2])))
    print('Не декодируется ни в один из классов: ' + str(int(res_strong[3])))
    print()
    print('Мягкое декодирование')
    print('Всего было обработано {} файла, из них:'.format(int(res_mild[0])))
    print('Верно однозначно определен класс: ' + str(int(res_mild[1])))
    print('Из верно определенных было исправление: {}'.format(int(res_strong[4])))
    print('Неверно определен класс: ' + str(int(res_mild[2])))
    print('Класс определен верно, но неоднозначно: ' + str(int(res_mild[3])))


trainingset1 = 'dataset/diminC1/'
trainingset2 = 'dataset/Tyler-84C2/'
trainingset3 = 'dataset/VadimCzechC3/'
trainingset4 = 'dataset/LandolkaC4/'
testset1 = 'test_data/data_for_testC1/'
testset2 = 'test_data/data_for_testC2/'
testset3 = 'test_data/data_for_testC3/'
testset4 = 'test_data/data_for_testC4/'
list_of_classifiers = [[1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 0, 0, 1], [0, 1, 0, 1, 0, 1, 0]]

# classify_folder_strong(testset1, list_of_classifiers)
# print(classify_folder_mild(testset1, list_of_classifiers))
print_res_all([testset2, testset1, testset4, testset3])
# out_file('dataset/diminC1/', 0)
# out_file('dataset/diminC1/', 1)
# out_file('dataset/Tyler-84C2/', 0)
# out_file('dataset/Tyler-84C2/', 1)
# out_file('dataset/VadimCzechC3/', 0)
# out_file('dataset/VadimCzechC3/', 1)
# out_file('dataset/LandolkaC4/', 0)
# out_file('dataset/LandolkaC4/', 1)
# check_date('data_for_testC3/','data_for_testC3/')
# print('------')
# check_date(testset4,trainingset4)
# print('------')
# check_date(testset4,testset4)
