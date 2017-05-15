# -*- coding: utf-8 -*-
import re
import collections
import math
import os
import numpy as np

np.set_printoptions(precision=5, suppress=True)


def tokenize(s):
    tokens = re.split(r"[^0-9A-Za-zа-яА-Я\-'_]+", s)
    return tokens


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
    # old return -0.66531213 * Res[1] - 1.74932097 * Res[3] + 1.55590547 * Res[5] + 0.01443481 * Res[6] + 2.5513533
    return 3.87344423 * Res[2] + 2.09653228 * Res[4] + 0.00931351 * Res[6] - 2.12646138


def f2(Res):
    # olf return -0.37162769 * Res[1] - 2.15999379 * Res[3] + 4.51894416 * Res[5] + 0.00976478 * Res[6] + 0.17700341
    return 2.71987327 * Res[2] - 1.60477892 * Res[3] + 2.88512435 * Res[4] + 3.09086243 * Res[5] + 0.00976104 * Res[
        6] - 2.49476899


def f3(Res):
    # old return 0.0061358 * Res[0] - 0.89109292 * Res[1] + 3.34285494 * Res[4] + 0.01302892 * Res[6] + 2.88912527
    return 0.00259444 * Res[0] - 0.59364292 * Res[1] - 2.88400263 * Res[2] - 2.40832936 * Res[3] - 0.98683601 * Res[
        5] + 0.01126026 * Res[6] + 5.32713185


def f4(Res):
    # old return 0.0084909 * Res[0] - 0.84377454 * Res[1] + 3.81141367 * Res[4] + 3.99526148 * Res[5] + 0.00803095 * Res[6] + 0.56127883
    return 0.0035173 * Res[0] - 0.43246268 * Res[1] - 1.95920243 * Res[2] - 2.77518217 * Res[3] + 2.64232624 * Res[
        4] + 2.28100837 * Res[5] + 0.01164255 * Res[6] + 2.29302797


def f5(Res):
    return -0.00487343 * Res[0] - 2.5557295 * Res[5] + 2.55706235


    # old  return -0.00610943 * Res[0] + 3.05443594 * Res[2] - 2.86487486 * Res[5] + 1.59262578


def f6(Res):
    return -0.00379366 * Res[0] + 0.3698661 * Res[1] + 3.75422388 * Res[2] + 1.06826961 * Res[3] + 0.90457188 * Res[
        5] - 3.16389093


    # old  return -0.00653362 * Res[0] + 0.62713759 * Res[1] + 3.86184454 * Res[2] - 3.04419763


def f7(Res):
    return -3.26575849 * Res[5] + 2.66313208
    # old return 2.56711228 * Res[2] - 3.50941948 * Res[5] + 1.74198514


def makevec(Res):
    # print(f1(Res))
    # print(f2(Res))
    # print(f3(Res))
    # print(f4(Res))
    # print(f5(Res))
    # print(f6(Res))
    # print(f7(Res))
    res_vec = [round(math.fabs(f1(Res))), round(math.fabs(f2(Res))), round(math.fabs(f3(Res))),
               round(math.fabs(f4(Res))), round(math.fabs(f5(Res))),
               round(math.fabs(f6(Res))), round(math.fabs(f7(Res)))]
    return res_vec

    # j = 0
    # while j <= 1:
    # outfile = open(str('res4_' + str(j) + '.txt'), 'w')


def getArray(src):
    sys = open(src, 'r')
    text = "".join(sys.readlines())
    text = " ".join(text.split("\n"))
    ResArray = np.zeros(7)

    sentences = [s for s in safeSplit(re1, text)]
    res = AvgSentLength(sentences)
    LexicalWordBasedFeature(sentences, ResArray)
    Hapax(text, ResArray)
    get_yules(text, ResArray)
    return makevec(ResArray)

    # while i <= 30:
    # sys = open(str('dataset/diminC1/f__' + str(i) + '.txt'), 'r')
    # sys = open(str('dataset/Tyler-84C2/f__' + str(i) + '.txt'), 'r')
    # sys = open(str('dataset/VadimCzechC3/f__' + str(i) + '.txt'), 'r')
    # sys = open(str('dataset/LandolkaC4/f__' + str(i) + '.txt'), 'r')
    # sys = open(str('dataset/test.txt'), 'r')

    # text = "".join(sys.readlines())

    # text = " ".join(text.split("\n"))
    # ResArray = np.zeros(7)

    # sentences = [s for s in safeSplit(re1, text)]
    # res = AvgSentLength(sentences)
    # LexicalCharacterFeature(text, outfile, ResArray)
    # LexicalWordBasedFeature(sentences, outfile, ResArray)
    # Hapax(text, outfile, ResArray)
    # get_yules(text, outfile, ResArray)
    # print(ResArray)
    # print(makevec(ResArray))
    # outfile.write(','.join(map(str, ResArray)) + ',' + str(j) + '.0' + '\n')


list_of_classifiers = [[1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 0, 0, 1], [0, 1, 0, 1, 0, 1, 0]]


def Hamming_strong(list_of_classifiers, testvec):
    for resclass, classvec in enumerate(list_of_classifiers, 1):
        i = 0
        cnt = 0
        while i < len(classvec):
            if classvec[i] != testvec[i]: cnt += 1
            i += 1
        if cnt <= 2: return resclass
    return 0


def classify_folder_strong(folder, list_of_classifiers, classid):
    files = os.listdir(folder)
    cntSuccess = 0
    cntMisClasify = 0
    for file in files:
        class_num = Hamming_strong(list_of_classifiers, getArray(str(folder + file)))
        if class_num == classid: cntSuccess += 1
        if class_num != classid and class_num != 0: cntMisClasify += 1

    print('Всего было обработано ' + str(len(files)) + ' файлов, из них:')
    print('Верно определен класс: ' + str(cntSuccess))
    print('Класс определен, но не верно: ' + str(cntMisClasify))
    print('Не декодируется ни в один из классов: ' + str(len(files) - cntMisClasify - cntSuccess))


classify_folder_strong('data_for_classification/', list_of_classifiers, 1)
