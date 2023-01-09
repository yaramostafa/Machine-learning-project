import numpy as np
from matplotlib import pyplot
from sources.samples import (loadDataFile,loadLabelsFile)


def visualize(t_x):
    #visualize first 9 instances of dataset
    for i in range(9):
        pyplot.subplot(330 + 1 + i)
        pyplot.imshow(t_x[i], cmap=pyplot.get_cmap('gray'))
    pyplot.show()

#Each digit is 28x28 pixels, and each face/non-face image is 60x74 , this is an error data is 60x70 not 74
def digit_train():
    train_digits_X = []

    with open('data/digitdata/trainingimages', 'r', encoding='utf-8') as file:

        c = 0
        datum = []
        for line in file:
            l =[]
            for char in line:
                if char == ' ':
                    l.append(int(0))
                elif char == '+':
                    l.append(int(1))
                elif char == '#':
                    l.append(int(2))
            datum.append(l)
            c=c+1
            if c%28 == 0:
                train_digits_X.append(datum)
                datum =[]


    train_digits_X = np.array(train_digits_X,dtype=np.int32)
    train_digits_Y = loadLabelsFile("data/digitdata/traininglabels", 5000)
    print("train labels " , len(train_digits_Y))

    return train_digits_X,train_digits_Y


def digit_test():
    train_digits_X = []

    with open('data/digitdata/testimages', 'r', encoding='utf-8') as file:

        c = 0
        datum = []
        for line in file:
            l =[]
            for char in line:
                if char ==' ':
                    l.append(int(0))
                elif char == '+':
                    l.append(int(1))
                elif char == '#':
                    l.append(int(2))

            datum.append(l)
            c=c+1
            if c%28 == 0:
                train_digits_X.append(datum)
                datum =[]

    train_digits_X = np.array(train_digits_X,dtype=np.int32)
    train_digits_Y = loadLabelsFile("data/digitdata/testlabels", 1000)


    return train_digits_X,train_digits_Y



def digit_valid():
    train_digits_X = []

    with open('data/digitdata/validationimages', 'r', encoding='utf-8') as file:

        c = 0
        datum = []
        for line in file:
            l =[]
            for char in line:
                if char ==' ':
                    l.append(int(0))
                elif  char == '+':
                    l.append(int(1))
                elif char == '#':
                    l.append(int(2))
            datum.append(l)
            c=c+1
            if c%28 == 0:
                train_digits_X.append(datum)
                datum =[]

    train_digits_X = np.array(train_digits_X,dtype=np.int32)
    train_digits_Y = loadLabelsFile("data/digitdata/validationlabels", 1000)
    return train_digits_X , train_digits_Y




def face_train():
    train_digits_X = []

    with open('data/facedata/facedatatrain', 'r', encoding='utf-8') as file:

        c = 0
        datum = []
        for line in file:
            l =[]
            for char in line:
                if char ==' ':
                    l.append(int(0))
                elif  char == '+':
                    l.append(int(1))
                elif char == '#':
                    l.append(int(2))
            datum.append(l)
            c=c+1
            if c%70 == 0:
                train_digits_X.append(datum)
                datum =[]

    train_digits_X = np.array(train_digits_X,dtype=np.int32)
    train_digits_Y = loadLabelsFile("data/facedata/facedatatrainlabels", 451)


    return train_digits_X , train_digits_Y



def face_valid():
    train_digits_X = []

    with open('data/facedata/facedatavalidation', 'r', encoding='utf-8') as file:

        c = 0
        datum = []
        for line in file:
            l =[]
            for char in line:
                if char ==' ':
                    l.append(int(0))
                elif  char == '+':
                    l.append(int(1))
                elif char == '#':
                    l.append(int(2))
            datum.append(l)
            c=c+1
            if c%70 == 0:
                train_digits_X.append(datum)
                datum =[]

    train_digits_X = np.array(train_digits_X,dtype=np.int32)
    train_digits_Y = loadLabelsFile("data/facedata/facedatavalidationlabels", 301)
    return train_digits_X , train_digits_Y




def face_test():
    train_digits_X = []

    with open('data/facedata/facedatatest', 'r', encoding='utf-8') as file:

        c = 0
        datum = []
        for line in file:
            l =[]
            for char in line:
                if char ==' ':
                    l.append(int(0))
                elif  char == '+':
                    l.append(int(1))
                elif char == '#':
                    l.append(int(2))
            datum.append(l)
            c=c+1
            if c%70 == 0:
                train_digits_X.append(datum)
                datum =[]

    train_digits_X = np.array(train_digits_X,dtype=np.int32)
    train_digits_Y = loadLabelsFile("data/facedata/facedatatestlabels", 150)
    return train_digits_X , train_digits_Y

#451 60 70