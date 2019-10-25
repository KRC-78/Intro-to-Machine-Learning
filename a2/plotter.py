import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def plot(data,val):
    train_acc = data[1]
    train_loss = (data[0])
    val_acc = data[3]
    val_loss = (data[2])
    test_acc = data[5]
    test_loss = (data[4])

    fig = plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(train_loss, label = 'Training Loss')
    plt.plot(val_loss, label = 'Validation Loss')
    plt.plot(test_loss, label = 'Test Loss')
    plt.title("Loss vs Epochs with Dropout (p = {})".format(val))
    plt.legend(loc = 'best') 
    plt.savefig("dp_{} Loss.png".format(val))
    plt.show()
    
    fig = plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(train_acc, label = 'Training Accuracy')
    plt.plot(val_acc, label = 'Validation Accuracy')
    plt.plot(test_acc, label = 'Test Accuracy')
    plt.title("Accuracy vs Epochs with Dropout (p = {})".format(val))
    plt.legend(loc = 'best')
    plt.ylim(0.0, 1.1)
    plt.savefig("dp_{} Accuracy.png".format(val))
    plt.show()
    return
