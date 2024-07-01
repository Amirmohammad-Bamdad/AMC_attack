import tensorflow as tf
import modelFile
import loader
import random
import numpy as np

def initialize_parameters():
    weight_path = './weights.h5'
    path = 'E:\Clemson\Codes\AMC_attack\RML2016.10a\RML2016.10a_dict.pkl'
    data = loader.RMLDataset(path)
    mods = data.mods
    snrs = data.snrs
    x_test, y_test = data.test_data[0], data.test_data[1]

    m = modelFile.CNNModel(input_shape=(2, 128), classes= len(mods))
    model = m.model
    model.load_weights(weight_path)
    loss_func = m.loss
    epsilons = [0.0001, 0.0005, 0.0008, 0.001, 0.005, 0.008, 0.01, 0.5]

    return model, loss_func, epsilons, snrs, x_test, y_test, mods


def sample_input(inputs, labels):
    random_index = random.randrange(len(inputs))
    return inputs[random_index], labels[random_index]


def create_perturbation(model, input_signal, input_label):
    with tf.GradientTape() as tape:
        tape.watch(input_signal)
        y = model(input_signal)
        loss = loss_func(input_label, y)

    gradient = tape.gradient(loss, input_signal)
    sign = tf.sign(gradient)
    return sign


def FGSM(model, epsilons, input_signal, input_label, classes):
    for eps in epsilons:
        perturbation = create_perturbation(model, input_signal, input_label)
        Y = model(input_signal)
        adv_signal = input_signal + eps*perturbation
        Y_adv_signal = model(adv_signal)
        
        class_Y = classes[int(np.argmax(Y, axis=1))]
        class_Y_adv_signal = classes[int(np.argmax(Y_adv_signal, axis=1))]
        class_input_label = classes[int(np.argmax(input_label, axis=1))]
        
        print(f"eps = {eps}\nGroundTruth Label: {class_input_label}")
        print(f"Original Model Detected Label: {class_Y}")
        print(f"Attacked signal Detected Label: {class_Y_adv_signal}")
        print("==================================================")


if __name__ == "__main__":
    model, loss_func, epsilons, snrs, x_test, y_test, mods = initialize_parameters()
    input_signal, input_label = sample_input(x_test, y_test)

    reshaped_input = tf.expand_dims(input_signal, axis=0)
    reshaped_label = tf.expand_dims(input_label, axis=0)

    FGSM(model, epsilons, reshaped_input, reshaped_label, mods)