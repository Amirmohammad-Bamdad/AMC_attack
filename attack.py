import tensorflow as tf
import modelFile
import loader
import random
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA

def initialize_parameters():
    weight_path = './weights.h5'
    path = 'E:\Clemson\Codes\AMC_attack\RML2016.10a\RML2016.10a_dict.pkl'
    data = loader.RMLDataset(path)
    mods = data.mods
    snrs = data.snrs
    x_val, y_val = data.val_data[0], data.val_data[1]
    x_test, y_test = data.test_data[0], data.test_data[1]

    m = modelFile.CNNModel(input_shape=(2, 128), classes= len(mods))
    model = m.model
    model.load_weights(weight_path)
    loss_func = m.loss
    epsilons = [0.0001, 0.0005, 0.0008, 0.001, 0.005, 0.008, 0.01, 0.5]

    return model, loss_func, epsilons, snrs, x_test, y_test, x_val, y_val, mods


def sample_input(inputs, labels):
    random_index = random.randrange(len(inputs))
    return inputs[random_index], labels[random_index]


def calculate_gradient(model, input_signal, input_label):
    with tf.GradientTape() as tape:
        tape.watch(input_signal)
        y = model(input_signal)
        loss = loss_func(input_label, y)

    gradient = tape.gradient(loss, input_signal)
    
    return gradient


def FGSM(model, epsilons, input_signal, input_label, classes):
    for eps in epsilons:
        gradient = calculate_gradient(model, input_signal, input_label)
        perturbation = tf.sign(gradient)
        adv_signal = input_signal + eps*perturbation
        Y_adv_signal = model(adv_signal)

        Y = model(input_signal)
        
        class_Y = classes[int(np.argmax(Y, axis=1))]
        class_Y_adv_signal = classes[int(np.argmax(Y_adv_signal, axis=1))]
        class_input_label = classes[int(np.argmax(input_label, axis=1))]
        
        print(f"eps = {eps}\nGroundTruth Label: {class_input_label}")
        print(f"Original Model Detected Label: {class_Y}")
        print(f"Attacked signal Detected Label: {class_Y_adv_signal}")
        print("==================================================")


def bisection_search_whitebox_attack():
    pass

def black_box_attack_test(model, r, data, labels):
    adv_data = data + r
    test_loss, test_acc  = model.evaluate(data, labels, batch_size= 256)
    adv_test_loss, adv_test_acc  = model.evaluate(adv_data, labels, batch_size= 256)
    print("Test accuracy (Normal): ", test_acc)
    print("Test loss (Normal): ", test_loss)
    print("Test accuracy (Normal): ", adv_test_acc)
    print("Test loss (Normal): ", adv_test_loss)


def pca_based_black_box_attack(model, data_points, label_points, x_test, y_test):
    max_epsilon = np.linalg.norm(data_points)
    
    #for i, data in tqdm(enumerate(data_points)):
    #    data = tf.convert_to_tensor(data)
    #    data = tf.expand_dims(data, axis=0)
    #    label = tf.expand_dims(label_points[i], axis=0)
#
    #    grad = calculate_gradient(model, data, label)
    #    norm_grad = grad / np.linalg.norm(grad)
    #    X.append(norm_grad)
#
    #X = np.vstack(X)
    #X = tf.convert_to_tensor(X)
    #print(X.shape)
    data_points = tf.convert_to_tensor(data_points)
    label_points = tf.convert_to_tensor(label_points)
    grad = calculate_gradient(model, data_points, label_points)
    X = grad / np.linalg.norm(grad)
    X = tf.convert_to_tensor(X)

    pca = PCA(n_components= 1)

    nsamples, nx, ny = X.shape
    X = tf.reshape(X, [nsamples, nx*ny])
    
    pca.fit_transform(X)
    
    #Projection
    v1 = pca.transform(X)
    UAP_r = max_epsilon * v1

    black_box_attack_test(model, UAP_r, x_test, y_test)



if __name__ == "__main__":
    model, loss_func, epsilons, snrs, x_test, y_test, x_val, y_val, mods = initialize_parameters()
    input_signal, input_label = sample_input(x_test, y_test)
    
    reshaped_input = tf.expand_dims(input_signal, axis=0)
    reshaped_label = tf.expand_dims(input_label, axis=0)

    # White-Box Attack
    FGSM(model, epsilons, reshaped_input, reshaped_label, mods)
    
    # BLack-Box Attack
    #pca_based_black_box_attack(model, x_val, y_val, x_test, y_test)
    