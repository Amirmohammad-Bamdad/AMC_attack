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
    model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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

def white_box_accuracy_test(Y, Y_adv_signal, input_label, classes):
    class_Y = classes[int(np.argmax(Y, axis=1))]
    class_Y_adv_signal = classes[int(np.argmax(Y_adv_signal, axis=1))]
    class_input_label = classes[int(np.argmax(input_label, axis=1))]
    
    print(f"GroundTruth Label: {class_input_label}")
    print(f"Original Model Detected Label: {class_Y}")
    print(f"Attacked signal Detected Label: {class_Y_adv_signal}")
    print("==================================================")
    

def FGSM(model, epsilons, input_signal, input_label, classes):
    for eps in epsilons:
        gradient = calculate_gradient(model, input_signal, input_label)
        perturbation = tf.sign(gradient)
        adv_signal = input_signal + eps*perturbation
        Y_adv_signal = model(adv_signal)

        Y = model.predict(input_signal)
        print(f"epsilon = {eps}")
        white_box_accuracy_test(Y, Y_adv_signal, input_label, classes)
        

def bisection_search_whitebox_attack(model, input_data, input_label, classes):
    eps_acc = 0.00001 * np.linalg.norm(input_data)  
    target_class_onehot = np.zeros([len(classes)])
    epsilon_vector = np.zeros([len(classes)])
    
    for class_index in range(len(classes)):
        max_epsilon = np.linalg.norm(input_data)
        min_epsilon = 0
        r = calculate_gradient(model, input_data, input_label) # adversary perturbation
        r_norm = r / np.linalg.norm(r)
        
        while (max_epsilon - min_epsilon > eps_acc):
            
            avg_epsilon = (max_epsilon + min_epsilon)/2
            adv_x = input_data - r_norm*avg_epsilon

            adv_label = model.predict(adv_x, verbose='False')
            
            if np.argmax(adv_label, axis=1) == np.argmax(input_label, axis=1):
                min_epsilon = avg_epsilon
            else:
                max_epsilon = avg_epsilon
            #print(adv_label)
        epsilon_vector[class_index] = max_epsilon

    target_class = np.argmin(epsilon_vector)
    target_class_onehot[target_class] = 1
    target_class_onehot = tf.expand_dims(target_class_onehot, axis=0)

    epsilon_star = np.min(epsilon_vector)

    perturbation = calculate_gradient(model, input_data, target_class_onehot)
    norm_perturbation = perturbation / np.linalg.norm(perturbation)
    perturbation = epsilon_star * norm_perturbation

    adv_input = input_data - perturbation
    Y_adv_signal = model.predict(adv_input)
    Y = model.predict(input_data)

    white_box_accuracy_test(Y, Y_adv_signal, input_label, classes)



def black_box_attack_test(model, r, data, labels):
    adv_data = data + r
    test_loss, test_acc  = model.evaluate(data, labels, batch_size= 256)
    adv_test_loss, adv_test_acc  = model.evaluate(adv_data, labels, batch_size= 256)
    print("Test accuracy (Normal): ", test_acc)
    print("Test loss (Normal): ", test_loss)
    print("Test accuracy (Adversarial): ", adv_test_acc)
    print("Test loss (Adversarial): ", adv_test_loss)


def pca_based_black_box_attack(model, data_points, label_points, x_test, y_test):
    max_epsilon = np.linalg.norm(data_points)
    
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
    UAP_r = tf.expand_dims(UAP_r, axis=-1)

    extended_UAP = tf.tile(UAP_r, [1,2,128])

    black_box_attack_test(model, extended_UAP, x_test, y_test)



if __name__ == "__main__":
    model, loss_func, epsilons, snrs, x_test, y_test, x_val, y_val, mods = initialize_parameters()
    input_signal, input_label = sample_input(x_test, y_test)
    
    reshaped_input = tf.expand_dims(input_signal, axis=0)
    reshaped_label = tf.expand_dims(input_label, axis=0)

    # White-Box Attack
    #FGSM(model, epsilons, reshaped_input, reshaped_label, mods)
    bisection_search_whitebox_attack(model, reshaped_input, reshaped_label, mods)
    
    # BLack-Box Attack
    #pca_based_black_box_attack(model, x_val, y_val, x_test, y_test)
    