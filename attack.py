import tensorflow as tf
import modelFile
import loader
import random
import numpy as np
import utils
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

##########################################################
model_name = "VTCNN"
oracle_name = "LSTM_AMC"

loss_func = tf.keras.losses.CategoricalCrossentropy()

model_VTCNN = modelFile.VTCNN(input_shape=(2, 128), num_classes= 11).model   ## VT-CNN
model_LSTM = modelFile.LSTM_AMC(input_shape=(2, 128), num_classes= 11)       ## LSTM
##########################################################

def initialize_parameters():
    weight_path = f'./{model_name}_weights/'
    path = 'E:\Clemson\Codes\AMC_attack\RML2016.10a\RML2016.10a_dict.pkl'
    data = loader.RMLDataset(path)
    mods = data.mods
    snrs = data.snrs
    labels = data.label
    test_indices = data.test_indices

    x_val, y_val = data.val_data[0], data.val_data[1] # Using val data or training purpose.(Because it is not too large like train set also has no overlap with test set)
    x_test, y_test = data.test_data[0], data.test_data[1]
    
    if model_name=="VTCNN": 
        model = model_VTCNN
    elif model_name=="LSTM_AMC":
        model = model_LSTM

    model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model = tf.saved_model.load(weight_path)

    epsilons = [0.0001, 0.0005, 0.0008, 0.001, 0.005, 0.008, 0.01, 0.5, 0.8, 1, 1.5, 2]

    return model, epsilons, snrs, x_test, y_test, x_val, y_val, mods, labels, test_indices


def sample_input(inputs, labels):
    random_index = random.randrange(len(inputs))
    return inputs[random_index], labels[random_index]

@tf.function
def calculate_gradient(model, input_signal, input_label):
    with tf.GradientTape() as tape:
        tape.watch(input_signal)
        y = model(input_signal, training=False)
        loss = loss_func(input_label, y)

    gradient = tape.gradient(loss, input_signal)
    return gradient


def single_sample_test(Y, Y_adv_signal, input_label, classes):
    class_Y = classes[int(np.argmax(Y, axis=1))]
    class_Y_adv_signal = classes[int(np.argmax(Y_adv_signal, axis=1))]
    class_input_label = classes[int(np.argmax(input_label, axis=1))]
    
    print(f"GroundTruth Label: {class_input_label}")
    print(f"Original Model Detected Label: {class_Y}")
    print(f"Attacked signal Detected Label: {class_Y_adv_signal}")
    print("==================================================")
    

def black_box(oracle, subtitute_model, attack, x_test, y_test, snrs, mods, 
              test_indices, snr_labels, epsilons):
    '''
    Oracle is the model that we want to attack to. We don't know its parameters or architecture.
    subtitute model is the model that we use to create perturbations for dataset (here 
    we use test set) and pass the preturbed data to the oracle for test.
    '''

    if attack == "FGSM":
        perturbation = FGSM(model= subtitute_model, epsilons= epsilons,
                             input_signal= x_test, input_label= y_test, x_test=None, 
                             y_test= None, snrs= None, classes= None, test_indices=None
                             , snr_labels=None, test_acc= False)
    elif attack == "PGD":
        perturbation = pgd_attack(model= subtitute_model, data_points=x_test, 
                                  label_points= y_test, x_test=None, y_test=None,
                                  iters= 20, eps= 0.5, snrs=None, mods=None, test_indices=None,
                                  snr_labels=None, test_acc=False)

    print(perturbation)
    adversary_test(oracle, perturbation, x_test, y_test, snrs, mods, 
              test_indices, snr_labels, attack_name= f"BlackBox_{attack}_attack_on_{oracle_name}")



def adversary_test(model, r, data, labels, snrs, mods,
                    test_indices, snr_labels, attack_name, is_blackbox= False):
    adv_data = data + r
    
    #Normal
    y_test_hat = model(data, training=False)
    
    max_values = tf.reduce_max(y_test_hat, axis=1, keepdims=True)
    mask = tf.equal(y_test_hat, max_values)
    y_test_hat = tf.cast(mask, dtype=tf.int32) 

    test_loss = loss_func(y_test_hat, labels)
    test_acc = accuracy_score(tf.math.argmax(labels, axis=1), tf.math.argmax(y_test_hat, axis=1))

    
    #Adversarial
    adv_y_test_hat = model(adv_data, training=False)
 
    adv_max_values = tf.reduce_max(adv_y_test_hat, axis=1, keepdims=True)
    adv_mask = tf.equal(adv_y_test_hat, adv_max_values)
    adv_y_test_hat = tf.cast(adv_mask, dtype=tf.int32) 
    
    #print(tf.argmax(adv_y_test_hat, 1))
    #adv_y_test_hat = tf.argmax(adv_y_test_hat, 1)
    #labels= tf.argmax(labels, 1)
    adv_test_loss = loss_func(adv_y_test_hat, labels)
    adv_test_acc = accuracy_score(tf.math.argmax(labels, axis=1), tf.math.argmax(adv_y_test_hat, axis=1))

    #print(np.sum(np.argmax(adv_y_test_hat, axis=1)==np.argmax(labels, axis=1)))
    #print(np.sum(np.argmax(y_test_hat, axis=1)==np.argmax(labels, axis=1)))
    
    print("Test accuracy (Normal): ", test_acc)
    print("Test loss (Normal): ", test_loss.numpy())
    print("Test accuracy (Adversarial): ", adv_test_acc)
    print("Test loss (Adversarial): ", adv_test_loss.numpy())
    
    proto_tensor = tf.make_tensor_proto(adv_data)
    adv_data = tf.make_ndarray(proto_tensor)
    
    if is_blackbox:
        name = oracle_name
    else:
        name = model_name

    _, acc_mod_snr, bers = utils.evaluate_per_snr(model= model, X_test= adv_data, Y_test= labels,
                                           snrs= snrs, classes= mods, labels= snr_labels,
                                             test_indices= test_indices, model_name= name)
    
    utils.plot_accuracy_per_snr(snrs= snrs, acc_mod_snr= acc_mod_snr, 
                                classes= mods, name= attack_name, model_name= name)
    utils.plot_ber_vs_snr(snrs, bers, name= attack_name, model_name= name)


def FGSM(model, epsilons, input_signal, input_label, x_test, y_test,
          snrs, classes, test_indices, snr_labels, test_acc = True):
    '''
        https://arxiv.org/pdf/1412.6572
    '''
    input_signal = tf.convert_to_tensor(input_signal)
    input_label = tf.convert_to_tensor(input_label)
    r = None
    
    for eps in epsilons:
        gradient = calculate_gradient(model, input_signal, input_label)
        perturbation = tf.sign(gradient)
        perturbation = eps*perturbation
        adv_signal = input_signal + perturbation
        Y_adv_signal = model(adv_signal, training=False)
        #Y = model(input_signal)
        thresh_size = 0.15

        #print(f"epsilon = {eps}")
        #single_sample_test(Y, Y_adv_signal, input_label, classes)
        #Y = np.argmax(Y, axis=1)
        #Y_adv_signal = np.argmax(Y_adv_signal, axis=1)
        
        max_values = tf.reduce_max(Y_adv_signal, axis=1, keepdims=True)
        mask = tf.equal(Y_adv_signal, max_values)
        Y_adv_signal = tf.cast(mask, dtype=tf.int32)    

        #max_values = tf.reduce_max(Y, axis=1, keepdims=True)
        #mask = tf.equal(Y, max_values)
        #Y = tf.cast(mask, dtype=tf.int32)

        #print(thresh_size)
        #print(np.sum(Y == Y_adv_signal))
        acc = accuracy_score(Y_adv_signal, input_label)
        #advacc2= accuracy_score(Y_adv_signal, input_label)
        #print(advacc)
        #print(advacc2)

        #acc = accuracy_score(Y, input_label).numpy()
        #acc2= accuracy_score(Y, input_label)
        #print(acc)
        #print(acc2)
        
        if (acc < thresh_size) or (eps == epsilons[-1]):
            print(f"The best epsilon is: {eps}")
            r= perturbation
            break
    
    if test_acc:
        adversary_test(model= model, r= r, data= x_test, labels= y_test,
                        snrs= snrs, mods= classes, test_indices= test_indices,
                          snr_labels= snr_labels, attack_name= "FGSM")
    return r
        

def pgd_attack(model, data_points, label_points, x_test, y_test, iters, eps,
                            snrs, mods, test_indices, snr_labels, test_acc=True):
    data_points = tf.convert_to_tensor(data_points)
    label_points = tf.convert_to_tensor(label_points)
    adv_signal = tf.identity(data_points)
    perturbation = tf.zeros((22000, 2, 128)) 

    for _ in range(iters):
        grad = calculate_gradient(model, adv_signal, label_points)
        tmp = tf.sign(grad)
        perturbation += eps* tmp
        adv_signal = adv_signal + perturbation
        adv_signal = tf.clip_by_value(adv_signal, adv_signal - eps, adv_signal + eps)

    perturbation = eps*perturbation
    if test_acc:
        adversary_test(model= model, r= perturbation, data= x_test, labels= y_test,
                        snrs= snrs, mods= mods, test_indices= test_indices,
                          snr_labels= snr_labels, attack_name= "PGD")
    return perturbation



def bisection_search_FGM(model, input_data, input_label, classes, snr_labels):
    '''
    https://arxiv.org/abs/1808.07713
    '''
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

            adv_label = model(adv_x, verbose='False')
            
            if np.argmax(adv_label, axis=1) == np.argmax(input_label, axis=1):
                min_epsilon = avg_epsilon
            else:
                max_epsilon = avg_epsilon

        epsilon_vector[class_index] = max_epsilon

    target_class = np.argmin(epsilon_vector)
    target_class_onehot[target_class] = 1
    target_class_onehot = tf.expand_dims(target_class_onehot, axis=0)

    epsilon_star = np.min(epsilon_vector)

    perturbation = calculate_gradient(model, input_data, target_class_onehot)
    norm_perturbation = perturbation / np.linalg.norm(perturbation)
    perturbation = epsilon_star * norm_perturbation

    adv_input = input_data - perturbation
    Y_adv_signal = model(adv_input)
    Y = model(input_data)

    #single_sample_test(Y, Y_adv_signal, input_label, classes)
    adversary_test(model, Y_adv_signal, x_test, y_test, snrs, mods,
                    test_indices, snr_labels, attack_name= "BiSearch_FGM")


def uap_pca_attack(model, data_points, label_points, x_test, y_test,
                                             snrs, mods, test_indices, snr_labels):
    '''
    https://arxiv.org/abs/1808.07713
    '''
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

    adversary_test(model, extended_UAP, x_test, y_test, snrs, mods,
                    test_indices, snr_labels, attack_name= "UAP_PCA")



if __name__ == "__main__":
    model, epsilons, snrs, x_test, y_test, x_val, y_val, mods,\
                    labels, test_indices = initialize_parameters()
    
    input_signal, input_label = sample_input(x_test, y_test)
    reshaped_input = tf.expand_dims(input_signal, axis=0)
    reshaped_label = tf.expand_dims(input_label, axis=0)

    #White-Box Attack
    #r1 = FGSM(model, epsilons, x_val, y_val, x_test, y_test, snrs, mods, test_indices, labels)
    #bisection_search_FGM(model, reshaped_input, reshaped_label, mods, labels)
    #uap_pca_attack(model, x_val, y_val, x_test, y_test, snrs, mods, test_indices, labels)
    #r2 = pgd_attack(model, x_val, y_val, x_test, y_test, 20, 0.5, snrs, mods, test_indices, labels)
    #print(r2)
    #print(r1)
    #
    
    #Black-Box Attack
    oracle = model_LSTM
    subtitute = model

    weight_path = f'./{oracle_name}_weights/'
    oracle.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    oracle.built = True
    oracle = tf.saved_model.load(weight_path)

    black_box(oracle= oracle, subtitute_model= subtitute, attack= "PGD",
               x_test= x_test, y_test= y_test, snrs= snrs, mods=mods, 
              test_indices= test_indices, snr_labels= labels,
                epsilons= epsilons)