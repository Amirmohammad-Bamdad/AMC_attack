import tensorflow as tf
import modelFile
import loader
import random
import numpy as np
import utils
import os
import AdversaryAug
import keras
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from modelFile import predict_in_batches

##########################################################
model_name = "LSTM_AMC"
oracle_name = "VTCNN"
epochs = 100
batch_size = 256
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
    x_train, y_train = data.train_data[0], data.train_data[1] # For adv_training 
    
    if model_name=="VTCNN": 
        model = model_VTCNN
    elif model_name=="LSTM_AMC":
        model = model_LSTM

    model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model = tf.saved_model.load(weight_path)

    epsilons = [0.0001, 0.0005, 0.0008, 0.001, 0.005, 0.008, 0.01, 0.5, 0.8, 1]

    return model, epsilons, snrs, x_test, y_test, x_val, y_val, x_train, y_train,\
        mods, labels, test_indices


def sample_input(inputs, labels):
    random_index = random.randrange(len(inputs))
    return inputs[random_index], labels[random_index]


def calculate_gradient(model, input_signal, input_label, batch_size= 1024):
    dataset = tf.data.Dataset.from_tensor_slices((input_signal, input_label))
    dataset = dataset.batch(batch_size)
    gradients = []
    for batch_input_signal, batch_input_label in dataset: 
        with tf.GradientTape() as tape:
            tape.watch(batch_input_signal)
            y = model(batch_input_signal, training=False)
            loss = loss_func(batch_input_label, y)

        gradient = tape.gradient(loss, batch_input_signal)
        gradients.append(gradient)

    return tf.concat(gradients, axis=0)


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



def adversary_test(model, adv_data, data, labels, snrs, mods,
                    test_indices, snr_labels, attack_name, is_blackbox= False):    
    #Normal
    y_test_hat = predict_in_batches(model, data)
    
    max_values = tf.reduce_max(y_test_hat, axis=1, keepdims=True)
    mask = tf.equal(y_test_hat, max_values)
    y_test_hat = tf.cast(mask, dtype=tf.int32) 

    test_loss = loss_func(y_test_hat, labels)
    test_acc = accuracy_score(tf.math.argmax(labels, axis=1), tf.math.argmax(y_test_hat, axis=1))

    
    #Adversarial
    adv_y_test_hat = predict_in_batches(model, adv_data)
 
    adv_max_values = tf.reduce_max(adv_y_test_hat, axis=1, keepdims=True)
    adv_mask = tf.equal(adv_y_test_hat, adv_max_values)
    adv_y_test_hat = tf.cast(adv_mask, dtype=tf.int32) 

    adv_test_loss = loss_func(adv_y_test_hat, labels)
    adv_test_acc = accuracy_score(tf.math.argmax(labels, axis=1), tf.math.argmax(adv_y_test_hat, axis=1))


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
          snrs, classes, test_indices, snr_labels, test_acc = True, is_adv_trained= False):
    '''
        https://arxiv.org/pdf/1412.6572
    '''
    input_signal = tf.convert_to_tensor(input_signal)
    input_label = tf.convert_to_tensor(input_label)
    adv_signal = None
    
    for eps in epsilons:
        gradient = calculate_gradient(model, input_signal, input_label)
        perturbation = tf.sign(gradient)
        perturbation = eps*perturbation
        adv_signal = input_signal + perturbation
        Y_adv_signal = predict_in_batches(model, adv_signal)

        thresh_size = 0.15

        max_values = tf.reduce_max(Y_adv_signal, axis=1, keepdims=True)
        mask = tf.equal(Y_adv_signal, max_values)
        Y_adv_signal = tf.cast(mask, dtype=tf.int32)    

        acc = accuracy_score(Y_adv_signal, input_label)

        
        if (acc < thresh_size) or (eps == epsilons[-1]):
            print(f"The best epsilon is: {eps}")
            break

    if is_adv_trained:
        name = "FGSM_on_adversarially_trained_model"
    else:
        name = 'FGSM'

    if test_acc:
        adversary_test(model= model, adv_data= adv_signal, data= x_test, labels= y_test,
                        snrs= snrs, mods= classes, test_indices= test_indices,
                          snr_labels= snr_labels, attack_name= name)
    return adv_signal
        

def pgd_attack(model, data_points, label_points, x_test, y_test, iters, eps,
                            snrs, mods, test_indices, snr_labels, test_acc=True):

    data_points = tf.convert_to_tensor(data_points)
    label_points = tf.convert_to_tensor(label_points)
    adv_signal = tf.identity(data_points)
    
    for _ in range(iters):
        grad = calculate_gradient(model, adv_signal, label_points)
        perturbation = eps* tf.sign(grad)
        adv_signal = adv_signal + perturbation
        adv_signal = tf.clip_by_value(adv_signal, data_points - eps, data_points + eps)

    if test_acc:
        adversary_test(model= model, adv_data= adv_signal, data= x_test, labels= y_test,
                        snrs= snrs, mods= mods, test_indices= test_indices,
                          snr_labels= snr_labels, attack_name= "PGD")
    return adv_signal



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

            adv_label = predict_in_batches(model, adv_x)
            
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
    Y_adv_signal = predict_in_batches(model, adv_input)
    Y = predict_in_batches(model, input_data)

    #single_sample_test(Y, Y_adv_signal, input_label, classes)
    adversary_test(model, Y_adv_signal, x_test, y_test, snrs, mods,
                    test_indices, snr_labels, attack_name= "BiSearch_FGM")


def uap_pca_attack(model, data_points, label_points, x_test, y_test,
                        snrs, mods, test_indices, snr_labels, test_acc= True):
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

    if test_acc:
        adversary_test(model, extended_UAP, x_test, y_test, snrs, mods,
                        test_indices, snr_labels, attack_name= "UAP_PCA")
    
    return extended_UAP



if __name__ == "__main__":
    model, epsilons, snrs, x_test, y_test, x_val, y_val, x_train, y_train, mods,\
                    labels, test_indices = initialize_parameters()
    
    input_signal, input_label = sample_input(x_test, y_test)
    reshaped_input = tf.expand_dims(input_signal, axis=0)
    reshaped_label = tf.expand_dims(input_label, axis=0)

    #White-Box Attack
    #FGSM(model, epsilons, x_val, y_val, x_test, y_test,\
    #           snrs, mods, test_indices, labels, test_acc=False)
    
    #bisection_search_FGM(model, reshaped_input, reshaped_label, mods, labels)
    #uap_pca_attack(model, x_val, y_val, x_test, y_test,\
    #  snrs, mods, test_indices, labels)

    #pgd_attack(model, x_val, y_val, x_test, y_test, 10, 0.005, snrs, mods, test_indices, labels)

    
    #Black-Box Attack
    #oracle = model_LSTM
    #subtitute = model
#
    #weight_path = f'./{oracle_name}_weights/'
    #oracle.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #oracle.built = True
    #oracle = tf.saved_model.load(weight_path)
#
    #black_box(oracle= oracle, subtitute_model= subtitute, attack= "PGD",
    #           x_test= x_test, y_test= y_test, snrs= snrs, mods=mods, 
    #          test_indices= test_indices, snr_labels= labels,
    #            epsilons= epsilons)


    #Attack on Adversarial Trained
    attack_name = "FGSM"
    if model_name=="VTCNN": 
        adv_model = model_VTCNN
    elif model_name=="LSTM_AMC":
        adv_model = model_LSTM
    
    adv_model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    adv_weight_path = f'./adv_{attack_name}_{model_name}_weights/'
    callbacks = [
        keras.callbacks.ModelCheckpoint(adv_weight_path, monitor='val_loss', verbose= 1, save_best_only= True, mode= 'auto', save_format="tf"),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor= 0.5, verbose= 1, patince= 5, min_lr= 0.000001),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience= 50, verbose= 1, mode= 'auto')
        ]
    
    if os.path.isdir(adv_weight_path):
        adv_model = tf.saved_model.load(adv_weight_path)
    
    else:
        aug = AdversaryAug.datasetAug(model_name, adv_model, x_train, y_train, x_val, y_val)
        x1, y1, x2, y2 = aug.sampler()
        
        x1 = FGSM(model, epsilons, x1, y1, None, None,\
                    None, None, None, None, test_acc= False)
        
        #x1 = pgd_attack(model, x1, y1, None, None, 10, 0.0005,\
        #                 None, None, None, None, test_acc= False)
        
        #r= uap_pca_attack(model, x1, y1, None, None,\
        #    None, None, None, None, test_acc = False)
        #x1 = tf.cast((r+x1), tf.float32)
        
        x2 = FGSM(model, epsilons, x2, y2, None, None,\
                    None, None, None, None, test_acc= False)
        
        #x2 = pgd_attack(model, x2, y2, None, None, 10, 0.0005,\
        #                 None, None, None, None, test_acc= False)
        
        #r= uap_pca_attack(model, x2, y2, None, None,\
        #    None, None, None, None, test_acc = False)
        #x2 = tf.cast((r+x2), tf.float32)

        aug_x_train, aug_y_train, aug_x_val, aug_y_val =\
              aug.augment_dataset(x1,y1,x2,y2)
        
        aug.adversary_train(batch_size, epochs, aug_x_train, aug_y_train,
                             aug_x_val, aug_y_val, callbacks)
        
    adv_data = FGSM(model, [0.005], x_test, y_test, None, None,\
               None, None, None, None, test_acc= False)
    
    #adv_data = pgd_attack(model, x_test, y_test, None, None, 10, 0.005,\
    #                 None, None, None, None, test_acc= False)
    
    #r= uap_pca_attack(model, x_val, y_val, None, None,\
    #        None, None, None, None, test_acc = False)
    #adv_data = tf.cast((r+x_test), tf.float32)
    
    adversary_test(model= adv_model, adv_data= adv_data, data= x_test, labels= y_test,
                snrs= snrs, mods= mods, test_indices= test_indices,
                  snr_labels= labels, attack_name= "FGSM_on_adversarially_trained_model_with_FGSM")