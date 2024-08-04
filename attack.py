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
tf.random.set_seed(0)
np.random.seed(0)

attacks = ["FGSM", "PGD", "PCA", "MIM"]
trained_attack_name = "PGD"

models = ["VTCNN", "ResNet18", "LSTM"]
model_name = models[0]
substitute_name = models[1]

epochs = 100
batch_size = 256
loss_func = tf.keras.losses.CategoricalCrossentropy()

model_VTCNN = modelFile.VTCNN(input_shape=(2, 128), num_classes= 11).model   ## VT-CNN
model_LSTM = modelFile.LSTM_AMC(input_shape=(2, 128), num_classes= 11)       ## LSTM
model_ResNet = modelFile.ResNet(input_shape=(2,128), num_layers=18, num_classes= 11)## ResNet18
##########################################################

def initialize_parameters():
    weight_path = f'./weights/{model_name}_weights/'
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
    elif model_name == "ResNet18":
        model = model_ResNet

    model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model = tf.saved_model.load(weight_path)
    
    if substitute_name=="VTCNN": 
        substitute_model = model_VTCNN
    elif substitute_name=="LSTM_AMC":
        substitute_model = model_LSTM
    elif substitute_name == "ResNet18":
        substitute_model = model_ResNet

    substitute_model = tf.saved_model.load(f'./weights/{substitute_name}_weights/')
    epsilons = [1e-7, 1e-6, 1e-5, 0.0001, 0.0005, 0.0008, 0.001, 0.005, 0.008, 0.01, 0.5, 0.8, 1]

    return model, substitute_model, epsilons, snrs, x_test, y_test, x_val, y_val, x_train, y_train,\
        mods, labels, test_indices


def sample_input(inputs, labels):
    random_index = random.randrange(len(inputs))
    return inputs[random_index], labels[random_index]


def calculate_gradient(model, input_signal, input_label, batch_size= 2048):
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
    

def adversary_test(model, adv_data, data, labels, snrs, mods, test_indices, snr_labels):    
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

    name = model_name
    _, acc_mod_snr, bers = utils.evaluate_per_snr(model= model, X_test= adv_data, Y_test= labels,
                                           snrs= snrs, classes= mods, labels= snr_labels,
                                             test_indices= test_indices, model_name= name)
    
    #utils.plot_accuracy_per_snr(snrs= snrs, acc_mod_snr= acc_mod_snr, 
    #                            classes= mods, name= attack_name, model_name= name)

    #utils.tensor_board_plotter(bers.keys(), bers.values())
    #utils.plot_ber_vs_snr(snrs, bers, name= attack_name, model_name= name)
    return acc_mod_snr, bers 


def FGSM(model, epsilons, input_signal, input_label, snrs, classes, test_indices,
          snr_labels, test_acc = True):
    '''
        https://arxiv.org/pdf/1412.6572
    '''
    x,y = input_signal, input_label

    input_signal = tf.convert_to_tensor(input_signal)
    input_label = tf.convert_to_tensor(input_label)
    adv_signal = None
    best_epsilon = None

    for eps in epsilons:
        gradient = calculate_gradient(model, input_signal, input_label)
        perturbation = tf.sign(gradient)
        perturbation = eps*perturbation
        adv_signal = input_signal + perturbation
        Y_adv_signal = predict_in_batches(model, adv_signal)

        thresh_size = 0.2
        max_values = tf.reduce_max(Y_adv_signal, axis=1, keepdims=True)
        mask = tf.equal(Y_adv_signal, max_values)
        Y_adv_signal = tf.cast(mask, dtype=tf.int32)    

        acc = accuracy_score(Y_adv_signal, input_label)

        
        if (acc < thresh_size) or (eps == epsilons[-1]):
            print(f"The best epsilon is: {eps}")
            best_epsilon = eps
            break

    if test_acc:
        _, bers = adversary_test(model= model, adv_data= adv_signal, data= x, labels= y,
                        snrs= snrs, mods= classes, test_indices= test_indices,
                          snr_labels= snr_labels)
        return bers
    
    return adv_signal
        

def PGD(model, data_points, label_points, iters, eps, snrs, mods,
                test_indices, snr_labels, test_acc=True):

    x,y = data_points, label_points

    data_points = tf.convert_to_tensor(data_points)
    label_points = tf.convert_to_tensor(label_points)
    adv_signal = tf.identity(data_points)
    
    for _ in range(iters):
        grad = calculate_gradient(model, adv_signal, label_points)
        perturbation = eps* tf.sign(grad)
        adv_signal = adv_signal + perturbation
        adv_signal = tf.clip_by_value(adv_signal, data_points - eps, data_points + eps)

    if test_acc:
        _, bers = adversary_test(model= model, adv_data= adv_signal, data= x, labels= y,
                        snrs= snrs, mods= mods, test_indices= test_indices,
                          snr_labels= snr_labels)
        return bers
    
    return adv_signal


def MIM(model, input_data, input_label, iters, decay_factor, eps, snrs, mods,
         test_indices, snr_labels, test_acc=True):
    '''
    Momentum Iterative Method
    https://arxiv.org/abs/1710.06081
    '''

    adv_data = tf.identity(input_data)
    g = tf.zeros_like(input_data) # Momentum
    
    for _ in range(iters):
        grad = calculate_gradient(model, adv_data, input_label)
        grad_l1_norm =tf.norm(grad, ord=1, axis=(1,2), keepdims=True)
        normal_grad = grad / (grad_l1_norm + 1e-10)

        g = decay_factor*g + normal_grad

        perturbation = eps * tf.sign(g)
        adv_data = adv_data + perturbation
        adv_data = tf.clip_by_value(adv_data, input_data - eps, input_data + eps)

    if test_acc:
        _, bers = adversary_test(model= model, adv_data= adv_data, data= input_data, labels= input_label,
                     snrs= snrs, mods= mods, test_indices= test_indices,
                       snr_labels= snr_labels)
        return bers
    return adv_data



def bisection_search_FGM(model, input_data, input_label, classes, snr_labels, snrs,
                        mods, test_indices, test_acc=True):
    '''
    https://arxiv.org/abs/1808.07713
    '''
    x,y = input_data, input_label
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

    if test_acc:
        _, bers = adversary_test(model, adv_input, x, y, snrs, mods, test_indices, snr_labels)
        return bers
    return adv_input


def uap_pca_attack(model, data_points, label_points, snrs, mods, test_indices,
                    snr_labels, test_acc= True):
    '''
    https://arxiv.org/abs/1808.07713
    '''
    x,y = data_points, label_points
    
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
    extended_UAP = tf.cast(extended_UAP, tf.float32)
    
    adv_data = extended_UAP + data_points

    if test_acc:
        _, bers = adversary_test(model, adv_data, x, y, snrs, mods, test_indices, snr_labels)
        return bers
    
    return extended_UAP, adv_data


def get_perturbed_data(model, attack_name, original_data, original_label, epsilons):
    if attack_name == "FGSM":
        x = FGSM(model, epsilons, original_data, original_label, None, None, None, None, test_acc= False)
    
    elif attack_name == "PGD":
        x = PGD(model, original_data, original_label, 10, 0.005, None, None, None, None, test_acc= False)
    
    elif attack_name == "PCA":
        _, x= uap_pca_attack(model, original_data, original_label, None, None, None, None, test_acc = False)
            
    elif attack_name == "MIM":
        x = MIM(model, original_data, original_label, 10, 1.0, 0.005, None, None, None, None, test_acc= False)
    
    return x


def white_box_attacks(model, epsilons, x_test, y_test, snrs, mods, test_indices, labels):
    bers = []
    attack_names = []
    
    ber = FGSM(model, epsilons, x_test, y_test, snrs, mods, test_indices, labels)
    bers.append(ber)
    attack_names.append("FGSM")

    ber = PGD(model, x_test, y_test, 10, 0.005, snrs, mods, test_indices, labels)
    bers.append(ber)
    attack_names.append("PGD")
    
    ber = uap_pca_attack(model, x_test, y_test, snrs, mods, test_indices, labels)
    bers.append(ber)
    attack_names.append("PCA")

    ber = MIM(model, x_test, y_test, 10, 1.0, 0.005, snrs, mods, test_indices, labels)
    bers.append(ber)
    attack_names.append("MIM")

    utils.plot_ber_vs_snr(snrs, bers, f'{model_name}', attack_names)
    

def black_box(oracle, substitute_model, x_test, y_test, snrs, mods, 
              test_indices, snr_labels, epsilons, title):
    '''
    Oracle is the model that we want to attack to; and We don't know its parameters or architecture.
    substitute model is the model that we use to create perturbations for dataset (here 
    we use test set) and pass the preturbed data to the oracle for test.

    '''
    bers = []

    for attack in attacks:
        adv_data = get_perturbed_data(substitute_model, attack, x_test, y_test, epsilons)
        _, ber= adversary_test(model= oracle, adv_data= adv_data, data= x_test,
                               labels= y_test, snrs= snrs, mods= mods,
                               test_indices= test_indices, snr_labels= snr_labels)
        bers.append(ber)
    utils.plot_ber_vs_snr(snrs, bers, title, attacks)


def adv_trained_attack(normal_model, substitute_model, epsilons, x_train, y_train,
                    x_val, y_val, x_test, y_test, snrs, mods, test_indices, labels):

    adv_model_name = model_name

    if adv_model_name=="VTCNN": 
        adv_model = model_VTCNN
    elif adv_model_name=="LSTM_AMC":
        adv_model = model_LSTM
    elif adv_model_name == "ResNet18":
        adv_model = model_ResNet

    adv_model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    adv_weight_path = f'./weights/adv_{trained_attack_name}_{adv_model_name}_weights/'
    callbacks = [
        keras.callbacks.ModelCheckpoint(adv_weight_path, monitor='val_loss', verbose= 1, save_best_only= True, mode= 'auto', save_format="tf"),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor= 0.5, verbose= 1, patince= 5, min_lr= 0.000001),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience= 50, verbose= 1, mode= 'auto')
        ]
    
    #########################################################################
    if trained_attack_name == "FGSM" and adv_model_name == "ResNet18":
        best_epsilon = [0.001] # When ResNet18 adversarial trained on FGSM
    elif trained_attack_name == "FGSM" and (adv_model_name == "VTCNN" or adv_model_name == "LSTM"):
        best_epsilon = [0.005] # When VTCNN and LSTM adversarial trained on FGSM
    else:
        best_epsilon = epsilons # When model is not adv trained on FGSM
    #########################################################################

    if os.path.isdir(adv_weight_path):
        adv_model = tf.saved_model.load(adv_weight_path)

    else:
        aug = AdversaryAug.datasetAug(adv_model_name, adv_model, x_train, y_train, x_val, y_val)
        sampled_x_trn, sampled_y_trn, sampled_x_val, sampled_y_val = aug.sampler()
        
        adv_sampled_x_trn = get_perturbed_data(normal_model, trained_attack_name,
                                            sampled_x_trn, sampled_y_trn, best_epsilon)
        adv_sampled_x_val = get_perturbed_data(normal_model, trained_attack_name,
                                            sampled_x_val, sampled_y_val, best_epsilon)

        aug_x_train, aug_y_train, aug_x_val, aug_y_val =\
            aug.augment_dataset(adv_sampled_x_trn, sampled_y_trn, adv_sampled_x_val, sampled_y_val)
        
        aug.adversary_train(batch_size, epochs, aug_x_train, aug_y_train,
                             aug_x_val, aug_y_val, callbacks)
    
    # Evaluate the adversarial trained model    
    black_box(oracle= adv_model, substitute_model= substitute_model,
              x_test= x_test, y_test= y_test, snrs= snrs, mods=mods, 
              test_indices= test_indices, snr_labels= labels, epsilons= best_epsilon,
              title= f'{model_name}_adv_trained_on_{trained_attack_name}__(sub={substitute_name})')

   
def main():
    model, substitute_model, epsilons, snrs, x_test, y_test, x_val, y_val,\
            x_train, y_train, mods, labels, test_indices = initialize_parameters()
    
    #input_signal, input_label = sample_input(x_test, y_test)
    #reshaped_input = tf.expand_dims(input_signal, axis=0)
    #reshaped_label = tf.expand_dims(input_label, axis=0)


    #White-Box Attack
    #white_box_attacks(model, epsilons, x_test, y_test, snrs, mods, test_indices, labels)


    #Black-Box Attack
    #black_box(oracle= model, substitute_model= substitute_model,
    #          x_test= x_test, y_test= y_test, snrs= snrs, mods=mods, 
    #          test_indices= test_indices, snr_labels= labels, epsilons= epsilons,
    #          title= f"BlackBox_attack_on_{model_name}")


    #Attack on Adversarial Trained
    adv_trained_attack(model, substitute_model, epsilons, x_train, y_train, x_val, y_val,
                       x_test, y_test, snrs, mods, test_indices, labels)


if __name__ == "__main__":
    main()