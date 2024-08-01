import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import json
import datetime
from tensorflow.keras.callbacks import TensorBoard

def total_plotter(history, model_name):
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(f'figure/{model_name}_Model_Loss.png')

    plt.plot(history['accuracy'], label='accuracy')
    plt.plot(history['val_accuracy'], label='val_accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(f'figure/{model_name}_Model_Accuracy.png')

    plt.close()
    

def save_results(acc, acc_mod_snr, bers, model_name):
    # Save accuracy for each modulation type per SNR
    with open(f'{model_name}_acc_mod_snr.json', 'w') as f:
        json.dump({"model": model_name, "acc_mod_snr": acc_mod_snr.tolist()}, f)

    # Save overall accuracy per SNR
    with open(f'{model_name}_acc.json', 'w') as f:
        json.dump({"model": model_name, "acc": acc}, f)

    # Save overall accuracy per SNR
    with open(f'{model_name}_bers.json', 'w') as f:
        json.dump({"model": model_name, "bers": bers}, f)


def evaluate_per_snr(model, X_test, Y_test, snrs, classes, labels, test_indices, model_name):
    acc = {}
    acc_mod_snr = np.zeros((len(classes), len(snrs)))
    bers = {}

    for i, snr in enumerate(snrs):
        snr_test = [labels[idx][1] for idx in test_indices]
        test_X_i = X_test[np.where(np.array(snr_test) == snr)]
        test_Y_i = Y_test[np.where(np.array(snr_test) == snr)]
        
        test_Y_i_hat = model(test_X_i)

        test_Y_i_hat = np.argmax(test_Y_i_hat, axis=1) #Predictions
        test_Y_i = np.argmax(test_Y_i, axis=1) #Labels

        bers[snr] = calculate_ber(y= test_Y_i_hat, labels= test_Y_i)
        cm = confusion_matrix(test_Y_i, test_Y_i_hat)

        plot_confusion_matrix(cm= cm, classes= classes, 
                            title= f"{model_name} Confusion matrix of {snr}db SNR",
                            save_filename= f"figure/{model_name}_Confusion_matrix_{snr}db_SNR.png")
        
        # Accuracy of current signal-to-noise ratio: sum of the correctly classified modulations(trace of cm)
        #  divided by the sum of all of all classification scores (sum of all elements of cm) 
        acc[snr] = np.trace(cm) / np.sum(cm) 

        # Accuracy of each Modulation
        acc_mod_snr[:, i] = np.diag(cm) / np.sum(cm, axis=1)

    return acc, acc_mod_snr, bers


def plot_accuracy_per_snr(snrs, acc_mod_snr, classes, model_name, name="No_Attack", display_num=11):
    num_classes = len(classes)
    num_plots = int(np.ceil(num_classes / display_num))  # Calculate number of plots needed

    for g in range(num_plots):
        beg_index = g * display_num
        end_index = np.min([(g + 1) * display_num, num_classes])
    
        plt.figure(figsize=(12, 10))
        plt.xlabel("Signal to Noise Ratio")
        plt.ylabel("Classification Accuracy")
        plt.title(f"Classification Accuracy per Modulation Type ({name})")

        for i in range(beg_index, end_index):
            plt.plot(snrs, acc_mod_snr[i], label=classes[i])
        
        plt.legend()
        plt.grid(True)
        plt.savefig(f'figure/{model_name}_acc_per_snr_{name}.png')
        plt.close()


def plot_confusion_matrix(cm, classes, title, save_filename):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.savefig(save_filename)
    plt.close()


def calculate_ber(y, labels):
    num_bits_in_error = np.sum(y != labels)
    total_bits = len(labels)

    ber = num_bits_in_error / total_bits
    return ber


def tensor_board_plotter(snrs, bers, name='ber'):
    log_dir = "logs/ber_test/"
    writer = tf.summary.create_file_writer(log_dir)
        
    with writer.as_default():
        for snr, ber in zip(snrs, bers):
            print(snr, ber)
            tf.summary.scalar(name, ber, step=snr)


def log_image_to_tensorboard(filename, identifier):
    log_dir = "logs/metric/"
    # Read the image file
    image = tf.io.read_file(filename)
    image = tf.image.decode_image(image, channels=3)
    
    # Ensure image is in uint8 format
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    
    # Create a file writer
    with tf.summary.create_file_writer(log_dir).as_default():
        with tf.summary.record_if(True):
            tf.summary.image(f'Composite Plot {identifier}', [image], step=identifier)


def plot_ber_vs_snr(snrs, bers, model_name, attack_name=["No_Attack"]):
    plt.figure(figsize=(10, 6))
    for i, ber_list in enumerate(bers):
        plt.plot(snrs, list(ber_list.values()), label= attack_name[i], marker='o', linestyle='-', color=f'C{i}')
    
    plt.ylim(0.0, 1.0)
    plt.xlabel("Signal to Noise Ratio (SNR)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.grid(True)
    plt.legend()
    plt.title(f"BER vs. SNR ({model_name})")
    #plt.tight_layout()
    filename= f"logs/ber_vs_snr_{model_name}.png"
    plt.savefig(filename)
    plt.close()