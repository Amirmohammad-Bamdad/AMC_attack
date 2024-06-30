import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import json

def total_plotter(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


def plot_accuracy_curve(snrs, acc):
    plt.plot(snrs, list(acc.values()))
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title("Classification Accuracy on RadioML 2016.10 Alpha")
    plt.tight_layout()
    plt.savefig('figure/each_acc.png')
    plt.close()


def save_results(acc, acc_mod_snr, model_name, dropout_rate):
    # Save accuracy for each modulation type per SNR
    with open('predictresult/acc_for_mod_on_{}.json'.format(model_name), 'w') as f:
        json.dump({"model": model_name, "dropout_rate": dropout_rate, "acc_mod_snr": acc_mod_snr.tolist()}, f)

    # Save overall accuracy per SNR
    with open('predictresult/{}_d{}.json'.format(model_name, dropout_rate), 'w') as f:
        json.dump({"model": model_name, "dropout_rate": dropout_rate, "acc": acc}, f)


def evaluate_per_snr(model, X_test, Y_test, snrs, classes):
    acc = {}
    acc_mod_snr = np.zeros((len(classes), len(snrs)))
    
    for i, snr in enumerate(snrs):
        test_X_i = X_test[np.where(np.array(snrs) == snr)]
        test_Y_i = Y_test[np.where(np.array(snrs) == snr)]

        test_Y_i_hat = model.predict(test_X_i)
        cm = confusion_matrix(test_Y_i, test_Y_i_hat)
        acc[snr] = 1.0 * np.trace(cm) / np.sum(cm) 

        acc_mod_snr[:, i] = np.diag(cm) / np.sum(cm, axis=1)

    return acc, acc_mod_snr


def plot_accuracy_per_snr(snrs, acc_mod_snr, classes, dis_num):
    num_classes = len(classes)
    num_plots = int(np.ceil(num_classes / dis_num))  # Calculate number of plots needed

    for g in range(num_plots):
        beg_index = g * dis_num
        end_index = np.min([(g + 1) * dis_num, num_classes])
    
        plt.figure(figsize=(12, 10))
        plt.xlabel("Signal to Noise Ratio")
        plt.ylabel("Classification Accuracy")
        plt.title("Classification Accuracy per Modulation Type")

        for i in range(beg_index, end_index):
            plt.plot(snrs, acc_mod_snr[i], label=classes[i])
        
        for x, y in zip(snrs, acc_mod_snr[i]):
            plt.text(x, y, f"{y:.2f}", ha='center', va='bottom', fontsize=8)  # Format accuracy to 2 decimals
    
        plt.legend()
        plt.grid(True)
        plt.savefig(f'figure/acc_with_mod_{g+1}.png')  # Save plot with unique filename
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