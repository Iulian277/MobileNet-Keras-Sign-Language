from imports import *
from tensorflow.keras.models import load_model
from data_processing import test_batches

# Load the model
model = load_model('models/mobile_net_sign_language_model2.h5')

# Predictions
predictions = model.predict(x = test_batches, verbose = 0)

# Confusion matrix
test_labels = test_batches.classes
cm = confusion_matrix(y_true = test_labels, y_pred = predictions.argmax(axis = 1))

def plot_confusion_matrix(cm, classes,
                          normalize = False,
                          title = 'Confusion Matrix',
                          cmap = plt.cm.Blues):

    # This function prints and plots the confusion matrix.
    # Normalization can be applied by setting 'normalize = True'.

    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment = "center",
            color = "white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


diag_sum = sum(np.diag(cm))
percentage = diag_sum / float(test_batches.n)
print('Percentage: ' + str(percentage))

cm_plot_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
plot_confusion_matrix(cm = cm, classes = cm_plot_labels)
