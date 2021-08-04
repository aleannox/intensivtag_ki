import os
import warnings

import tensorflow
import gensim
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Shut up TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)

# Shut up FastText
gensim.models.keyedvectors.logger.disabled = True


def plot_ROC(f_name, true, predicted):
    fpr, tpr, thresholds = roc_curve(true, predicted)
    auroc = auc(fpr, tpr)
    accuracy_ratio = auroc * 2 - 1

    plt.title(f_name)
    plt.plot(fpr, tpr, 'b',
             label=f'AR = {(accuracy_ratio * 100):0.1f}, '
                   f'AUC = {(auroc * 100):0.1f}'
             )
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    return accuracy_ratio
