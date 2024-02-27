import torch
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def make_confusion_matrix(y_true, y_pred, ckpt_dir, epoch):
    cm = confusion_matrix(y_true, y_pred)

    classes = ['Negative', 'Positive']
    
    plt.figure(figsize=(6, 4))

    cm = sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=classes, yticklabels=classes)
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    
    plt.savefig(os.path.join(ckpt_dir, f'confusion_matrix_{epoch}.png'))
    
    print(f'Saving the Confusion Matrix : {ckpt_dir}')

def make_metric(y_true, y_pred):
    
    acc = round(accuracy_score(y_true, y_pred), 3)
    precision = round(precision_score(y_true, y_pred), 3)
    recall = round(recall_score(y_true, y_pred), 3)
    f1 = round(f1_score(y_true, y_pred), 3)
    
    return acc, precision, recall, f1

    