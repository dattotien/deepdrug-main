import numpy as np
import torch
from sklearn.metrics import (
    f1_score, accuracy_score, precision_recall_curve,
    roc_curve, auc, r2_score, explained_variance_score
)
from lifelines.utils import concordance_index
from scipy.stats import pearsonr

t2np = lambda t: t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t

def to_categorical_func(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    return np.reshape(categorical, input_shape + (num_classes,))

def evaluate_binary(y_true, y_pred):
    y_true = t2np(y_true).astype(int)
    y_pred = t2np(y_pred).reshape(-1)
    y_pred_cls = (y_pred >= 0.5).astype(int)

    metric_dict = {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_true_cls': y_true,
        'y_pred_cls': y_pred_cls,
        'F1': f1_score(y_true, y_pred_cls, average='binary'),
        'Acc': accuracy_score(y_true, y_pred_cls),
    }
    metric_dict['prc_prec'], metric_dict['prc_recall'], metric_dict['prc_thres'] = precision_recall_curve(y_true, y_pred)
    metric_dict['roc_tpr'], metric_dict['roc_fpr'], metric_dict['roc_thres'] = roc_curve(y_true, y_pred)
    metric_dict['auROC'] = auc(metric_dict['roc_fpr'], metric_dict['roc_tpr'])
    metric_dict['auPRC'] = auc(metric_dict['prc_recall'], metric_dict['prc_prec'])
    return metric_dict

def evaluate_multiclass(y_true, y_pred, to_categorical=False, num_classes=None):
    y_true = t2np(y_true)
    y_pred = t2np(y_pred)
    if to_categorical:
        y_true_cls = y_true.copy()
        y_true = to_categorical_func(y_true, num_classes)
    else:
        y_true_cls = y_true.argmax(axis=1)

    y_pred_cls = y_pred.argmax(axis=1)

    metric_dict = {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_true_cls': y_true_cls,
        'y_pred_cls': y_pred_cls,
        'F1-macro': f1_score(y_true_cls, y_pred_cls, average='macro'),
        'F1-micro': f1_score(y_true_cls, y_pred_cls, average='micro'),
    }
    metric_dict['F1'] = metric_dict['F1-macro']

    for idx in range(num_classes):
        p, r, t = precision_recall_curve(y_true[:, idx], y_pred[:, idx])
        metric_dict[f'prc_prec@{idx}'] = p
        metric_dict[f'prc_recall@{idx}'] = r
        metric_dict[f'prc_thres@{idx}'] = t

        tpr, fpr, thr = roc_curve(y_true[:, idx], y_pred[:, idx])
        metric_dict[f'roc_tpr@{idx}'] = tpr
        metric_dict[f'roc_fpr@{idx}'] = fpr
        metric_dict[f'roc_thres@{idx}'] = thr

        metric_dict[f'auROC@{idx}'] = auc(fpr, tpr)
        metric_dict[f'auPRC@{idx}'] = auc(r, p)

    metric_dict['auPRC'] = np.nanmean([metric_dict[f'auPRC@{idx}'] for idx in range(num_classes)])
    metric_dict['auROC'] = np.nanmean([metric_dict[f'auROC@{idx}'] for idx in range(num_classes)])
    return metric_dict

def evaluate_multilabel(y_true, y_pred, thers=0.5):
    y_true = t2np(y_true).astype(int)
    y_pred = t2np(y_pred)
    y_pred_cls = (y_pred >= thers).astype(int)
    num_classes = y_true.shape[-1]

    metric_dict = {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_true_cls': y_true,
        'y_pred_cls': y_pred_cls,
        'F1-macro-old': f1_score(y_true, y_pred_cls, average='macro'),
        'F1-micro-old': f1_score(y_true.ravel(), y_pred_cls.ravel(), average='micro')
    }

    for idx in range(num_classes):
        metric_dict[f'F1@{idx}'] = f1_score(y_true[:, idx], y_pred_cls[:, idx])
        p, r, t = precision_recall_curve(y_true[:, idx], y_pred[:, idx])
        metric_dict[f'prc_prec@{idx}'] = p
        metric_dict[f'prc_recall@{idx}'] = r
        metric_dict[f'prc_thres@{idx}'] = t
        tpr, fpr, thr = roc_curve(y_true[:, idx], y_pred[:, idx])
        metric_dict[f'roc_tpr@{idx}'] = tpr
        metric_dict[f'roc_fpr@{idx}'] = fpr
        metric_dict[f'roc_thres@{idx}'] = thr
        metric_dict[f'auROC@{idx}'] = auc(fpr, tpr)
        metric_dict[f'auPRC@{idx}'] = auc(r, p)

    metric_dict['auPRC'] = np.nanmean([metric_dict[f'auPRC@{idx}'] for idx in range(num_classes)])
    metric_dict['auROC'] = np.nanmean([metric_dict[f'auROC@{idx}'] for idx in range(num_classes)])
    metric_dict['F1-macro'] = np.nanmean([metric_dict[f'F1@{idx}'] for idx in range(num_classes)])
    metric_dict['F1'] = metric_dict['F1-macro']
    return metric_dict

def evaluate_regression(y_true, y_pred):
    y_true = t2np(y_true).reshape(-1)
    y_pred = t2np(y_pred).reshape(-1)
    pr, pr_p_val = pearsonr(y_true, y_pred)

    metric_dict = {
        'y_true': y_true,
        'y_pred': y_pred,
        'r2': r2_score(y_true, y_pred),
        'mse': np.mean((y_true - y_pred) ** 2),
        'pearsonr': pr,
        'pearsonr_p_val': pr_p_val,
        'concordance_index': concordance_index(y_true, y_pred),
        'explained_variance': explained_variance_score(y_true, y_pred),
        'cindex': -1,
    }
    # Optional: advanced metric rm2
    def r_squared_error(y_obs, y_pred):  # helper
        y_obs_mean = np.mean(y_obs)
        y_pred_mean = np.mean(y_pred)
        mult = np.sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean)) ** 2
        y_obs_sq = np.sum((y_obs - y_obs_mean) ** 2)
        y_pred_sq = np.sum((y_pred - y_pred_mean) ** 2)
        return mult / float(y_obs_sq * y_pred_sq)
    def get_rm2(ys_orig, ys_line):
        r2 = r_squared_error(ys_orig, ys_line)
        r02 = 1 - np.sum((ys_orig - ys_line * np.sum(ys_orig*ys_line)/np.sum(ys_line**2))**2) / np.sum((ys_orig - np.mean(ys_orig))**2)
        return r2 * (1 - np.sqrt(np.abs(r2**2 - r02**2)))
    metric_dict['rm2'] = get_rm2(y_true, y_pred)
    return metric_dict
