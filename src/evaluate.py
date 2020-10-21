from sklearn import metrics

def evaluation_all(task, y_true, y_pred, threshold=0.5):

    if task == 'regression':
        # Mean Squared Error, MSE 
        mse = metrics.mean_squared_error(y_true, y_pred)
        # Mean Absolute Error, MAE
        mae = metrics.mean_absolute_error(y_true, y_pred)

        results = {'MSE':mse, 'MAE':mae}

    elif task == 'classification':
        y_prob = y_pred.copy()
        y_pred = [1 if p > threshold else 0 for p in y_prob]

        # Accuracy = (TP + FP) / (TP + TN + FP + FN)
        acc = metrics.accuracy_score(y_true, y_pred)
        # Precision = TP / (TP + FP)
        precision = metrics.precision_score(y_true, y_pred)
        # Recall = TP / (TP + FN)
        recall = metrics.recall_score(y_true, y_pred)
        # F1 score = 2 * (Precision * Recall) / (Predicion + Recall)
        f1_score = metrics.f1_score(y_true, y_pred)
        # Area Under Curve of Receive Operating Characteristic
        auc = metrics.roc_auc_score(y_true, y_prob)

        results = {'Acc':acc, 'Precision':precision, 'Recall':recall, 'F1 score':f1_score, 'AUC':auc}

    return results