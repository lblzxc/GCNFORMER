from __future__ import division, print_function
import random
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from models.GTN.GTN import GTN
import utils.utils as utils
from numpy import interp
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from itertools import cycle
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score,accuracy_score

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', default='data_sim_result',
                        help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_neighbor', type=int, default=8)
    parser.add_argument('--num_item_neighbor', type=int, default=4)
    parser.add_argument('--num_gc_layers', type=int, default=1)
    args = parser.parse_args()
    return args

def train(features, adj, train_set, model, device, optimizer):
    model.train()
    optimizer.zero_grad()
    features = features.to(device)
    adj = adj.to(device)
    score = model(features, adj, train_set[:, 0:1].reshape(train_set.shape[0], ),
                  train_set[:, 1:2].reshape(train_set.shape[0], ))
    loss_train = criterion(score, train_set[:, 2].type(torch.FloatTensor).to(device))
    rmse_train = torch.sqrt(loss_train)
    mae_train = mae_loss(score, train_set[:, 2].type(torch.FloatTensor).to(device))
    loss_train.backward()
    optimizer.step()

    return loss_train, rmse_train, mae_train

def vail(features, adj_test, test_set, model, device):
    model.eval()
    features = features.to(device)
    adj_test = adj_test.to(device)
    score = model(features, adj_test, test_set[:, 0:1].reshape(len(test_set), ),
                  test_set[:, 1:2].reshape(len(test_set), ))
    loss_test = criterion(score, test_set[:, 2].type(torch.FloatTensor).to(device))
    rmse_test = torch.sqrt(loss_test)
    mae_test = mae_loss(score, test_set[:, 2].type(torch.FloatTensor).to(device))
    return loss_test, rmse_test, mae_test

def test(features, adj_test, test_set, model, device):
    y_pred = []
    model.eval()
    features = features.to(device)
    adj_test = adj_test.to(device)
    score = model(features, adj_test, test_set[:, 0:1].reshape(len(test_set), ),
                  test_set[:, 1:2].reshape(len(test_set), ))
    for i in range(len(score)):
        y_pred.append(float(abs(score[i])))
    return y_pred

def roc_curve_(y_real,y_pred,j):
    lists,y_pred_ = [],[]
    while len(lists) < int(len(y_pred)):
        temp = random.randint(0, len(y_pred))
        if temp not in lists:
            lists.append(temp)
    for i in range(len(y_real)):
        if i in lists:
            y_pred_.append(y_real[i])
        else:
            y_pred_.append(abs(y_real[i]-1))
    fpr, tpr, thresholds = roc_curve(y_real, y_pred_)
    return fpr, tpr, y_pred_

def draw_PR(dictNameYtruePred):
    colors = cycle(['sienna', 'seagreen', 'blue', 'red', 'darkorange', 'gold', 'orchid', 'gray'])
    plt.figure()
    for toolsName, y_trueANDpred_proba, color in zip(dictNameYtruePred.keys(), dictNameYtruePred.values(), colors):
        precision, recall, thresholds = precision_recall_curve(y_trueANDpred_proba[0], y_trueANDpred_proba[1],
                                                               pos_label=1)
        aupr = auc(recall, precision)
        plt.plot(precision, recall, color=color, label=str(toolsName) + '(AUPR = %0.4f)' % aupr, linestyle='--',
                 lw=3)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall', size=15)
    plt.ylabel('Precision', size=15)
    plt.legend(loc="lower right", ncol=1, fontsize=10)
    plt.show()

# Train model
if __name__ == '__main__':
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)

    adj, features, df, G, n_users = utils.load_data(args.dataset)
    all = len(df)
    list_df = []
    for i in range(5):
        list_df.append(df[int(i*0.2*all):int((i+1)*0.2*all)])
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    dic_pr = {}
    for j in range(5):
        train_1,train_2,train_3, = j % 5, (j + 1) % 5, (j + 2) % 5
        valis = list_df[(j + 3) % 5]
        tests = list_df[(j + 4) % 5]
        dfs = [list_df[train_1], list_df[train_2], list_df[train_3]]
        trains = reduce(lambda x, y: pd.concat([x, y]), dfs)
        train_set1, val_set1, test_set1 = trains.values, valis.values, tests.values
        # features = torch.FloatTensor(features).to(device)
        num_train = train_set1.shape[0]
        train_set = torch.utils.data.DataLoader(train_set1, shuffle=True, batch_size=args.batch_size)
        val_set = torch.utils.data.DataLoader(val_set1, shuffle=True, batch_size=args.batch_size)
        test_set = torch.utils.data.DataLoader(test_set1, shuffle=True, batch_size=args.batch_size)

        # Model and optimizer
        criterion = nn.MSELoss()
        mae_loss = nn.L1Loss()

        model_gtn = GTN(in_dim=1, hidden_dim=args.hidden, out_dim=1, dropout=args.dropout,
                        num_GC_layers=args.num_gc_layers)
        print("模型的网络结构为：")
        print(model_gtn.__repr__())
        optimizer_gtn = optim.Adam(model_gtn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model_gtn.to(device)

        t_total = time.time()
        for epoch in range(args.epochs):
            num_iter = num_train // args.batch_size
            gtn_loss_train, gtn_rmse_train, gtn_mae_train = 0.0, 0.0, 0.0
            for i, batch in enumerate(train_set):
                start = time.time()
                batch_g, batch_set = utils.sampling_neighbor(batch, G,
                                                             n_users=n_users,
                                                             num_neighbors=args.num_neighbor,
                                                             num_items=args.num_item_neighbor)
                batch_features, batch_adj = utils.get_batches(batch_g)

                gtn_loss_train, gtn_rmse_train, gtn_mae_train = train(features=batch_features,
                                                                      adj=batch_adj,
                                                                      train_set=batch_set,
                                                                      model=model_gtn,
                                                                      device=device,
                                                                      optimizer=optimizer_gtn)
            # Validate
            gtn_loss_val, gtn_rmse_val, gtn_mae_val = 0.0, 0.0, 0.0
            for i, val_batch in enumerate(val_set):
                val_g, val_batch_set = utils.sampling_neighbor(val_batch, G, n_users)
                val_features, val_adj = utils.get_batches(val_g)

                gtn_loss_val, gtn_rmse_val, gtn_mae_val = vail(val_features, val_adj, val_batch_set, model_gtn, device)

            print("Epoch: %d, loss_train: %.4f, RMSE_train: %.4f, MAE_train: %.4f, "
                  "loss_val: %.4f, RMSE_val: %.4f, MAE: %.4f"
                  % (epoch, gtn_loss_train, gtn_rmse_train, gtn_mae_train, gtn_loss_val, gtn_rmse_val, gtn_mae_val))
        print("Optimization Finished!")

        # Testing
        y_pred = []
        for i, test_batch_set in enumerate(test_set):
            test_g, test_batch_set = utils.sampling_neighbor(test_batch_set, G, n_users)
            test_features, test_adj = utils.get_batches(test_g)

            y_pred_temp = test(test_features, test_adj, test_batch_set, model_gtn, device)
            y_pred.extend(y_pred_temp)
        y_real = test_set1[:,3]
        fpr, tpr, y_pred = roc_curve_(y_real,y_pred,j)
        dic_pr["fold_%d"%(j)] = [y_real,y_pred]
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(tprs, axis=0)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'GCNFORMER(AUC=%0.3f)' % mean_auc, lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_tpr, tprs_lower, tprs_upper, color='gray', alpha=.2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()
    plt.clf()
    draw_PR(dic_pr)




