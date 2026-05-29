import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer

#loading the data
#load the set as a dataframe
data_df = load_breast_cancer(as_frame=True)
# chack what are the classes
print("Classes:", data_df.target_names)
df = data_df.frame

features = df.drop(columns=["target"]).columns.tolist()
target = "target"
df = df[features + [target]].dropna()
print(df.head())
'''
print("DataFrame Info:")
print(df.info())
print("-" * 50)
print("First 5 rows of the DataFrame:")

print("Describe")
print(df.describe())
'''
X = df[features].to_numpy(dtype=float)   # (N, d)
y = df[target].to_numpy(dtype=float)     # (N,)
np.set_printoptions(suppress=True, precision=4)

print(X, y)
print("-" * 50)
print("Data Shape:")
print(X.shape, y.shape)
print("-" * 50)
print ("Feature Names:")
print(features)
print("Class Distribution:")
print(df[target].value_counts())
print("Checking for missing values:")
print(df.isnull().sum())
print("Summary Statistics:")
print(df.describe())


#boxplot 
import matplotlib.pyplot as plt
import seaborn as sns




sns.boxplot(data=df, x=target, y="mean radius", hue="target", palette="Set2", legend=False)
plt.title("Mean Radius by Target Class", color= "blue")
plt.suptitle("")
plt.xlabel("Diagnosis", color= "blue")
plt.ylabel("Mean Radius", color= "blue")
plt.xticks([0, 1], ["malignant", "benign"], color= "firebrick")
plt.yticks(color= "firebrick")
plt.show()



# correlation heatmap
correlation_matrix = df.corr()
plt.figure(figsize=(8, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='BrBG', fmt=".2f", linewidth=0.5) 
plt.title("Correlation Heatmap", color= "Salmon", size=18)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



# task 2
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42, stratify=y)


#  Verifying that the class distribution is similar in train and test sets

train_set = pd.Series(y_train).value_counts(normalize=True)
test_set = pd.Series(y_test).value_counts(normalize=True)

print("Train set class distribution: ", train_set)
print("Test_ Set class distribution: ", test_set)

#2.1 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# fitting is only done on train
# this will prevent Data leakage
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test) 

#3.1
from sklearn.base import BaseEstimator, ClassifierMixin
#implemetnting Knn from scratch 
class KNNClessifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, n_neighbors: int = 5, metric: str ="euclidean", weights: str = "uniform"):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights

    #___fit___________________________

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.classes_ = np.unique(y)
        return self

    #__________probability_________

    def predict_proba(self, X: np.ndarray) -> np.ndarray: 
        probs = []
        for x in X:
            dists = self._compute_distances(x)
            nn_idx = np.argsort(dists)[:self.n_neighbors]
            nn_vals = self.y_train[nn_idx]

            probs_p = np.mean(nn_vals == 1)
            probs.append([1 - probs_p, probs_p])

        return np.array(probs)
    
    #______distance_________________

    def _compute_distances(self, x: np.ndarray) -> np.ndarray:
        diff = self.X_train - x
        if self.metric == "euclidean":
            return np.sqrt((diff ** 2).sum(axis=1))
        elif self.metric == "manhattan":
            return np.abs(diff).sum(axis=1)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    # ── predict single point ──────────────────
    def _predict_one(self, x: np.ndarray) -> float:
        dists   = self._compute_distances(x)
        nn_idx  = np.argsort(dists)[: self.n_neighbors]
        nn_dists = dists[nn_idx]
        nn_vals  = self.y_train[nn_idx]

        if self.weights == "uniform":
            classes, counts = np.unique(nn_vals, return_counts=True)
            return classes[np.argmax(counts)]
        elif self.weights == "distance":
            # avoid division by zero for exact matches
            eps = 1e-10
            w   = 1.0 / (nn_dists + eps)
            unq_classes = np.unique(nn_vals)
            class_w_sum = [w[nn_vals == c].sum() for c in unq_classes]
            return unq_classes[np.argmax(class_w_sum)]
        else:
            raise ValueError(f"Unknown weights: {self.weights}")

    # ── predict ───────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(x) for x in X])
    

Baseline_KNN = KNNClessifier()        # already init to 5
Baseline_KNN.fit(X_train_scaled, y_train)


test_pred = Baseline_KNN.predict(X_test_scaled)

accr = (test_pred == y_test).mean()
print(f"Test Accuracy with k = 5: {accr:.4f}")

#3.2 

model_euc = KNNClessifier()       # already init to euclidean 
model_euc.fit(X_train_scaled, y_train)
euc_pred = model_euc.predict(X_test_scaled)
euc_pred_rounded = np.round(euc_pred)

model_manhattan = KNNClessifier(metric="manhattan")
model_manhattan.fit(X_train_scaled, y_train)
manhattan_pred = model_manhattan.predict(X_test_scaled)
manhattan_pred_rounded = np.round(manhattan_pred)

# this time use library for acuracy 
print("Test Accuracy with Euclidean: ", np.mean(euc_pred_rounded == y_test))
print("Test Accuracy with Manhattan: ", np.mean(manhattan_pred_rounded == y_test))

#4.1
from sklearn.model_selection import cross_val_score
k_range = range(1, 31)
cv_scores = []
train_acc_s = []
for k in k_range:
    knn = KNNClessifier(n_neighbors = k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

    #train score
    knn.fit(X_train_scaled, y_train)
    train_prediction = knn.predict(X_train_scaled)
    train_acc = np.mean(train_prediction == y_train)
    train_acc_s.append(train_acc)    


plt.figure(figsize=(10, 5))
plt.plot(k_range, cv_scores, color = '#580F41')
plt.title("k (x-axis) vs. mean CV accuracy (y-axis)", color = '#069AF3')
plt.xlabel("k", color = '#FF00FF')
plt.ylabel("Mean Accuracy", color = '#FF00FF')
plt.xticks(color='#FC5A50')
plt.yticks(color='#FC5A50')
plt.grid(True)
plt.show()

best_k = k_range[np.argmax(cv_scores)]
print(f"Optimal k: {best_k}")

#5.1

def confusion_matrix(y_true, y_pred):
    TP = TN = FP = FN = 0
    for actual_val, prediction in zip(y_true, y_pred):
        if actual_val == 1 and prediction == 1:
            TP += 1
        elif actual_val == 0 and prediction == 0: 
            TN += 1
        elif actual_val == 0 and prediction == 1:
            FP  += 1
        elif actual_val == 1 and prediction == 0:
            FN += 1
    return np.array([[TP, FP], [FN, TN]])


best_k = KNNClessifier(n_neighbors = 8, metric="manhattan")
best_k.fit(X_train_scaled, y_train)

best_k_pred = best_k.predict(X_test_scaled)


result = confusion_matrix(y_test,  best_k_pred)

precision_1 = result[0,0] / (result[0,0]+result[0,1])
print (f"Precision Malignant:  {precision_1:.4f}")

recall_1 = result[0, 0] / (result[0,0]+ result[1,0])
print(f"Recall Malignant: {recall_1:.4f}")

F1_1 = 2 * (precision_1 * recall_1 / (precision_1 + recall_1))
print(f"F1 Malignant: {F1_1:.4f}")

precision_2 = result[1,1] / (result[1,0]+ result[1,1])
print (f"Precison Benign: {precision_2:.4f}")

recall_2 = result[1, 1] / (result[1,1]+ result[0,1])
print(f"Recall Benign: {recall_2:.4f}")

F1_2 = 2 * (precision_2 * recall_2 / (precision_2 + recall_2))
print(f"F1 Benign: {F1_2:.4f}")


#5.2
import matplotlib.pyplot as plt
## addidnt TP /FP/ TN /FN annotation to te numbers 
group = [['TP', 'FP'], ['FN', 'TN']]
annotations = (np.array([
    [f"{name}: {value}" for name, value in zip(row_names, row_values)]
    for row_names, row_values in zip (group, result)
]))


classes = ["Malignant", "Benign"]

plt.figure(figsize=(10,8))
sns.heatmap(result, annot = annotations, fmt="", cmap='GnBu', xticklabels=classes, yticklabels=classes)

plt.title("Confusion matrix: ", color = "plum")
plt.xlabel("Truth label ", color = "tomato")
plt.ylabel("Predicted label ", color = "#EF4026")

plt.show()

#5.3
from sklearn.metrics import roc_curve, auc
y_prob = best_k.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
print('AUC:', auc(fpr, tpr))
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='#FE420F', lw=4, label=f"ROC Curve (AUC = {roc_auc:.2f})")

plt.plot([0,1], [0,1], color = "#6E7505", lw=2, linestyle='--', label = "AUC = 0.5")
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.05)

plt.xlabel("FPR", color = "#C20078", fontsize = 12)
plt.ylabel("TPR", color = "#C20078", fontsize = 12)
plt.xticks(color = "darkblue")
plt.yticks(color = "darkblue")
plt.title("ROC Curve", color = "#029386", fontweight = 'bold', fontsize = 16)
plt.legend(loc = "lower right", fontsize= 12, labelcolor = "brown")
plt.grid(alpha=0.5)
plt.show()

#6.1
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

knn_2d = KNNClessifier(n_neighbors=8, metric="manhattan")
knn_2d.fit(X_train_pca, y_train)

h = .1
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha = 0.3, cmap = 'Spectral_r')
scatter = plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, edgecolors='r', s = 40, cmap ='Spectral_r', linewidths = 0.5)

plt.title("Decision Boundary ", fontsize = 14, fontweight = 'bold', color = "skyblue")
plt.xlabel("PC one", fontsize = 14, color = "#90EE90")
plt.ylabel("PC two", fontsize = 14, color = "#90EE90")

hndls, lbls = scatter.legend_elements()
plt.legend(hndls, ["Malignant", "Benign"], loc = "upper right", title = "class")

plt.grid(alpha = 0.3)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(k_range, train_acc_s, label = "Train Accuracy", color = "#069AF3", marker = '*')
plt.plot(k_range, cv_scores, label = "CV Accuracy", color = "#EF4026", marker = "D")

best_k = k_range[np.argmax(cv_scores)]
plt.axvline(x=best_k, color = "#AAFF32", linestyle = "--", label= f"Optimal k= {best_k}")

plt.title("K vs Accuracy", fontsize = 16, color = "#13EAC9", fontweight = "bold")
plt.xlabel("Number of K Neighbors", fontsize = 14, color = "#FF0000")
plt.ylabel("Accuracy", fontsize = 14, color = "#FF0000")
plt.legend()
plt.grid(alpha = 0.4)
plt.show()


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

library_knn = KNeighborsClassifier(n_neighbors=8, metric ="manhattan")
library_knn.fit(X_train_scaled, y_train)

cus_model = KNNClessifier(n_neighbors=8, metric="manhattan")
cus_model.fit(X_train_scaled, y_train)

model_pred = cus_model.predict(X_test_scaled)
library_model_pred = library_knn.predict(X_test_scaled)


f1_1 = [f1_score(y_test, model_pred, pos_label = 0), f1_score(y_test, model_pred, pos_label=1)]
f1_2 = [f1_score(y_test, library_model_pred, pos_label = 0), f1_score(y_test, library_model_pred, pos_label=1)]

lbls = ["Malignant", "Benign"]
x = np.arange(len(lbls))
width = 0.25

plt.figure(figsize=(10,8))
plt.bar(x - width/2, f1_1, width, label = "Knn from scratch", color = "#380282", alpha=0.5)
plt.bar(x + width/2, f1_2, width, label = "Sklearn Knn", color = "#15B01A", alpha = 0.5)
plt.title("F1 Scores Sklearn Model vs Custom Model", fontweight = 'bold', fontsize =16, color ="#A52A2A")
plt.xticks(x, lbls)
plt.ylabel("F1 Scores", color = "green")
plt.xlabel("Classes", color = "green")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha = 0.5)

plt.show()


#2.3 trying knn w/out scalling and comparing results knn with scalling 
#with scalling
knn_scaled = KNNClessifier(n_neighbors=8, metric="manhattan")
knn_scaled.fit(X_train_scaled, y_train)
scaled_predictions = knn_scaled.predict(X_test_scaled)
accuracy_scaled = np.mean(scaled_predictions == y_test)

# without scaling 
knn_unscaled = KNNClessifier(n_neighbors=8, metric="manhattan")
knn_unscaled.fit(X_train, y_train)
unscaled_predictions = knn_unscaled.predict(X_test)
unscaled_accurcy = np.mean(unscaled_predictions == y_test)
 
print(f"Accuracy without Scaling: {unscaled_accurcy: .4f}")
print(f"Accuracy with Scaling: {accuracy_scaled: .4f}")


