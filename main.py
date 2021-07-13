import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
# 主成分分析　PCA
# 教師なし線形変換法
# 特徴量抽出と次元削減
# 1. データを標準化する

# 2列目以降のデータをXに、１列目のデータをyに格納
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
# 平均と標準偏差を用いて標準化
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)
# 2. 共分散行列を作成する
cov_mat = np.cov(X_train_std.T)  # 共分散行列を作成
# 3. 共分散行列の固有値と固有ベクトルを取得する
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)  # 固有値と固有ベクトルを計算
# 4. 固有値を降順でソートすることで固有ベクトルをランク付する
# 固有値を合計
tot = sum(eigen_vals)
# 分散説明率を計算
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
# 分散説明率の累積和を取得
cum_var_exp = np.cumsum(var_exp)
import matplotlib.pyplot as plt

# 分散説明率の棒グラフ
# plt.bar(range(1, 14), var_exp, alpha=0.5, align='center', label='Individual explained variance')
# # 分散説明率の累積話の階段グラフを作成
# plt.step(range(1, 14), cum_var_exp, where='mid', label='Cumulative explained variance')
#
# plt.ylabel('Explained variance ratio')
# plt.xlabel('Principal Component index')
# plt.tight_layout()
# plt.show()
# 一つ目の主成分だけで分散の40%をしめている. 2つの主成分を合わせると分散の60%になる

# 特徴量変換
# 5. もっとも大きいk個の固有値に対応するk個の固有ベクトルを選択する。この場合のkは新しい特徴量部分空間の次元数を表す。
# 6. 上位k個の固有ベクトルかr射影行列Wを作成する
# 7. 射影行列Wを使ってd次元の入力データセットXを変換し、新しいk次元の特徴量部分空間を取得する

# (固有値, 固有ベクトル)のタプルリスト
# abs 絶対値取る
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
# 固有値の大きいものから順に固有対を並び替え、選択された固有ベクトルか射影行列を生成する
# この射影行列を使ってデータをより低い次元の部分空間に変換する
eigen_pairs.sort(key=lambda k: k[0], reverse=True)
w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
print(w)


# scikit-learn での実装 PCA

# plot用　以前やったやつのコピペ
def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot examples by class
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.6,
                    color=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx],
                    label=cl)


# 主成分数は2 PCAのインスタンスを作成
pca = PCA(n_components=2)
# 次元削除
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
# ロジスティック回帰のインスタンスを作成
lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')
# 削減したデータセットでロジスティック回帰モデルを適合
lr = lr.fit(X_train_pca, y_train)
# 決定境界をプロット
# plot_decision_regions(X_train_pca, y_train, classifier=lr)
# plt.xlabel('PC 1')
# plt.ylabel('PC 2')
# plt.legend(loc='lower left')
# plt.tight_layout()
# plt.show()
# テスト
# plot_decision_regions(X_test_pca, y_test, classifier=lr)
# plt.xlabel('PC 1')
# plt.ylabel('PC 2')
# plt.legend(loc='lower left')
# plt.tight_layout()
# # plt.savefig('images/05_05.png', dpi=300)
# plt.show()

# 線形判別分析
# LDA 特徴量抽出手法
# 計算効率を高め、正則化されていないモデルで「次元の呪い」による過学習を抑制
# クラスの分離を最適化する特徴量部分空間を見つける
# 1. D次元のデータセットを標準化
# 2. クラスごとにd次元の平均ベクトルを計算
# 3. 平均ベクトルを使って、クラス間変動行列SBとクラスない変動行列SWを生成
# 4. 行列Sw-1Sbの固有ベクトルと対応する固有値を計算する
# 5. 固有値を降順でソートすることで対応する固有ベクトルをランク付けする
# 6. d*k次元の変換行列Wを生成するために、もっとも大きいk個の固有値に対応するk個の固有ベクトルを選択する
# 7. 変動行列wを使ってデータ点を新しい特徴部分空間へ射影する

np.set_printoptions(precision=4)
mean_vecs = []
for label in range(1, 4):
    mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))

d = 13
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.zeros((d, d))
    for row in X_train_std[y_train == label]:
        row, mv = row.reshape(d, 1), mv.reshape(d, 1)
        class_scatter += (row - mv).dot((row - mv).T)
    S_W += class_scatter
print('Within-class scatter matrix:%sx%s' % (S_W.shape[0], S_W.shape[1]))
# クラスラベルが一様かどうか確認
print(np.bincount(y_train)[1:])
# 一様ではないため、スケーリングを行う
# 共分散行列の計算と同じ
for label, mv in zip(range(1,4), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train == label].T)
    S_W += class_scatter
print('Within-class scatter matrix:%sx%s' % (S_W.shape[0], S_W.shape[1]))
# クラス間変動行列
mean_overall = np.mean(X_train_std, axis=0)
d = 13 # 特徴量の個数
S_B = np.zeros((d, d))
for i, mean_vec in enumerate(mean_vecs):
    n = X_train_std[y_train == i+1, :].shape[0]
    mean_vec = mean_vec.reshape(d, 1) # 列ベクトル
    mean_overall = mean_overall.reshape(d, 1)
    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)

print('Within-class scatter matrix:%sx%s' % (S_W.shape[0], S_W.shape[1]))

# 線形判別の選択
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
eigen_pairs = sorted(eigen_pairs, key=lambda k:k[0], reverse=True)

tot = sum(eigen_vals.real)
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)

plt.bar(range(1, 14), discr, alpha=0.5, align='center',
        label='Individual "discriminability"')
plt.step(range(1, 14), cum_discr, where='mid',
         label='Cumulative "discriminability"')
plt.ylabel('"Discriminability" ratio')
plt.xlabel('Linear discriminants')
plt.ylim([-0.1, 1.1])
plt.legend(loc='best')
plt.tight_layout()
# plt.savefig('images/05_07.png', dpi=300)
plt.show()

w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
              eigen_pairs[1][1][:, np.newaxis].real))
print('Matrix W:\n', w)

X_train_lda = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_lda[y_train == l, 0],
                X_train_lda[y_train == l, 1] * (-1),
                c=c, label=l, marker=m)

plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower right')
plt.tight_layout()
# plt.savefig('images/05_08.png', dpi=300)
plt.show()

lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)





lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')
lr = lr.fit(X_train_lda, y_train)

plot_decision_regions(X_train_lda, y_train, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('images/05_09.png', dpi=300)
plt.show()




X_test_lda = lda.transform(X_test_std)

plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('images/05_10.png', dpi=300)
plt.show()

