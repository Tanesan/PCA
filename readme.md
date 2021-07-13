# 主成分分析　Principal Component Analysis
-  教師なし線形変換法
-  特徴量抽出と次元削減
    

## 1. データを標準化する
データの読み込み
``` python
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
``` 
2列目以降のデータをXに、１列目のデータをyに格納
``` python
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
``` 
訓練データとテストデータに分割
``` python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
```
平均と標準偏差を用いて標準化
``` python
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)
```
## 2. 共分散行列を作成する
``` python
cov_mat = np.cov(X_train_std.T)  # 共分散行列を作成
```
## 3. 共分散行列の固有値と固有ベクトルを取得する
``` python
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)  # 固有値と固有ベクトルを計算
```
## 4. 固有値を降順でソートすることで固有ベクトルをランク付する
固有値を合計
``` python
tot = sum(eigen_vals)
```
分散説明率を計算
``` python
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
```
分散説明率の累積和を取得
``` python
cum_var_exp = np.cumsum(var_exp)
import matplotlib.pyplot as plt
``` 
分散説明率の棒グラフ
``` python
plt.bar(range(1, 14), var_exp, alpha=0.5, align='center', label='Individual explained variance')
``` 
分散説明率の累積話の階段グラフを作成
``` python
plt.step(range(1, 14), cum_var_exp, where='mid', label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal Component index')
plt.tight_layout()
plt.show()
``` 
一つ目の主成分だけで分散の40%をしめている. 2つの主成分を合わせると分散の60%になる

### 特徴量変換
## 5. もっとも大きいk個の固有値に対応するk個の固有ベクトルを選択する。この場合のkは新しい特徴量部分空間の次元数を表す。
## 6. 上位k個の固有ベクトルかr射影行列Wを作成する
## 7. 射影行列Wを使ってd次元の入力データセットXを変換し、新しいk次元の特徴量部分空間を取得する
 
(固有値, 固有ベクトル)のタプルリスト  
abs 絶対値取る関数  
``` python
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
``` 
## 固有値の大きいものから順に固有対を並び替え、選択された固有ベクトルか射影行列を生成する
この射影行列を使ってデータをより低い次元の部分空間に変換する
``` python
eigen_pairs.sort(key=lambda k: k[0], reverse=True)
w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
print(w)
``` 

# scikit-learn での実装 PCA

plot用　以前やったやつのコピペ
``` python
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
```
### 主成分数は2 PCAのインスタンスを作成
``` python
pca = PCA(n_components=2)
``` 
### 次元削除
``` python
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
``` 
### ロジスティック回帰のインスタンスを作成
``` python
lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')
```
### 削減したデータセットでロジスティック回帰モデルを適合
``` python
lr = lr.fit(X_train_pca, y_train)
``` 
### 決定境界をプロット
``` python
plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
``` 
### テスト
``` python
plot_decision_regions(X_test_pca, y_test, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('images/05_05.png', dpi=300)
plt.show()
```
## 線形判別分析
### LDA 特徴量抽出手法
計算効率を高め、正則化されていないモデルで「次元の呪い」による過学習を抑制   
クラスの分離を最適化する特徴量部分空間を見つける.  
線形判別分析とは2つのクラスを"最もよく判別できる"直線を求める手法です．  
　データが直線のどちら側にあるかを見ることで，どちらのクラスに属するか判別することができます． 
by Qiita [https://qiita.com/pira/items/4c84399671be2cb598e4]  


- 1. D次元のデータセットを標準化
- 2. クラスごとにd次元の平均ベクトルを計算
- 3. 平均ベクトルを使って、クラス間変動行列SBとクラスない変動行列SWを生成
- 4. 行列Sw-1Sbの固有ベクトルと対応する固有値を計算する
- 5. 固有値を降順でソートすることで対応する固有ベクトルをランク付けする
- 6. d*k次元の変換行列Wを生成するために、もっとも大きいk個の固有値に対応するk個の固有ベクトルを選択する
- 7. 変動行列wを使ってデータ点を新しい特徴部分空間へ射影する
