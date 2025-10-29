---
layout: post
title:  "つくりながら学ぶ！LLM自作入門 ― PyTorch編（第2部）"
date:   2025-10-28 10:43:22 +0900
categories: LLM自作入門
---



本日は**『つくりながら学ぶ！LLM自作入門』**PyTorch編、その第2部です。

個人的な記録用なので、内容は非常に圧縮された形で記載されます。

## データローダーのセットアップ

PyTorchはDatasetクラスとDataLoaderクラスを実装しています。Datasetは、各データレコードがどのように読み込まれるかを定義するオブジェクトのインスタンス化に使われます。DataLoaderは、データがシュッフルされ、バッチ化される方法を定義します。

PyTorchでは、クラスラベルはラベル0から始まります。クラスラベルの最大値は、出力ノードの数から1を引いた値を超えてはならないことになっています。Pythonのインデックスは0から始まるからです。

```python
from torch.utils.data import Dataset


class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]        
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]

train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)
```

PyTorchでは、カスタムDatasetクラスの主要なコンポーネントは、`__init__()`コンストラクタメソッド、`__getitem__()`メソッド、`__len__()`メソッドの3つです。`__init__()`メソッドでは、あとから`__getitem__()`メソッドと`__len__()`メソッドでアクセスできる属性を設定します。これらの属性は、ファイルパス、ファイルオブジェクト、データベース接続などです。`__getitem__()`メソッドでは、indexを使ってデータセットのアイテムを1つだけ取得する方法を定義します。このアイテムは、訓練サンプルまたはテストインスタンスに対応する特徴量とクラスラベルです。`__len__()`メソッドは、データセットの長さを取得します。

```python
from torch.utils.data import DataLoader

torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=0
    drop_last=True
)

test_ds = ToyDataset(X_test, y_test)

test_loader = DataLoader(
    dataset=test_ds,
    batch_size=2,
    shuffle=False,
    num_workers=0
)
```

`train_loader`は訓練データセットを反復処理して、各訓練サンプルに1回ずつアクセスします。これを「訓練エポック」と呼びます。訓練エポックの最後のバッチとしてかなり小さいバッチを使うと、訓練中の収束の妨げになることがあります。この問題を回避するために、`drop_last=True`を設定します。`num_workers=0`に設定すると、データの読み込みはメインプロセスで実行され、別のワーカープロセスでは実行されなくなります。対照的に、`num_workers`を0よりも大きな値に設定すると、データを平行して読み込むために複数のワーカープロセスが開始され、メインプロセスがモデルの訓練に集中できるようになるため、システムリソースをより効率的に活用できるようになります。

## 訓練ループ

```python
import torch.nn.functional as F


torch.manual_seed(123)
model = NeuralNetwork(num_inputs=2, num_outputs=2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

num_epochs = 3

for epoch in range(num_epochs):
    
    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):

        logits = model(features)
        
        loss = F.cross_entropy(logits, labels) # 損失関数
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        ### ログ
        print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
              f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
              f" | Train/Val Loss: {loss:.2f}")

    model.eval()
    with torch.no_grad():
        outputs = model(X_train)

    print(outputs)
```

学習率はハイパーパラメータであり、損失値を観測しながら調整しなければならない設定です。また、`model.train()`と`model.eval()`という設定は、ドロップアウト層やバッチ正規化層など、訓練モードと評価モードで挙動が異なるコンポーネントで必要になります。

また、上記のコードではロジットを直接`cross_entropy()`損失関数に渡しています。この関数は効率性と数値的な安定性の観点から、**内部でソフトマックス関数を適用します**。続いて、`loss.backward()`を呼び出すと、PyTorchがバックラウンドで構築した計算グラフで、勾配が計算されます。`optimizer.step()`呼び出しでは、損失を最小化するために、その勾配を使ってモデルパラメータを更新します。勾配が累積されるのを防ぐために、各イテレーションで`optimizer.zero_grad()`を呼び出し、勾配を0にリセットすることが重要です。

PyTorchの`argmax()`関数を使うと、これらの値をクラスラベルの予測値に変換できます。`argmax()`関数に`dim=1`を指定すると、各行において最も大きい値のインデックス位置が返されます（`dim=0`の場合は、各列において最も大きい値のインデックスが返されます）。クラスラベルを得るためにソフトマックス確率を計算する必要はないことに注意してください。`argmax()`関数をロジットに直接適用することもできます。

## モデルの保存と読み込み

```python
torch.save(model.state_dict(), "model.pth")
```

モデルの`state_dict`はPythonのディクショナリ（辞書）オブジェクトであり、モデル内の各層とその訓練可能なパラメータ（重みとバイアス）をマッピングします。"model.pth"は、ディスクに保存するモデルファイルの任意のファイル名です。

```python
model = NeuralNetwork(2, 2) # 本来のモデルのアーキテクチャと一致しなければならない
model.load_state_dict(torch.load("model.pth", weights_only=True))
```

`torch.load("model.pth")`関数は、ファイル"model.pth"を読み込む、モデルのパラメータを含んでいるPythonディクショナリオブジェクトを再構築します。一方、`model.load_state_dict()`関数は、これらのパラメータをモデルに適用することで、モデルが保存されたときの学習状態を実質的に復元します。

## GPUとデバイス

Pytorchでは、「デバイス」とは計算が実行され、データが保存される場所のことです。PyTorchのテンソルはデバイス上にあり、テンソルでの演算は同じデバイス上で実行されます。

`to()`メソッドを使うと、テンソルをGPUやCPUに転送できます。

```python
tensor_1 = tensor_1.to("cuda")
```

コンピュータに複数のGPUが搭載されている場合は、`to("cuda:0")`や`to("cuda:1")`のように、転送コマンドにデバイスIDを指定できます。`device = torch.device("cuda")`のようにオブジェクトを作成することもできます。

分散訓練は、モデルの訓練を複数のGPUやコンピュータに分散させるという概念です。PyTorchは各GPUで別々のプロセクを起動し、このプロセスにそれぞれモデルのコピーを渡します。これらのモデルのコピーは訓練時に同期されます。DistributedSamplerを使うことで、各GPUに異なるバッチを確実に受け取らせることができます。

モデルの各コピーは、訓練データの異なるサンプルを受け取ります。このため、各コピーから返されるロジットは異なり、バックワードパスで異なる勾配が計算されんす。これらの勾配は、モデルを更新するために訓練中に平均化され、同期されます。このようにして、モデルが発散しないようになります。

PyTorchのmultiprocessingサブモジュールには、複数のプロセスを生成し、複数の入力に対して何らかの関数を並列に適用するための`multiprocessing.spawn()`のような関数が含まれています。`init_process_group()`と`destroy_process_group()`は、分散訓練の初期化と終了に使われます。`init_process_group()`は、分散環境のプロセスごとにプロセスグループを初期化するために、訓練スクリプトの最初に呼び出します。`destroy_process_group()`は、プロセスグループを解除してそのリソースを開放するために、訓練スクリプトの最後に呼び出します。



**参考文献**  
Sebastian Raschka. 『つくりながら学ぶ！LLM自作入門』（Build a Large Language Model (From Scratch)）. 株式会社クイープ訳、東京: マイナビ出版, 2025.