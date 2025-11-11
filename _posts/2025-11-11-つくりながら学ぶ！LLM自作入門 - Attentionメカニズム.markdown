---
layout: post
title:  "つくりながら学ぶ！LLM自作入門 ― Attentionメカニズム（第2部）"
date:   2025-11-11 10:43:22 +0900
categories: LLM自作入門
---



本日は**『つくりながら学ぶ！LLM自作入門』**Attentionメカニズム編、その第2部です。

個人的な記録用なので、内容は非常に圧縮された形で記載されます。

## 訓練可能な重みをもつSelf-Attention

次のステップは、オリジナルのTransformerアーキテクチャ、GPTモデル、その他のよく知られているLLMで使われているSelf-Attentionメカニズムを実装することです。このSelf-Attentionメカニズムは、Scaled Dot-Product Attentionとも呼ばれます。訓練可能な重みを持つSelf-Attentionも、その基本的な原理は単純化されたSelf-Attentionと同じです。つまり、特定の入力要素に対応する入力ベクトルの加重和としてコンテキストベクトルを計算します。

単純化されたSelf-Attentionとの最も顕著な違いは、モデルの訓練中に更新される重み行列を導入する点です。このような訓練可能な重み行列は、モデル（具体的には、モデル内部のAttentionモジュール）が適切なコンテキストベクトルの構築方法を学習する上で非常に重要です。

ここでは、訓練可能な重み行列W<sub>q</sub>、W<sub>k</sub>、W<sub>v</sub>を導入することで、Self-Attentionメカニズムを段階的に実装します。この3つの行列は、それぞれ入力トーク埋め込みx<sup>i</sup>をクエリベクトル、キーベクトル、値ベクトルに射影するために使われます。

訓練可能な重み行列を持つSelf-Attentionメカニズムの最初のステップでは、入力要素xに対してクエリベクトル（q）、キーベクトル（k）、値ベクトル（v）を計算します。たとえば、クエリベクトルq<sup>2</sup>は、入力x<sup>2</sup>と重み行列W<sub>q</sub>の行列積によって得られます。同様に、キーベクトルk<sup>2</sup>と値ベクトルv<sup>2</sup>はそれぞれ入力x<sup>2</sup>と重み行列W<sub>k</sub>、W<sub>v</sub>の行列積によって得られます。

まず、変数をいくつか定義します。

```python
x_2 = inputs[1] # 入力x_2
d_in = inputs.shape[1] # 入力の埋め込みサイズ, d=3
d_out = 3 # 出力の埋め込みサイズ, d=2
```

GPT型のモデルでは、入力と出力の次元は通常は同じくします。次は重み行列の初期化です。

```python
torch.manual_seed(123)

W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=True)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=True)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=True)
```

モデルの訓練に重み行列を使う場合は、訓練中にそれらの行列を更新するためにrequires_grad=Trueに設定することになります。次に、クエリベクトル、キーベクトル、値ベクトルを計算します。

```python
query_2 = x_2 @ W_query # この例では入力要素x_2を対象としています。
key_2 = x_2 @ W_key 
value_2 = x_2 @ W_value

print(query_2)
```

重み行列Wの「重み」は、「重みパラメータ」の略です。重みパラメータはニューラルネットワークの訓練中に最適化される値です。これをAttentionの重みと混同しないように注意してください。すでに見てきたように、Attentionの重みは、コンテキストベクトルが入力の異なる部分にどの程度依存するのかーつまり、ネットワークが入力の異なる部分に注意を払う度合いを決定します。

要するに、重みパラメータはネットワークの接続（結合）を定義するために学習される基本的な係数であり、Attentionの重みはコンテキストに特化した動的な値です。

とりあえずの目標はコンテキストベクトルを1つ計算することですが、すべての入力要素のキーベクトルと値ベクトルも必要です。なぜなら、それらのベクトルはクエリq<sup>2</sup>に対するAttentionの重みの計算に必要だからです。

これらのキーベクトルと値ベクトルは行列積を使って求めることができます。

```python
keys = inputs @ W_key 
values = inputs @ W_value

print("keys.shape:", keys.shape)
print("values.shape:", values.shape)
```

Attentionスコアの計算は、単純化されたSelf-Attentionメカニズムで使ったものと同様のドット計算です。新しい部分は、入力要素のドット積を直接計算するのではなく、入力を対応する重み行列で線形変換することによって得られるクエリとキーを使うことです。まず、コンテキストベクトルを導出しようとする入力トークンからクエリを導出します。そのクエリベクトルとキーベクトルのドット積として、スケールしていないAttentionスコアを計算します。

```python
keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)
```

このクエリに対するすべてのAttentonスコアも行列積で計算できます。

```python
attn_scores_2 = query_2 @ keys.T
print(attn_scores_2)
```

ここで、AttentionスコアからAttentionの重みの計算に進みます。Attentionスコアをスケールし、ソフトマックス関数を使ってAttentionの重みを計算します、ただし今回は、Attentionスコアをキーの埋め込み次元の平方根で割るという方法でスケールします（平方根をとることは、数学的には0.5で指数をとるのと同じです）。繰り返すと、Attentionスコアwを計算した後、ソフトマックス関数で正規化し、Attentionの重みaを求めます。

```python
d_k = keys.shape[1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print(attn_weights_2)
```

埋め込み次元のサイズで正規化を行う理由は、小さな勾配を避けて訓練性能を向上させることにあります。たとえばGPT型のLLMでは、埋め込み次元が通常は1,000を超えるようにスケールアップされるため、大胡なドット積にソフトマックス関数が適用され、誤差逆伝播法で非常に小さな勾配が発生する可能性があります。ドット積が大きくなるに従い、ソフトマックス関数の振る舞いがステップ関数のようになり、勾配がゼロに近づいていきます。このような小さな勾配は、学習を極端に遅くしたり、訓練を停滞させたりする原因になります。埋め込み次元の平方根によるスケーリングは、このSelf-AttentionメカニズムがScaled Dot-Product Attentionと呼ばれる所以です。

Self-Attentionの最後のステップでは、Attentionの重みを使ってすべての値ベクトルを組み合わせることで、コンテキストベクトルを計算します。訓練可能な重みを持つSelf-Attentionでは値ベクトルに対する加重和としてコンテキストベクトルを計算します。この計算に使われるAttentionの重みは、各値ベクトルの重要度を評価して重み付けする因子として機能します。また、前回と同様に、行列積を使うと出力を1ステップで計算できます。

```python
context_vec_2 = attn_weights_2 @ values
print(context_vec_2)
```

Attentionメカニズムで言うところの「クエリ」、「キー」、「値」は、情報検索とデータベースの分野から拝借したものです。それらの分野では、情報の格納、検索、取得に同じような概念が使われています。

クエリ（query）は、データベースの検索クエリに相当するもので、文中の単語やトークンなど、モデルが注目している（理解しようとしている）現在のアイテムを表します。クエリは、入力シーケンスの他の部分にどの程度注意を払うべきかを判断するために使われます。

キー（key）は、インデックス付けや検索に使われるデータベースのキーのようなものです。Attentionメカニズムでは、文中の単語といった入力シーケンスの各アイテムにキーが関連付けられています。これらのキーはクエリのマッチングに使われます。

Attentionメカニズムでの値（value）は、データベースのキーバリューペアのバリューと同じで、入力アイテムの実際の内容（表現）を表します。モデルは、クエリに最も関連しているのはどのキーかーつまり、現在注目しているアイテムに最も関連しているのは入力のどの部分かを判断し、対応する値を取り出します。

それでは、今までの実装を念頭に置いて、クラスを設計してみましょう。

```python
import torch.nn as nn

class SelfAttention_v1(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )

        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))
```

このSelfAttention_v1は、torch.nn.Moduleの派生クラスです。Moduleは、PyTorchのモデル層の作成と管理に必要な機能を提供するPyTorchモデルの基本的な構成要素です。

__init__()メソッドは、クエリ、キー、値に対する訓練可能な重み行列（W_query、W_key、W_value）を初期化します。これらの重み行列のサイズはそれぞれ入力次元d_inから出力次元d_outへの変換を表します。

フォワードパスでは、forward()メソッドを使います。このメソッドは、クエリとキーの積に基づいてAttentionスコア（attn_scores）を計算し、ソフトマックス関数を使って正規化し、Attentionの重みを求めます。最後に、これらの重みを値に掛けることで、コンテキストベクトルを生成します。

言い換えると、Self-Attentionでは、入力行列Xの入力ベクトルを3つの重み行列W<sub>q</sub>、W<sub>k</sub>、W<sub>v</sub>で変換します。結果として得られたクエリ（Q）と値（K）に基づいてAttention重み行列を計算します。続いて、Attention重み行列と値（V）を使ってコンテキストベクトル（Z）を計算します。

Self-Attentionでは、訓練可能な重み行列W<sub>q</sub>、W<sub>k</sub>、W<sub>v</sub>が使われます。これらの行列はそれぞれ入力データを変換し、Attentionメカニズムの重要な構成要素であるクエリ、キー、値を生成します。**こうした訓練可能な重み行列は、モデルが訓練中にさらにデータを見ることによって調整されます**。

PyTorchのtorch.nn.Linear層を利用すれば、SelfAttention_v1の実装をさらに改善できます。バイアスユニットを無効にすると、Linearは実質的に行列積を計算します。さらに、torch.nn.Parameter(torch.rand(...))を明示的に実装する代わりにLinearを利用することには、Linearが最適化された重み初期化スキームを組み込んでいて、より安定した効果的なモデルの訓練に貢献するという大きな利点もあります。

```python
class SelfAttention_v2(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))
```

第3部では、因果的要素とマルチヘッド要素を取り入れることに焦点を合わせた上で、このSelf-Attentionメカニズムを改良します。因果的要素は、モデルがシーケンスの未来の情報にアクセスしないようにAttentionメカニズムを修正するために使われます。このアプローチは、各単語の予測が前の単語にのみ依存すべきである言語モデリングのようなタスクにとって非常に重要です。

マルチヘッド要素は、Attentionメカニズムを複数の「ヘッド」に分割するために使われます。それぞれのヘッドがデータの異なる側面を学習することで、異なる位置にあるさまざまな表現部分空間の情報にモデルが同時に注意を払えるようになります。これにより、複雑なタスクでのモデルの性能がよくなります。


**参考文献**  
Sebastian Raschka. 『つくりながら学ぶ！LLM自作入門』（Build a Large Language Model (From Scratch)）. 株式会社クイープ訳、東京: マイナビ出版, 2025.