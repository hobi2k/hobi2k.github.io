---
layout: post
title:  "つくりながら学ぶ！LLM自作入門 ― Attentionメカニズム（第3部）"
date:   2025-11-12 00:10:22 +0900
categories: LLM自作入門
---



本日は**『つくりながら学ぶ！LLM自作入門』**Attentionメカニズム編、その第3部です。

個人的な記録用なので、内容は非常に圧縮された形で記載されます。

## Causal Attention

多くのLLMタスクでは、シーケンスの次のトークンを予測するときに、Self-Attentionメカニズムに現在の位置よりも前に現れるトークンだけを考慮させたくなります。Causal Attentionは、Masked Attentionとも呼ばれる特殊なSelf-Attentionであり、与えられたトークンを処理してAttentionスコアを計算するときに、モデルがシーケンスの前と現在の入力だけを考慮するように制約します。この点で、入力シーケンス全体に一度にアクセスできる標準のSelf-Attentionメカニズムとは対照的です。

Causal Attentionは、この後の章で取り組むLLMの開発に不可欠なメカニズムです。このメカニズムをGPT型のLLMで実現するために、トークンを処理するたびに、入力テキストの現在のトークンよりも後ろにある未来のトークンをマスクします。行列の上三角部分（対角線よりも上の部分）にあるAttentionの重みをマスクし、マスクされていない重みを正規化して各行の重みの合計が1になるようにします。Causal Attentionでは、与えられた入力に対してLLMがAttentionの重みを使ってコンテキストベクトルを計算するときに、未来のトークンにアクセスできないように行列の上三角部分（対角線よりも上の部分）をマスクします。

Causal AttentionでマスクされたAttention重み行列を求める方法の1つは、Attentionスコアにソフトマックス関数を適用し、行列の上三角部分（対角線よりも上の部分）の要素をゼロ化し、結果の行列を正規化することです。

最初のステップでは、前回と同じように、ソフトマックス関数を使ってAttentionの重みを計算します。

```python
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs) 
attn_scores = queries @ keys.T

attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)
```

2つ目のステップは、PyTorchのtril()関数を使ってマスクを作成することです。このマスクの対角線よりも上の部分の値は0です。

```python
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print(mask_simple)
```

このマスクをAttentionの重みに掛けると、行列の上三角部分の値を0にできます。

```python
masked_simple = attn_weights*mask_simple
print(masked_simple)
```

3つ目のステップは、各行のAttentionの重みが合計で1になるように再び正規化することです。各行の各要素を各行の合計で割ると、この正規化を実現できます。

```python
row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)
```

マスクを適用した後にAttention重み行列を再び正規化する際には、最初は（マスクするはずの）未来のトークンの情報がまだ現在のトークンに影響を与える可能性があるように見えるかもしれません。なぜなら、未来のトークンの値がソフトマックス関数の計算に含まれているからです。ただし、ここでの重要なポイントは、マスキング後にAttention重み行列を再び正規化するときに実質的に行っているのは、（マスクされた位置を除いた）より小さなサブセットでのソフトマックスの再計算だということです。マスクされた位置はソフトマックスの値に寄与しないからです。

ソフトマックスには、最初はすべての位置を分母に取り込んでいたにもかかわらず、マスキングと再正規化の後は、マスクされた位置が無効になるという優れた数学的性質があります。つまり、それらの位置が意味のある方法でソフトマックスのスコアに影響を与えることはありません。

もう少し単純に言うと、マスキングと再正規化後のAttention重み行列の分布は、まるで最初からマスクされていない位置だけで計算されたかのようになります。

しかし、このコードにはまだ改善の余地があります。ソフトマックス関数の数学的性質を利用して、マスクされたAttentionの重みの計算をより少ないステップ数で効率よく実装してみましょう。

Causal AttentionでマスクされたAttention重み行列を求めるより効率的な方法は、ソフトマックス関数を適用する前に、Attentionスコアを負の無限大の値でマスクすることです。ソフトマックス関数は、入力を確率分布に変換します。負の無限大の値（-∞）が行に存在する場合、ソフトマックス関数はそれらの値をゼロの確率として扱います（数学的には、e<sup>-∞</sup>が0に近づくためです）。

対角線よりも上の部分に1を持つマスクを作成し、これらの1を
負の無限大の値（-∞）に置き換えると、より効率的なマスキングトリックを実装できます。

```python
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)
```

あとは、マスクされた結果にソフトマックス関数を適用すれば完了です。

```python
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)
```

ディープラーニングでのドロップアウト（dropout）とは、ランダムに選択された隠れ層ユニットを訓練中に無視することで、実質的に「ドロップアウト」させるテクニックのことです。このテクニックは、特定の隠れ層ユニットのセットにモデルが過度に依存しないようにすることで、過剰適合を防ぐのに役立ちます。ここで重要なのは、ドロップアウトが使われるのは訓練時だけで、訓練後は無効になることです。

GPTのようなモデルをはじめとするTransformerアーキテクチャでは、Attentionメカニズムのドロップアウトは一般に2つのタイミングで適用されます。1つはAttentionの重みを計算した後であり、もう1つはAttentionの重みを値ベクトルに適用した後です。

次のコードでは、ドロップアウト率を50%にしており、Attentionの重みの半分がマスクされることになります。後でGPTモデルを訓練する際には、0.1や0.2といったもっと低いドロップアウト率を使います。

```python
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
example = torch.ones(6, 6)

print(dropout(example))
```

Attention重み行列に50%の割合でドロップアウトを適用すると、行列の要素の半分がランダムにゼロ化されます。アクティブな（0に設定されなかった）要素の減少を補うために、行列の残りの要素の値は係数`1 / 0.5 = 2`でスケールアップされます。このスケーリングはAttentionの重みの全体的なバランスを維持する上で非常に重要です。このようにすると、Attentionメカニズムの平均的な影響力が訓練フェースと推論フェースの両方で一貫した状態に保たれるからです。

次は、Casual Attentionとドロップアウトによる調整を追加したPythonクラスです。

```python
class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # ドロップアウト
        self.dropout = nn.Dropout(dropout)
        # register_buffer()呼び出し追加
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        # バッチ次元b
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 次元1と2を入れ替える
        attn_scores = queries @ keys.transpose(1, 2)
        # PyTorchでは、末尾にアンダースコアをもつ演算はインプレースで実行され、無駄なメモリコピーが回避される。
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)  
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)

context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)

context_vecs = ca(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)
```

PyTorchでの`register_buffer()`呼び出しはすべてのケースで必要というわけではありませんが、この場合はいくつかの利点があります。たとえば、LLMでCausalAttentionクラスを使うときには、LLMを訓練するときに重要であろう、「バッファ（登録されたテンソル）をモデルとともに適切なデバイス（CPUまたはGPU）に移動させる」という操作が行われます。このことは、これらのテンソルがモデルパラメータと同じデバイス上にあることを明示的に確認して、デバイス不一致エラーを防ぐという操作がいらなくなることを意味します。

## Multi-head Attention拡張

最後のステップは、Causal Attentionクラスを複数のヘッドで拡張することです。このメカニズムはMulti-head Attentionとも呼ばれます。「Multi-head」は、Attentionメカニズムを複数の「ヘッド」に分割し、それぞれのヘッドが独立して動作することを表します。

実際にMulti-head Attentionを実装するには、Self-Attentionメカニズムのインスタンスを複数作成し、それぞれのインスタンスに独自の重みを持たせて、それぞれのインスタンスの出力を組み合わせる必要があります。

Multi-head Attentionの主な考え方は、学習済みの異なる線形射影でAttentionメカニズムを複数回（並列に）実行するというものです。学習済みの線形射影とは、入力データ（Attentionメカニズムのクエリベクトル、キーベクトル、値ベクトルなど）に重み行列を掛けた結果のことです。

```python
class MultiHeadAttentionWrapper(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) 
             for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


torch.manual_seed(123)

context_length = batch.shape[1] # トークン数
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, num_heads=2
)

context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)
```

MultiHeadAttentionWrapperを使うときには、Attentionヘッド数（num_heads）を指定します。`num_heads=2`と指定すると、コンテキストベクトル行列が2つ含まれたテンソル得られます。各コンテキストベクトル行列の行はトークンに対応するコンテキストベクトルを表し、列は`d_out=4`で指定された埋め込み次元を表します。これんらのコンテキストベクトル行列を列の次元に沿って連結します。Attentionヘッドが2つ、埋め込み次元が2なので、最終的な埋め込み次元は`2x2=4`です。

ただし、ここでのモジュールは`forward()`メソッドの`[head(x) for head in self.heads]`で逐次的に処理されます。複数のヘッドを並列に処理すれば、この実装を改善できるはずです。そのための1つの方法は、行列積を使ってすべてのAttentionヘッドの出力を同時に計算することです。

次のMultiHeadAttentionクラスは、マルチヘッド機能を1つのクラスに統合します。このクラスは、射影されたクエリテンソル、キーテンソル、値テンソルの形状を変更することで入力を複数のヘッドに分割し、Attentionを計算した後、これらのヘッドからの結果を組み合わせます。

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads 
        # 出力次元数をヘッド数で分割

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  
        # Linear層を使ってヘッドの出力を組み合わせる
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x) 
        # テンソルの形状は(b,num_tokens,d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # num_heads次元を追加して行列を暗黙的に分割。
        # 続いて、最後の次元を展開し、形状を(b, num_tokes, d_out)から(b, num_tokens, num_heads, head_dim)に変換
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 形状を(b, num_tokens, num_heads, head_dim)から(b, num_heads,num_tokens, head_dim)に転置
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 各ヘッドのドット積を計算
        attn_scores = queries @ keys.transpose(2, 3)

        # マスクをトークン数で切り捨て
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Attentionスコアを埋めるためにマスクを使う
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # テンソルの形状は(b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2) 
        
        # self.d_out = self.num_heads * self.head_dimに基づいてヘッドを結合
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        # 線形射影を追加
        context_vec = self.out_proj(context_vec)

        return context_vec

torch.manual_seed(123)

batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)

context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)
```

MultiHeadAttentionクラスでは、統合アプローチをとっています。つまり、マルチヘッド層を出発点とし、内部でマルチヘッド層を個々のAttentionヘッドに分割します。

2つのAttentionヘッドを持つMultiHeadAttentionWrapperクラスでは、2つの重み行列W<sub>q1</sub>、W<sub>q2</sub>を初期化し、2つのクエリ行列Q<sub>1</sub>、Q<sub>2</sub>うぃ計算しました。MultiHeadAttentionクラスでは、より大きな重み行列W<sub>q</sub>を1つだけ初期化し、入力との行列積を1回だけ計算してクエリ行列Qを求め、このクエリ行列をQ<sub>1</sub>とQ<sub>2</sub>に分割します。キーと値についても同じです。

クエリテンソル、キーテンソル、値テンソルの分割は、PyTorchの`view()`メソッドと`transpose()`メソッドを使った
テンソルの形状変更演算と転置演算を通じて表現されます。最初に入力を（クエリ、キー、値に対する線形層を使って）変換し、続いてマルチヘッドを表すために形状を変更します。

この操作の鍵は、d_out次元をnum_headsとhead_dimに分割する部分にあります（head_dim = d_out / num_heads）。この分割は`view()`メソッドを使って実現されます。つまり、次元（b,num_tokens,d_out）のテンソルの形状を次元（b, num_tokens, num_heads, head_dim）に変更します。

これらのテンソルはnum_heads次元がnum_tokens次元の前に来るように転置され、結果として（b, num_heads, num_tokens, head_dim）という形状になります。こうした転置は、異なるヘッド間でクエリ、キー、値を正しい順番に揃え、バッチベースの行列積を効率よく計算する上で非常に重要です。

この場合、PyTorchでの行列積の実装は4次元の入力テンソルを扱うため、行列積は最後の2つの次元（num_tokensとhead_dim）で計算され、個々のヘッドごとに繰り返されます。

MultiHeadAttentionでは、Attention重み行列とコンテキストベクトルを計算した後、すべてのヘッドからのコンテキストベクトルを転置して、（b, num_tokens, num_heads, head_dim）の形状に戻します。これらのベクトルの形状を（b,num_tokens,d_out）に変更（フラット化）すると、実質的にすべてのヘッドからの出力を結合することになります。

さらに、ヘッドを結合した後のMultiHeadAttentionに対して、CausalAttentionクラスにはなかった出力射影層（self.out_proj）を追加しています。この出力射影層は、厳密には必要ありませんが、多くのLLMアーキテクチャで一般的に使われているため、ここでは参考で追加します。

## まとめ

- Attentionメカニズムは入力要素を拡張されたコンテキストベクトル表現に変換します。それらの表現には、すべての入力に関する情報が組み込まれています。
- Self-Attentionメカニズムは、コンテキストベクトル表現を入力の加重和として計算します。
- LLMで使われているSelf-Attentionメカニズムは、Scaled Dot-Product Attentionとも呼ばれ、入力の中間変換（クエリ、キー、値）を計算するために訓練可能な重み行列を導入します。
- テキストを左から右に読んで生成するLLMを扱うときには、LLMが未来のトークンにアクセスしないようにするために、Casual Attentionマスクを追加します。
- LLMでの過剰適合を抑制するために、Attentionの重みをゼロ化するCausal Attentionマスクを加えて、ドロップアウトマスクを追加することができます。
- TransformerベースのLLMのAttentionモジュールは、Causal Attentionの複数のインスタンスを使うため、Multi-head Attentionと呼ばれます。
- Causal Attentionモジュールの複数のインスタンスを積み重ねると、Multi-head Attentionモジュールを作成できます。
- Multi-head Attentionモジュールをより効率よく作成する方法では、バットベースの行列積を使います。



**参考文献**  
Sebastian Raschka. 『つくりながら学ぶ！LLM自作入門』（Build a Large Language Model (From Scratch)）. 株式会社クイープ訳、東京: マイナビ出版, 2025.