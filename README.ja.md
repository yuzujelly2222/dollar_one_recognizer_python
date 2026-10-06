[English](README.md) | [日本語](README.ja.md)

# dollar_one_recognizer_python

$1 Unistroke Recognizer の Python 3 実装


[![PyPI](https://img.shields.io/pypi/v/dollar-one-recognizer-py)](https://pypi.org/project/dollar-one-recognizer-py/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

## これは何か

**$1 Unistroke Recognizer** は、ワシントン大学の Wobbrock, Wilson, Li が UIST 2007 で発表した、
一筆書き（unistroke）ジェスチャーのための軽量な認識アルゴリズムです。機械学習の訓練は不要で、
各図形につき手本（テンプレート）を1つ登録するだけで動作します。仕組みはシンプルで、入力された
点列を一定数に等間隔化し、回転・大きさ・位置のブレを正規化したうえで、登録済みテンプレートと
最も距離が近いものを選ぶ、というものです。ライブラリ依存もほぼなく、タブレットやタッチパネル、
マウスジェスチャーのプロトタイピングに向いています。

`recognize()` / `normalize()` に渡す点列は `[[x, y], ...]` という単純な数値の並びであれば
出所は問いません。マウスやタッチ以外にも、OpenCV のトラッキング（`cv2.calcOpticalFlowPyrLK`
やカラー物体・指先トラッキングなど）で得た座標列をそのままフレームごとに追加していけば、
体の動きや物体の軌跡を一筆書きジェスチャーとして認識させることもできます。

本リポジトリでは、認識器本体をパッケージ化した `dollar_one_recognizer_py/` に加えて、ジェスチャーを
マウスで描いてテンプレートとして保存する `export_gesture_app.py` と、保存したテンプレートと照合して
認識を試せる `recognize_app.py` という2つの Tkinter アプリを同梱しています。

![recognize_app.pyで手描きジェスチャーをテンプレートと照合するデモ](assets/demo.gif)

## インストール

```bash
pip install dollar-one-recognizer-py
```

認識器本体（`dollar_one_recognizer_py`）がインストールされます。依存パッケージは `numpy`（幾何計算）のみです。

GUIアプリ（`export_gesture_app.py` / `recognize_app.py`）も使う場合は、リポジトリごとcloneしてください。
こちらはジェスチャー画像のPNG入出力用に `pillow` も追加でインストールされます。

```bash
git clone https://github.com/yuzujelly2222/dollar_one_recognizer_python.git
cd dollar_one_recognizer_python
pip install -r requirements.txt
```

## 使い方

```python
from dollar_one_recognizer_py import dollar_one_recognizer

# テンプレート登録（各図形につき手本の点列を1つ用意するだけでよい）
line_template = [[x, x] for x in range(0, 101, 5)]
check_template = [[0, 30], [30, 0], [90, 60]]

recognizer = dollar_one_recognizer(
    size=250,
    templates=[line_template, check_template],
    templates_name=["line", "checkmark"],
    n=64,
)

# 認識したいジェスチャーの生座標列（例: マウスドラッグで集めた点）
query = [[x, x * 0.9 + 2] for x in range(0, 101, 4)]

best_points, best_name, score = recognizer.recognize(query)
print(best_name, score)  # -> line 0.96...
```

### OpenCVのトラッキング座標を使う例

点列の出所は問わないため、OpenCVで追跡した座標をフレームごとに貯めて渡すだけで使えます。

```python
import cv2

points = []
cap = cv2.VideoCapture(0)
# ... 何らかのトラッキング処理（オプティカルフロー、色検出、姿勢推定など）で
#     1フレームごとの追跡座標 (x, y) を求め、points に追記していく
while tracking:
    x, y = track_next_point(cap)  # トラッキング処理の実装は用途に応じて用意する
    points.append([x, y])

best_points, best_name, score = recognizer.recognize(points)
```

GUIで試したい場合は、テンプレート登録用アプリと認識用アプリをそれぞれ起動します。

```bash
python export_gesture_app.py   # 描いて gestures/ に CSV+PNG として保存
python recognize_app.py        # gestures/ を自動読み込みして認識をテスト
```

## アルゴリズム解説

$1 は入力された点列に次の4ステップを順に適用し、テンプレートの中から最も近いものを選びます。

1. **リサンプリング** (`_resample`) — 道のり長ベースで点を `n`（既定64）個の等間隔な点に打ち直す。
   描く速さのムラを吸収する。
2. **回転補正** (`_rotate_to_zero`) — 重心から始点への角度が0になるよう全体を回し、大まかな回転を揃える。
3. **スケール＆移動** (`_scale_to_square` → `_translate_to_origin`) — バウンディングボックスを
   `size × size`（既定250×250）の正方形に拡縮し、重心を原点へ移動。大きさ・位置のブレを吸収する。
4. **照合** (`recognize`) — 正規化済みの各テンプレートに対し、±45°の範囲で回転角を振りながら
   64点間の平均ユークリッド距離が最小になる角度を探し（黄金分割探索、`_distance_at_best_angle`）、
   最も距離が小さいテンプレートを認識結果として返す。距離は0〜1のスコアに変換される。

```
入力点列 → _resample(n点に等間隔化) → _rotate_to_zero(回転補正)
        → _scale_to_square + _translate_to_origin(正規化)
        → 各テンプレートと ±45° の黄金分割探索で最小距離を計算 → 最近傍テンプレートを返す
```

ステップ1〜3は `_normalize()` としてまとめられており、テンプレート登録時・認識時のどちらでも
同じ前処理が適用されます。

**Protractorについて**: ステップ4の回転角探索（黄金分割探索）は、同じ著者らが提案した後継手法
「Protractor」ではベクトルの内積計算に置き換えられ、探索ループなしで最適角が閉じた式で求まるため
さらに高速になります。組み込み環境などで速度が要る場合の選択肢です（本実装には未収録）。

## API

### `dollar_one_recognizer`

外部から使うのは以下の公開メンバのみです。幾何計算用のメソッド（`_get_distance` /
`_get_centroid` / `_get_bounding_box` / `_all_point_rotate` / `_resample` / `_rotate_to_zero` /
`_scale_to_square` / `_translate_to_origin` / `_normalize` / `_distance_at_best_angle` /
`_distance_at_angle` / `_path_distance`）は内部実装用のため `_` を付けて非公開にしています。

| メンバ | 説明 |
|---|---|
| `__init__(size, templates, templates_name, n)` | `templates`（点列のリスト）と対応する `templates_name` からテンプレートを登録して初期化 |
| `add_template(points, name)` | テンプレートを1つ追加登録 |
| `recognize(points)` | 生の点列を正規化して全テンプレートと照合し `(best_points, best_name, score)` を返す |

### バリデーション

`__init__` と `add_template` は不正な入力を `ValueError` で弾きます。

- `size` は0より大きいこと、`n` は1より大きいこと
- `templates` と `templates_name` の要素数が一致していること
- 各テンプレートの点列は2点以上であること
- テンプレート名は空（空白のみも含む）でないこと、かつ既存の名前と重複していないこと（前後の空白は自動でトリムされる）

### その他

- `GestureTemplate` — テンプレート1件分の `points`（正規化済み点列）と `name` を持つ入れ物
- `_deg` — 度数法で三角関数を扱うための内部ヘルパー（`sin`/`cos`/`tan`/`arcsin`/`arccos`/`arctan`/`arctan2`）
- `export_gesture_app.py` / `recognize_app.py` — ライブラリではなく、動作確認用のTkinter GUIアプリ

## ドキュメント・テスト

Sphinxで生成したAPIリファレンスが `docs/` にあり、GitHub Pagesとして公開する設定になっています
（`docs_src/` がSphinxのソース、`docs/` がビルド済みの公開用HTMLです）。ソースから再生成する場合:

```bash
pip install sphinx sphinx-rtd-theme
sphinx-build -b html docs_src docs
```

`tests/` にはpytestベースのテストスイートがあります。

```bash
pip install pytest
pytest
```

## ライセンスと出典

本リポジトリのコード（`dollar_one_recognizer_py/`を含む）は New BSD License
（[LICENSE](LICENSE)）で公開しています。

原典・引用元:

> Wobbrock, J.O., Wilson, A.D. and Li, Y. (2007). Gestures without libraries, toolkits or
> training: A $1 recognizer for user interface prototypes. *Proceedings of the ACM Symposium
> on User Interface Software and Technology (UIST '07)*, pp. 159-168.

本実装はこの論文で示された擬似コードをもとに独自に実装したものですが、参考にした原著の
公式実装（公式サイト）も New BSD License で公開されており、原著作者・著作権者は
本実装とは別に存在します。原著のライセンス表記:

```
This software is distributed under the "New BSD License" agreement:

Copyright (C) 2007-2012, Jacob O. Wobbrock, Andrew D. Wilson
and Yang Li. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
   * Neither the names of the University of Washington nor Microsoft,
     nor the names of its contributors may be used to endorse or promote
     products derived from this software without specific prior written
     permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

公式サイト（アルゴリズム・擬似コード・公式実装）: https://depts.washington.edu/acelab/proj/dollar/index.html
