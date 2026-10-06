import csv
import os
import tkinter as tk

from PIL import Image, ImageDraw

CANVAS_SIZE = 600
# 保存先フォルダ。このスクリプトと同じ階層に gestures/ を作る
GESTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gestures")


# 一筆書きジェスチャーをマウスで描いて座標列を取得するだけのお絵描きアプリ。
# dollar_one_recognizer との連携（テンプレートとして読み込む等）は呼び出し側（recognize_app.py）で行う。
class GestureDrawer:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Drawer")

        # ジェスチャーを描くキャンバス本体
        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 状態表示用ラベル（点数や保存結果を表示する）
        self.status = tk.Label(root, text="Drag on the canvas to draw a unistroke gesture")
        self.status.pack(fill=tk.X)

        # クリアボタン・ファイル名入力欄・保存ボタンをまとめた操作バー
        button_frame = tk.Frame(root)
        button_frame.pack(fill=tk.X)
        tk.Button(button_frame, text="Clear", command=self.clear).pack(side=tk.LEFT)

        tk.Label(button_frame, text="Filename:").pack(side=tk.LEFT, padx=(10, 2))
        self.filename_var = tk.StringVar(value="gesture")
        tk.Entry(button_frame, textvariable=self.filename_var, width=20).pack(side=tk.LEFT)
        tk.Button(button_frame, text="Save (CSV+PNG)", command=self.save).pack(side=tk.LEFT, padx=(4, 0))

        # points: 描画中のストロークの座標を逐次ためるリスト
        # last_points: 直前に描き終えたストロークの座標（保存対象）
        self.points = []
        self.last_points = []
        self._prev_xy = None  # 直前にドラッグしたときの座標（線を繋ぐために使う）

        # last_image: 直前のストロークを描いたPillow画像（PNG保存用）
        self._image = None
        self._draw = None
        self.last_image = None
        self._reset_image()

        # マウス操作とイベントハンドラの紐付け
        self.canvas.bind("<ButtonPress-1>", self.on_press)   # クリックでストローク開始
        self.canvas.bind("<B1-Motion>", self.on_drag)         # ドラッグ中は線を追加
        self.canvas.bind("<ButtonRelease-1>", self.on_release)  # 離したらストローク確定

    def _reset_image(self):
        # キャンバスと同じ内容をPNG保存用に複製して描くためのオフスクリーン画像を作り直す
        self._image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "white")
        self._draw = ImageDraw.Draw(self._image)

    def clear(self):
        # 画面・座標リスト・PNG用画像をすべて初期状態に戻す
        self.canvas.delete("all")
        self.points = []
        self._prev_xy = None
        self._reset_image()
        self.status.config(text="Drag on the canvas to draw a unistroke gesture")

    def on_press(self, event):
        # 新しいストロークを描き始めるタイミングなので、まず前回の描画をクリアする
        self.clear()
        self.points = [[event.x, event.y]]
        self._prev_xy = (event.x, event.y)

    def on_drag(self, event):
        if self._prev_xy is not None:
            # キャンバス表示用の線
            self.canvas.create_line(
                *self._prev_xy, event.x, event.y,
                width=3, fill="black", capstyle=tk.ROUND, smooth=True,
            )
            # PNG保存用画像にも同じ線を描く
            self._draw.line(
                [self._prev_xy, (event.x, event.y)],
                fill="black", width=3, joint="curve",
            )
        self.points.append([event.x, event.y])
        self._prev_xy = (event.x, event.y)

    def on_release(self, event):
        # ストローク確定。この時点の座標列・画像を保存対象として控えておく
        self.last_points = self.points[:]
        self.last_image = self._image
        self.status.config(text=f"Captured {len(self.last_points)} points (available via get_last_stroke())")
        print(f"[gesture_app] captured points ({len(self.last_points)}):")
        print(self.last_points)

    def get_last_stroke(self):
        # 直前に描いた一筆書きの生座標列 [[x, y], ...] を返す
        return self.last_points

    def save(self):
        # 描いたストロークが無ければ保存しない
        if not self.last_points or self.last_image is None:
            self.status.config(text="Nothing to save yet. Draw a gesture first")
            return
        name = self.filename_var.get().strip()
        if not name:
            self.status.config(text="Please enter a filename")
            return
        # 拡張子を打っていても打っていなくても、ベース名だけ取り出す
        base, _ = os.path.splitext(name)
        if not base:
            base = name

        os.makedirs(GESTURES_DIR, exist_ok=True)
        csv_path = os.path.join(GESTURES_DIR, base + ".csv")
        png_path = os.path.join(GESTURES_DIR, base + ".png")

        # 座標列はCSVに、見た目はPNGに、同じベース名で保存する
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y"])
            writer.writerows(self.last_points)
        self.last_image.save(png_path)

        self.status.config(text=f"Saved: {base}.csv / {base}.png")
        print(f"[gesture_app] saved: {csv_path} , {png_path}")


def main():
    root = tk.Tk()
    GestureDrawer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
