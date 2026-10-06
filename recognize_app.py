import csv
import os
import tkinter as tk
from typing import Optional

from PIL import Image, ImageTk

from dollar_one_recognizer_py import dollar_one_recognizer
CANVAS_SIZE = 600
THUMB_SIZE = 80
GESTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gestures")

# gestures/ 内の *.csv を全部読み込み、テンプレートの点列と名前のリストを返す
def load_templates() -> tuple[list[list[list[float]]], list[str]]:
    templates: list[list[list[float]]] = []
    names: list[str] = []
    if not os.path.isdir(GESTURES_DIR):
        return templates, names
    for filename in sorted(os.listdir(GESTURES_DIR)):
        if not filename.lower().endswith(".csv"):
            continue
        path = os.path.join(GESTURES_DIR, filename)
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # ヘッダー行 (x,y) を読み飛ばす
            points = [[float(x), float(y)] for x, y in reader]
        if len(points) < 2:
            continue  # 点が1個以下のCSVはテンプレートとして使えないのでスキップ
        templates.append(points)
        names.append(os.path.splitext(filename)[0])  # ファイル名（拡張子抜き）をテンプレート名にする
    return templates, names


# export_gesture_app と同じドラッグ操作で描いたジェスチャーを、
# gestures/ に保存済みのテンプレートと照合して認識するアプリ。
# 左にテンプレート一覧（画像+名前）、右に描画キャンバスを並べる。
class RecognizeApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Gesture Recognizer")

        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- 左: 登録済みテンプレート一覧 ---
        left_container = tk.Frame(main_frame, width=220)
        left_container.pack(side=tk.LEFT, fill=tk.Y)
        left_container.pack_propagate(False)

        tk.Label(left_container, text="Registered Templates", font=("", 11, "bold")).pack(fill=tk.X, pady=4)

        list_canvas = tk.Canvas(left_container, highlightthickness=0)
        scrollbar = tk.Scrollbar(left_container, orient=tk.VERTICAL, command=list_canvas.yview)
        self.list_frame = tk.Frame(list_canvas)
        self.list_frame.bind(
            "<Configure>",
            lambda e: list_canvas.configure(scrollregion=list_canvas.bbox("all")),
        )
        list_canvas.create_window((0, 0), window=self.list_frame, anchor="nw")
        list_canvas.configure(yscrollcommand=scrollbar.set)
        list_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._thumbnails: list[ImageTk.PhotoImage] = []
        self._row_labels: dict[str, tk.Label] = {}

        # --- 右: 描画キャンバス ---
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(right_frame, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.status = tk.Label(right_frame, text="Loading...", font=("", 12))
        self.status.pack(fill=tk.X)

        tk.Button(right_frame, text="Clear", command=self.clear).pack(side=tk.LEFT)

        self.points: list[list[float]] = []
        self._prev_xy: Optional[tuple[float, float]] = None

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        self.recognizer = None
        self.reload_templates()  # 起動時に gestures/ を読み込んでrecognizerを組み立てる

    def reload_templates(self) -> None:
        templates, names = load_templates()

        # 一覧を作り直す前に、既存の行ウィジェットとサムネイル参照を全部捨てる
        for widget in self.list_frame.winfo_children():
            widget.destroy()
        self._thumbnails.clear()
        self._row_labels.clear()

        if not templates:
            # テンプレートが1つも無い場合は認識できないので、その旨を表示して終わる
            self.recognizer = None
            self.status.config(text="No templates in gestures/ (register some with export_gesture_app.py)")
            return

        # 読み込んだ全テンプレートで dollar_one_recognizer を組み立て直す
        self.recognizer = dollar_one_recognizer(size=250, templates=templates, templates_name=names, n=64)
        self.status.config(text=f"Loaded {len(names)} template(s). Draw a gesture on the canvas")

        # テンプレートごとに「サムネイル画像 + 名前」の行を左のリストに追加する
        for name in names:
            row = tk.Frame(self.list_frame)
            row.pack(fill=tk.X, pady=2, padx=2)

            png_path = os.path.join(GESTURES_DIR, name + ".png")
            if os.path.exists(png_path):
                thumb = Image.open(png_path).resize((THUMB_SIZE, THUMB_SIZE))
                photo = ImageTk.PhotoImage(thumb)
                # PhotoImageはどこかで参照を保持しないとGCで消えて表示が消えるため、リストに保持しておく
                self._thumbnails.append(photo)
                tk.Label(row, image=photo).pack(side=tk.LEFT)

            name_label = tk.Label(row, text=name, anchor="w")
            name_label.pack(side=tk.LEFT, padx=4)
            self._row_labels[name] = name_label  # 後で認識結果に応じて太字にするために名前で引けるようにしておく

    def _highlight(self, name: Optional[str]) -> None:
        # 認識結果に一致したテンプレート名だけ太字にし、それ以外は通常表示に戻す
        for label_name, label in self._row_labels.items():
            label.config(font=("", 9, "bold" if label_name == name else "normal"))

    def clear(self) -> None:
        # キャンバスと座標リストをリセットし、ハイライトも解除する
        self.canvas.delete("all")
        self.points = []
        self._prev_xy = None
        self._highlight(None)

    def on_press(self, event: tk.Event) -> None:
        # クリックした瞬間に新しいストロークとして開始する（前回の描画はクリア）
        self.clear()
        self.points = [[event.x, event.y]]
        self._prev_xy = (event.x, event.y)

    def on_drag(self, event: tk.Event) -> None:
        # ドラッグ中は前回の座標から今回の座標まで線を引きつつ、座標を蓄積する
        if self._prev_xy is not None:
            self.canvas.create_line(
                *self._prev_xy, event.x, event.y,
                width=3, fill="black", capstyle=tk.ROUND, smooth=True,
            )
        self.points.append([event.x, event.y])
        self._prev_xy = (event.x, event.y)

    def on_release(self, event: tk.Event) -> None:
        # マウスを離した = ストローク確定のタイミングで認識処理を1回だけ実行する
        if self.recognizer is None or len(self.points) < 2:
            return
        _, name, score = self.recognizer.recognize(self.points)
        self.status.config(text=f"Recognized: {name}  (score={score:.3f})")
        self._highlight(name)


def main() -> None:
    root = tk.Tk()
    RecognizeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
