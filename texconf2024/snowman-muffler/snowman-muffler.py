from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from shapely.geometry import Polygon
import torch
from segmentation_models_pytorch import Unet
from contextlib import redirect_stdout

def render_glyph(character, font_path='SourceHanSerif-Bold.otf', size=512, margin=0):
    font = ImageFont.truetype(font_path, size)
    
    # 仮のキャンバスを作成して文字のバウンディングボックスを取得
    temp_image = Image.new('L', (size*2, size*2), 255)
    draw = ImageDraw.Draw(temp_image)
    bbox = draw.textbbox((margin, margin), character, font=font)
    
    # 実際のキャンバスサイズをバウンディングボックスに基づいて設定
    width, height = bbox[2] - bbox[0], 512
    image = Image.new('L', (width + 2*margin, height + 2*margin), 255)
    draw = ImageDraw.Draw(image)
    
    # Y座標をベースラインに合わせて調整
    baseline_y = margin - bbox[1] + (512 - (bbox[3] - bbox[1])) / 2 # 縦センターに配置する
    draw.text((margin, baseline_y), character, font=font, fill=0)
    
    image_np = np.array(image)  # PIL.Image を NumPy 配列に変換
    resized = cv2.resize(image_np, (512, 512), interpolation=cv2.INTER_AREA)

    return resized

# 推論関数
def predict_and_get_polygon(image, model_path):
    # モデルのロード
    model = Unet(
        encoder_name="resnet34",
        encoder_weights=None,  # 訓練済みモデルの重みをロード
        in_channels=3,
        classes=1
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 画像の前処理
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0

    # 推論
    with torch.no_grad():
        output = model(input_image).squeeze().cpu().numpy()

    # マスクをバイナリ化
    mask = (output > 0.5).astype(np.uint8)

    # 輪郭を取得
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = [tuple(pt[0]) for contour in contours for pt in contour]

    return polygons

def generate_tikz_node(coords):
    conversion_factor = 72.27 / 72 * (200/512) # PostScriptポイントからTeXのptへの変換
    tikz_coords = " -- ".join([f"({x * conversion_factor:.2f}pt,{y * conversion_factor:.2f}pt)" for x, y in coords])
    tikz_code = f"\\clip [shift={{(A.south west)}}, xshift=0mm, yshift=200pt, xscale=1, yscale=-1] (0,0) {tikz_coords} -- cycle;"
    return tikz_code

def main():
    character = "☃"
    font_path = "SourceHanSerif-Bold.otf"
    model_path = 'unet_model.pth'
    latex_pre = r"""
\documentclass[lualatex]{jlreq}
\usepackage[haranoaji,deluxe]{luatexja-preset}
\usepackage{tikz}
\usetikzlibrary{positioning,calc}

\usepackage{xcolor}
\definecolor{red}{rgb}{1, 0.2, 0.2}
\begin{document}

\begin{tikzpicture}
  \node[anchor=base, yshift=-0ex, inner sep=0pt, outer sep=0pt, minimum height=0pt, minimum width=0pt] 
    (A) at (0,0) 
    {\color{black}\rmfamily\bfseries\fontsize{200pt}{200pt}\selectfont ☃};

  \begin{scope}
    """

    latex_post = r"""
  \node[anchor=base, yshift=-0ex, inner sep=0pt, outer sep=0pt, minimum height=0pt, minimum width=0pt] 
      (0,0) 
    {\color{red}\rmfamily\bfseries\fontsize{200pt}{200pt}\selectfont ☃};
  \end{scope}

\end{tikzpicture}

\end{document}
    """

    # グリフのレンダリング
    glyph_image = render_glyph(character, font_path)

    # パーツのセグメンテーション
    polygons = predict_and_get_polygon(image=glyph_image, model_path=model_path)
    tikz_code = generate_tikz_node(polygons)

    
    with open("snowman-muffler.tex", "w") as file:
        with redirect_stdout(file):
            print(latex_pre)
            print(tikz_code)
            print(latex_post)

if __name__ == "__main__":
    main()
