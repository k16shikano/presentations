import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from shapely.geometry import Polygon

# 1. グリフのレンダリング
def render_glyph(character, font_path='NotoSansCJKjp-Black.otf', size=512, margin=0):
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
    
    # 文字を描画
    draw.text((margin, baseline_y), character, font=font, fill=0)
    return np.array(image)

# 2. 画像処理によるパーツのセグメンテーション
def segment_parts(image):
    # 画像の前処理（しきい値処理と輪郭検出）
    _, binary = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    parts = []
    for contour in contours:
        # バウンディングボックスの計算
        x, y, w, h = cv2.boundingRect(contour)
        print(f"Bounding box - x: {x}, y: {y}, w: {w}, h: {h}")  # デバッグ用
        part = image[y:y+h, x:x+w]  # 各パーツを抽出
        parts.append((contour, (x, y, w, h)))  # 輪郭とバウンディングボックスを保存
        
        # 各パーツに矩形を描画（デバッグ用）
        cv2.rectangle(image, (x, y), (x+w, y+h), (128,), 2)

    return parts, image

# 3. 重なりのない領域のポリゴンを計算して描画
def compute_non_overlapping_polygon(parts, n, image):
    # 画像をカラーに変換
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # 基準ボックスのポリゴン
    reference_box = parts[n][1]
    ref_poly = Polygon([
        (reference_box[0], reference_box[1]),
        (reference_box[0] + reference_box[2], reference_box[1]),
        (reference_box[0] + reference_box[2], reference_box[1] + reference_box[3]),
        (reference_box[0], reference_box[1] + reference_box[3])
    ])
    
    # 他のパーツの領域を除外
    for i, (contour, box) in enumerate(parts):
        if i == n:
            continue
        # パーツの輪郭をポリゴンとして扱う
        other_poly = Polygon([(p[0][0], p[0][1]) for p in contour])
        ref_poly = ref_poly.difference(other_poly)
    
    # 残った領域のポリゴンの頂点を返す
    if not ref_poly.is_empty:
        coords = list(ref_poly.exterior.coords)
        
        # 多角形の頂点を赤線で結ぶ
        for i in range(len(coords) - 1):
            cv2.line(color_image, (int(coords[i][0]), int(coords[i][1])), (int(coords[i+1][0]), int(coords[i+1][1])), (0, 0, 255), 2)
        cv2.line(color_image, (int(coords[-1][0]), int(coords[-1][1])), (int(coords[0][0]), int(coords[0][1])), (0, 0, 255), 2)

        return coords, color_image
    else:
        return [], color_image

def generate_tikz_node(coords):
    conversion_factor = 72.27 / 72 * (200/512) # PostScriptポイントからTeXのptへの変換。200ptで使う場合
    tikz_coords = " -- ".join([f"({x * conversion_factor:.2f}pt,{y * conversion_factor:.2f}pt)" for x, y in coords])
    tikz_code = f"\\clip [shift={{(A.south west)}}, xshift=0mm, yshift=200pt, xscale=1, yscale=-1] (0,0) {tikz_coords} -- cycle;"
    return tikz_code

# 4. 全体の処理の流れ
def main():
    character = "彩"  # 漢字を指定
    font_path = "NotoSerifCJKjp-Black.otf"

    # グリフのレンダリング
    glyph_image = render_glyph(character, font_path)

    # パーツのセグメンテーション
    parts, segmented_image = segment_parts(glyph_image)

    # n番目のバウンディングボックスと他のパーツの重なりを除外
    print(len(parts))
    n = 0
    if parts:
        non_overlapping_polygon, result_image = compute_non_overlapping_polygon(parts, n, segmented_image)
        tikz_code = generate_tikz_node(non_overlapping_polygon)
        print("Generated TikZ Code:")
        print(tikz_code)
    else:
        print("No parts detected.")

    # デバッグ: 画像全体のサイズと保存前の確認
    print(f"Segmented image size: {segmented_image.shape}")
    cv2.imwrite('segmented_glyph_with_polygon.png', result_image)

    # 結果の表示
    plt.imshow(result_image)
    plt.title('Segmented Glyph with Polygon')
    plt.show()

if __name__ == "__main__":
    main()
