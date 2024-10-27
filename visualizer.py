import cv2
import numpy as np

def add_alpha_channel(img):
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img

# 이미지와 마스크 합치기
def overlay_transparent(background_img, img_to_overlay, x, y, overlay_size=None):
    # 알파 채널이 있는 배경 이미지를 생성
    bg_img = np.zeros((background_img.shape[0], background_img.shape[1], 4), dtype=np.uint8)
    bg_img[:, :, :3] = background_img
    bg_img[:, :, 3] = 255  # 알파 채널을 255로 설정 (완전 불투명)
    
    if overlay_size is not None:
        img_to_overlay = cv2.resize(img_to_overlay.copy(), overlay_size)

    b, g, r, a = cv2.split(img_to_overlay)
    mask = cv2.medianBlur(a, 5)
    h, w, _ = img_to_overlay.shape
    roi = bg_img[int(y):int(y+h), int(x):int(x+w)]
    
    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(img_to_overlay, img_to_overlay, mask=mask)
    
    bg_img[int(y):int(y+h), int(x):int(x+w)] = cv2.add(img1_bg, img2_fg)

    return bg_img

def cell_clr(cell, clr, mask):
    
    for row in range(300):
        for col in range(600):
            # 투명한 사각형 그리기
            cv2.rectangle(mask, (row+600, col+300), (600, 300), clr, -1) 
            img_output = overlay_transparent(cell, mask, 0, 0)
            
    return img_output
