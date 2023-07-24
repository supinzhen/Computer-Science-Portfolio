import cv2
import numpy as np

# 讀入圖片
img = cv2.imread('8.jpg')

# 取得圖片長寬與通道數
rows, cols, ch = img.shape
    
# 算出圖片中心點
cy, cx = int(rows / 2), int(cols/2)

# 設定低頻範圍(由於每張圖片大小不一樣，這裡是以與圖片的row的比例去計算)
f_range_l = 4

# 設定高頻範圍(由於每張圖片大小不一樣，這裡是以與圖片的row的比例去計算)
f_range_h = 37

# 傅立葉轉換後位移低頻至中心點
shifted = np.fft.fftshift(np.fft.fftn(img))

max_amp = np.max(np.abs(shifted))
print(shifted.shape)


# 濾掉高頻，留下低頻範圍
# 設定遮罩，大小和原圖片一致
mask_l = np.zeros((rows, cols, ch), np.uint8)
# 使中心位置，上下左右距離範圍(f_range_l)設置為1
mask_l[cy - f_range_l:cy + f_range_l, cx - f_range_l:cx + f_range_l, :] = 1
# 將傅立葉轉換過的圖片乘上遮罩
low = shifted * mask_l
# 將結果逆轉換
inversed_l = np.fft.ifftn(np.fft.ifftshift(low))
# 取絕對值並改變資料型態
inversed_img_l = np.abs(inversed_l).astype('float32')
           

# 濾掉低頻，留下高頻範圍
# 設定遮罩，大小和原圖片一致
mask_h = np.ones((rows, cols, ch), np.uint8)
# 使中心位置，上下左右距離範圍(f_range_h)設置為0
mask_h[cy - f_range_h:cy + f_range_h, cx - f_range_h:cx + f_range_h, :] = 0
# 將傅立葉轉換過的圖片乘上遮罩
high = shifted * mask_h
# 將結果逆轉換
inversed_h = np.fft.ifftn(np.fft.ifftshift(high))
# 取絕對值並改變資料型態
inversed_img_h = np.abs(inversed_h)
      
# 對低頻原圖、高頻原圖、原圖做權重的加乘
res = cv2.addWeighted(img.astype(np.float32), 0.3, inversed_img_l, 0.7, 1)
res = cv2.addWeighted(res, 0.7, inversed_img_h.astype(np.float32), 0.3, 1)
res = cv2.addWeighted(img.astype(np.float32), 0.4, res, 0.6, 1)

# 對結果做亮度與對比度的調整
contrast = 60
brightness = 30
output = res * (contrast/127 + 1) - contrast + brightness # 轉換公式
# 將調整結果限制在0-255之內
output = np.clip(output, 0, 255)
# 改變資料型態
output = np.uint8(output)

# 顯示留下的低頻頻域
cv2.imshow('shifted', (np.abs(shifted) / max_amp * 255))   
# 顯示留下的低頻頻域
cv2.imshow('FFT 2D low', np.abs(low) / max_amp * 255)
# 顯示剩下低頻的原圖片
cv2.imshow('INVERSE FFT 2D low', (inversed_img_l.astype('uint8')))
# 顯示留下的高頻頻域
cv2.imshow('FFT 2D high', np.abs(high) / max_amp * 255) 
# 顯示剩下高頻的原圖片
cv2.imshow('INVERSE FFT 2D high', inversed_img_h.astype('uint8'))  

# 顯示原圖
cv2.imshow('original', img)
# 顯示美化結果      
cv2.imshow('beautified', output)  
# 儲存結果圖片

cv2.imwrite('shifted.jpg', np.abs(shifted)/ max_amp * 255 * 255)
cv2.imwrite('FT 2D low.jpg', np.abs(low) / max_amp * 255* 255 )  
cv2.imwrite('INVERSE FFT 2D low.jpg', (inversed_img_l.astype('uint8')))  
cv2.imwrite('FFT 2D high.jpg', np.abs(high) / max_amp * 255* 255 )  
cv2.imwrite('INVERSE FFT 2D high.jpg', inversed_img_h.astype('uint8'))  

cv2.waitKey()
cv2.destroyAllWindows()