try:
    import cv2
    img = cv2.imread("qr-code.png")
    det = cv2.QRCodeDetector()
    data, pts, _ = det.detectAndDecode(img)
    print("解码结果:", repr(data))
except Exception as e:
    print("解码失败:", e)
