import qrcode
from qrcode.constants import ERROR_CORRECT_M

url = "https://7832782.github.io/vision-screening/"
qr = qrcode.QRCode(version=None, error_correction=ERROR_CORRECT_M, box_size=12, border=3)
qr.add_data(url)
qr.make(fit=True)
img = qr.make_image(fill_color="#0d6b5c", back_color="#ffffff")
img.save("手机打开二维码.png")
print("二维码已生成，尺寸", img.size, "内容", url)
