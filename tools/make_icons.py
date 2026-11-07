from PIL import Image, ImageDraw

BG = (13, 107, 92, 255)      # #0d6b5c
WHITE = (255, 255, 255, 255)
E_POLY = [(0, 0), (5, 0), (5, 1), (1, 1), (1, 2), (4, 2), (4, 3), (1, 3), (1, 4), (5, 4), (5, 5), (0, 5)]


def make_icon(size, path, radius_ratio=0.22, e_ratio=0.56):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    r = int(size * radius_ratio)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=r, fill=BG)

    e_size = size * e_ratio
    ox = (size - e_size) / 2
    oy = (size - e_size) / 2
    pts = [(ox + x * e_size / 5, oy + y * e_size / 5) for x, y in E_POLY]
    d.polygon(pts, fill=WHITE)
    img.save(path)
    print(f"{path} {size}x{size}")


make_icon(192, "icon-192.png")
make_icon(512, "icon-512.png")
make_icon(180, "apple-touch-icon.png", radius_ratio=0.0, e_ratio=0.58)
