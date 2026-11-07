import glob, os
import pymupdf

for pdf in sorted(glob.glob("*.pdf")):
    doc = pymupdf.open(pdf)
    png = pdf[:-4] + ".png"
    if len(doc) == 1:
        pix = doc[0].get_pixmap(matrix=pymupdf.Matrix(3, 3))
        pix.save(png)
    else:
        # 多页就纵向拼接
        imgs = [p.get_pixmap(matrix=pymupdf.Matrix(3, 3)) for p in doc]
        total_h = sum(i.height for i in imgs)
        import pymupdf as fitz2
        out = fitz2.Pixmap(imgs[0].width, total_h)
        y = 0
        for i in imgs:
            i.set_origin(0, y)
            out.copy(i, i.irect)
            y += i.height
        out.save(png)
    print(png, os.path.getsize(png))
