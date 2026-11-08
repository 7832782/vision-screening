import math

LEVELS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
DECIMAL = ['0.10', '0.125', '0.16', '0.20', '0.25', '0.32', '0.40', '0.50', '0.63', '0.80', '1.00']


def h_mm(logMAR, dist_m):
    arcmin = 5 * (10 ** logMAR)
    rad = arcmin * math.pi / (180 * 60)
    return math.tan(rad) * dist_m * 1000


def cap_index(dist_m, dpm):
    for i in range(len(LEVELS) - 1, -1, -1):
        if h_mm(LEVELS[i], dist_m) * dpm / 5 >= 1.0:
            return i
    return 0


DEVICES = [
    ("手机 6.1\" 1080p  400ppi", 400),
    ("手机 6.7\" 1440p  515ppi", 515),
    ("平板 iPad 10.9\"  264ppi", 264),
    ("一体机 21.5\" 1080p 102ppi", 102.5),
    ("一体机 21.5\" 4K   205ppi", 205),
    ("一体机 24\"   1080p 92ppi", 92),
]

DISTS = [0.33, 0.40, 0.45, 1.0, 1.5, 2.0, 3.0, 5.0]

head = "| 设备 | " + " | ".join(
    (f"{int(d*100)} cm" if d < 1 else f"{d:g} m") for d in DISTS) + " |"
print(head)
print("|" + "---|" * (len(DISTS) + 1))
for name, ppi in DEVICES:
    dpm = ppi / 25.4
    cells = []
    for d in DISTS:
        i = cap_index(d, dpm)
        cells.append(f"{DECIMAL[i]} (L{LEVELS[i]:.1f})")
    print(f"| {name} | " + " | ".join(cells) + " |")

print()
print("=== 单个视标的设备像素尺寸（高度 / 笔画）===")
for name, ppi in DEVICES:
    dpm = ppi / 25.4
    row = []
    for d in [0.40, 2.0]:
        h = h_mm(0.0, d) * dpm
        row.append(f"{'40cm' if d < 1 else '2m'}: {h:.1f}px / 笔画 {h/5:.2f}px")
    print(f"{name:30s} logMAR 0.0 -> " + "   ".join(row))

print()
print("=== 单调性检查：每个设备 33/40/45cm 的档位索引（应该随距离变大而单调不降）===")
for name, ppi in DEVICES:
    dpm = ppi / 25.4
    idx = [cap_index(d, dpm) for d in (0.33, 0.40, 0.45)]
    ok = idx[0] <= idx[1] <= idx[2]
    print(f"{name:30s} {idx}  单调={'OK' if ok else 'FAIL'}  decimal={[DECIMAL[i] for i in idx]}")

print()
print("=== 102 ppi 21.5\" 在 40cm 的边界 ===")
dpm = 102.5 / 25.4
for i, L in enumerate(LEVELS):
    h = h_mm(L, 0.40)
    print(f"logMAR {L:.1f} -> h={h:.3f} mm, {h*dpm:.2f} dev px, 笔画 {h*dpm/5:.2f} px")
