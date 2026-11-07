# WHOeyes 2.6.3 逆向分析（算法篇）

来源：WHOeyes 官方 APK（`org.who.whoeyes` 2.6.3，Play 分发，v2/v3 签名 + SourceStamp 已验证），
解包后的 `assets/www/static/js/main.*.chunk.js` 是一个未混淆的 React 生产构建，
核心测试引擎是一个叫 `ta` 的类。以下是逐行读出来后还原的算法。

---

## 1. 档位表（直接来自源码常量）

```js
Y = ["0.10","0.125","0.16","0.20","0.25","0.32","0.40","0.50","0.63","0.80","1.00"]   // 小数视力
Z = [10,9,8,7,6,5,4,3,2,1,0,-1,-2,-3]                                                 // logMAR × 10
$ = ["6/60","6/48","6/38","6/30","6/24","6/19","6/15","6/12","6/9.5","6/7.5","6/6","6/5","6/4","6/3"] // Snellen
```

`Z` 是 14 档（logMAR 1.0 → -0.3），但实际测试循环里被 `Y.length`（11 档，1.0 → 0.0）限制，
`-1/-2/-3` 只在 `getMaxVA()` 里用来判断屏幕能画到多小。

## 2. 视标尺寸公式

```js
calculateA5(distance, e) {
  t = Math.pow(1.26, e) * Math.PI / 180 / 12;
  return Math.tan(t) * (distance + 0.007);
}
```

- `e` = `Z[currentVa]` = logMAR × 10
- `1.26^10 ≈ 10.0`，所以 `1.26^e ≈ 10^logMAR = MAR（弧分）`
- `π/180/12 = π/2160` 正好是 **5 弧分**转弧度 —— 即 logMAR 0 的 E 高度为 5 弧分
- `+0.007`（7 mm）是距离修正项
- 返回值单位是**米**

画到屏幕上：

```js
getPPI()          // 无 Cordova 插件时：sqrt(screenW² + (screenH+statusH)²) / 5 英寸（假设 5 寸屏）
convertInchToPixel(inch) = getPPI() * inch
```

**关键技巧**：`getMaxVA()` 会从最小视标往上找，找到第一个"像素高度是 5 的整数倍"的档位：

```js
if (5 === 5*parseInt(s/5) || 5 === s) { e = t; break; }
```

因为 tumbling E 的笔画宽度 = 高度的 1/5，只有当高度是 5 的整数倍时，每一笔才落在整数像素上，
不会被抗锯齿糊掉。**这是它比一般视力 App 精细的地方**，我们的实现也照搬了。

## 3. 测试流程（与你笔记里观察到的完全一致）

```js
test() {
  currentVa = 档位索引; currentE = 0; currentWrong = 0; currentCorrect = 0;
  drawCanvas(distance, currentVa);  // 画一组（5 个位置的画布）
}
```

每组固定 **5 个视标**，但内部计数逻辑是：

```js
currentE++;
if (currentE < 5 && currentCorrect < 4) { highlight(next); }
else { 过组 / 失败 }
```

- 连续答对 4 个 → **提前结束该组**（所以"全对时每组只出现 4 个"）
- 答错 2 个 → 该组失败（最多只能对 3 个，已不可能达到 4）

**起始档位 logMAR 1.0，每次通过 +2 档（0.2 logMAR）**：
1.0 → 0.8 → 0.6 → 0.4 → 0.2 → 0.0，正好 **6 组**，与你观察到的"总共 6 组"吻合。

## 4. 失败后的回退逻辑（源码逐字翻译）

```js
answer(dir) {
  if (dir !== 当前正确方向) {
    currentWrong++;
    if ((currentVa === 0 && currentWrong > 2) || (currentVa !== 0 && currentWrong > 1)) {
      // 该档失败
      if (currentVa === 0) finalVa = "< 0.1";          // 连最大的都看不清
      if (当前档 !== endVa-3 && 当前档 !== endVa-1) {
        currentVa = max(0, currentVa - 3);             // 回退 3 档（0.3 logMAR）
        test();
      } else {
        finalVa = Y[currentVa - 1];                    // 结算：取上一档
        result();
      }
    }
  } else currentCorrect++;
  ...
}
```

`endVa` = 第一次失败时的档位。翻译成人话：

- **首次失败**：记下 `endVa = F`，退到 `F-3`（即 0.3 logMAR 更大的字）重测
- 在 `F-3` 通过 → 前进 2 档到 `F-1` 再测：
  - `F-1` 通过 → 结果 `F-1`
  - `F-1` 失败 → 结果 `F-2`
- 在 `F-3` 失败 → 结果 `F-4`

由于每次通过固定 +2 档，`F-2` 一定是之前已经通过的档位，所以 `F-3` 这一轮其实是**冗余的**
（我们的实现里省掉了，直接用 `F-1` 做二分，题量更少、结果等价）。

## 5. 拥挤框

论文原文：

> A black bounding box with thickness equal to the arm of the E optotype is used to simulate the
> crowding effect of the ETDRS VA chart, and the space between the optotype and the box is equal
> to half of the optotype size.

即：框线宽 = E 的笔画宽（高度/5），框与 E 的间距 = E 高度的一半。
源码里画布宽度是 `i + 8*s`（i=视标高度，s=笔画宽），与 2×(i/2) + 2×s 基本一致。

⚠️ 但**当视标很大时（如 2 米下的 logMAR 1.0，E 高约 58 mm），拥挤框会超出手机屏幕**，
WHOeyes 的做法是让画布溢出（`overflow:hidden`）把框裁掉。我们的实现改成：先压缩间距、
再等比缩小，保证完整可见。

## 6. 手势映射

```js
swipeleft → "W"; swiperight → "E"; swipeup → "N"; swipedown → "S"
旋转映射：N:90°, W:0°, S:270°, E:180°
```

由旋转角度反推：底图是**开口朝左**的 E，所以 `W`（向左滑）对应 0°。方向 = E 的开口方向。

## 7. 距离

```js
this.distance = (type === "distant") ? 2 : 0.45;
```

即远距 2 米、**近距 0.45 米**（论文写的是 40 cm）。这个差异值得注意：
0.45 m 时 logMAR 0 的 E 比 40 cm 时大 12.5%，结果会略微偏乐观。我们的实现按论文用 40 cm，
并在设置里允许 33/45 切换。

## 8. 其他细节

- 语言：JS 里以 `\uXXXX` 转义存了几十种语言（含中文），所以按原始字节搜中文会搜不到
- 权限极少：INTERNET / RECORD_AUDIO / VIBRATE / MODIFY_AUDIO_SETTINGS / WRITE_EXTERNAL_STORAGE
- 带两个 TFLite 模型（`assets/www/float/050`、`float/075`），Android 版没申请相机权限，
  推测是给 iOS 的实时测距（ADC）复用同一套前端资源留下的
- 结果分级：`Impaired` / `Good` 两档，文案里带"建议找眼科医生"
