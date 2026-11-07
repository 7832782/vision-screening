"""检查 index.html 中所有内联 <script> 的语法（跳过 src 引用的文件）。"""
import re
import subprocess
import tempfile
import os

html = open("index.html", encoding="utf-8").read()
blocks = re.findall(r"<script(?:\s[^>]*)?>(.*?)</script>", html, re.S)
failed = 0
for i, js in enumerate(blocks):
    if not js.strip():
        continue
    ext = ".mjs" if "import " in js else ".js"
    path = os.path.join(tempfile.gettempdir(), f"vs_check_{i}{ext}")
    open(path, "w", encoding="utf-8").write(js)
    r = subprocess.run(["node", "--check", path], capture_output=True, text=True)
    if r.returncode:
        failed += 1
        print(f"[FAIL] inline script #{i} ({len(js)} chars)")
        print(r.stderr[-2000:])
    else:
        print(f"[PASS] inline script #{i} ({len(js)} chars)")
print("exit:", failed)
raise SystemExit(failed)
