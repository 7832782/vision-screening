import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# 归档快照里的链接指向已删除的实验目录，跳过
md_files = []
for dirpath, dirnames, filenames in os.walk('.'):
    if any(x in dirpath for x in ('.git', '_gesture_lab', 'node_modules', '实验归档')):
        continue
    for f in filenames:
        if f.endswith('.md'):
            md_files.append(os.path.join(dirpath, f))

link_re = re.compile(r'\[[^\]]*\]\(([^)]+)\)')
bad = 0
total = 0
for path in sorted(md_files):
    with open(path, encoding='utf-8') as fh:
        text = fh.read()
    for target in link_re.findall(text):
        if target.startswith(('http://', 'https://', '#', 'mailto:')):
            continue
        total += 1
        clean = target.split('#')[0].strip()
        if not clean:
            continue
        resolved = os.path.normpath(os.path.join(os.path.dirname(path), clean))
        if not os.path.exists(resolved):
            bad += 1
            print(f'BROKEN  {path}  ->  {target}')
print(f'\nchecked {len(md_files)} markdown files, {total} relative links, {bad} broken')
sys.exit(1 if bad else 0)
