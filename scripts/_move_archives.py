import shutil, pathlib

root=pathlib.Path(__file__).resolve().parents[1]
archive=root/'archive'/'20251214'
archive.mkdir(parents=True, exist_ok=True)

c1=root/'cleanup'/'archived_cleanup_20251214'
c2=root/'cleanup'/'1765662996'
items=[]
if c1.exists():
    items.append(c1)
if c2.exists():
    items.append(c2)
arch_dir=root/'models'/'archived'
if arch_dir.exists():
    for it in arch_dir.iterdir():
        items.append(it)

for it in items:
    dest=archive/it.name
    print('moving', it, '->', dest)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(it), str(dest))

# try removing empty parent dirs
for p in [root/'cleanup', root/'models'/'archived']:
    try:
        if p.exists() and not any(p.iterdir()):
            p.rmdir()
            print('removed empty', p)
    except Exception as e:
        print('err removing', p, e)

print('done')
