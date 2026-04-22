"""Find classes that were renamed with a numeric suffix and have a base class to merge into."""
import re
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
files = sorted(f for f in os.listdir(".") if f.startswith("test_") and f.endswith(".py"))

total = 0
for f in files:
    with open(f) as fh:
        content = fh.read()
    classes = [
        (m.group(1), m.start())
        for m in re.finditer(r"^class (Test\w+)", content, re.MULTILINE)
    ]
    names = [c[0] for c in classes]
    found = []
    for name, pos in classes:
        m = re.match(r"^(.*?)(\d+)$", name)
        if m:
            base, suffix = m.group(1), m.group(2)
            if base in names and suffix in ("2", "22"):
                line = content[:pos].count("\n") + 1
                found.append((name, base, line))
    if found:
        print(f"\n=== {f} ===")
        for name, base, line in found:
            print(f"  L{line}: {name} -> merge into {base}")
            total += 1

print(f"\nTotal: {total} merges needed")
