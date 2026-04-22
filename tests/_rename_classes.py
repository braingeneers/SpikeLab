"""Rename EdgeCase/ErrorPath test classes (rename only, no merges)."""
import re
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def process_file(filename):
    with open(filename) as f:
        content = f.read()
    all_classes = re.findall(r"^class (Test\w+)", content, re.MULTILINE)
    bad_names = [n for n in all_classes if "EdgeCase" in n or "ErrorPath" in n]
    if not bad_names:
        return 0
    used_names = set(all_classes)
    rename_map = {}
    for name in bad_names:
        base = name.replace("EdgeCases", "").replace("EdgeCase", "").replace("ErrorPaths", "")
        target = base
        if target in used_names and target not in bad_names:
            suffix = 2
            while f"{base}{suffix}" in used_names:
                suffix += 1
            target = f"{base}{suffix}"
        rename_map[name] = target
        used_names.add(target)
    for old_name, new_name in rename_map.items():
        if old_name != new_name:
            content = content.replace(f"class {old_name}:", f"class {new_name}:")
            content = content.replace(f"# {old_name}", f"# {new_name}")
    with open(filename, "w") as f:
        f.write(content)
    return len(rename_map)

files = sorted(f for f in os.listdir(".") if f.startswith("test_") and f.endswith(".py"))
total = 0
for f in files:
    count = process_file(f)
    if count > 0:
        print(f"{f}: {count} renames")
        total += count
print(f"Total: {total} renames")
