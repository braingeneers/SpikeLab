"""Merge test classes with numeric suffixes into their base classes.

Uses ast to find class boundaries, then does a single bottom-to-top pass
of line-level edits per file. No intermediate writes or re-parses.
"""
import ast
import re
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def get_class_boundaries(source):
    """Return {class_name: (start_0idx, end_0idx_exclusive)} for top-level Test* classes."""
    tree = ast.parse(source)
    lines = source.split("\n")
    total = len(lines)

    test_classes = [
        n for n in ast.iter_child_nodes(tree)
        if isinstance(n, ast.ClassDef) and n.name.startswith("Test")
    ]
    all_top = sorted(
        (n for n in ast.iter_child_nodes(tree) if hasattr(n, "lineno")),
        key=lambda n: n.lineno,
    )

    boundaries = {}
    for cls in test_classes:
        start = cls.lineno - 1  # 0-indexed

        idx = next(j for j, n in enumerate(all_top) if n is cls)
        if idx + 1 < len(all_top):
            raw_end = all_top[idx + 1].lineno - 1  # 0-indexed start of next node
        else:
            raw_end = total

        # Strip trailing blank lines
        end = raw_end
        while end > start + 1 and lines[end - 1].strip() == "":
            end -= 1

        boundaries[cls.name] = (start, end)  # [start, end) in 0-indexed lines

    return boundaries


def extract_methods(lines, start, end):
    """Return lines[first_method_or_decorator : end] from a class body."""
    for i in range(start + 1, end):
        stripped = lines[i].strip()
        if stripped.startswith("def ") or stripped.startswith("@"):
            result = lines[i:end]
            # Strip trailing blank lines
            while result and result[-1].strip() == "":
                result.pop()
            return result
    return []


def find_section_header_start(lines, class_start):
    """Walk backwards from class_start to find section header comments/blanks."""
    i = class_start
    while i > 0:
        prev = lines[i - 1].strip()
        if prev == "" or prev.startswith("# ===") or prev.startswith("# ---"):
            i -= 1
        else:
            break
    return i


def process_file(filename):
    with open(filename) as f:
        source = f.read()
    lines = source.split("\n")

    try:
        boundaries = get_class_boundaries(source)
    except SyntaxError:
        print(f"  SKIP {filename}: syntax error")
        return 0

    # Find merge candidates
    merges = []
    for name, (start, end) in boundaries.items():
        m = re.match(r"^(.*?)(2|22)$", name)
        if m:
            base = m.group(1)
            if base in boundaries and base != name:
                merges.append((name, base, start, end))

    if not merges:
        return 0

    # Sort by start line descending — process from bottom to top
    merges.sort(key=lambda x: x[2], reverse=True)

    count = 0
    for src_name, tgt_name, src_start, src_end in merges:
        tgt_start, tgt_end = boundaries[tgt_name]

        methods = extract_methods(lines, src_start, src_end)

        # Delete source class (including section headers above it)
        del_start = find_section_header_start(lines, src_start)
        # Also consume trailing blank lines after the class
        del_end = src_end
        while del_end < len(lines) and lines[del_end].strip() == "":
            del_end += 1

        deleted_count = del_end - del_start
        del lines[del_start:del_end]

        # Adjust target boundaries if deletion was before target
        if del_start < tgt_start:
            tgt_start -= deleted_count
            tgt_end -= deleted_count
        elif del_start < tgt_end:
            tgt_end -= deleted_count

        # Insert methods at end of target class
        if methods:
            insert_block = [""] + methods
            for k, bline in enumerate(insert_block):
                lines.insert(tgt_end + k, bline)
            adjustment = len(insert_block)
        else:
            adjustment = 0

        # Update boundaries for subsequent merges
        # Rebuild from scratch since offsets are complex
        try:
            boundaries = get_class_boundaries("\n".join(lines))
        except SyntaxError:
            print(f"  ERROR in {filename} after merging {src_name} -> {tgt_name}")
            print(f"  Writing partial result for debugging")
            with open(filename, "w") as f:
                f.write("\n".join(lines))
            return count

        count += 1

    with open(filename, "w") as f:
        f.write("\n".join(lines))

    return count


if __name__ == "__main__":
    files = sorted(
        f for f in os.listdir(".")
        if f.startswith("test_") and f.endswith(".py")
    )
    total = 0
    for f in files:
        count = process_file(f)
        if count > 0:
            print(f"  {f}: {count} merged")
            total += count
    print(f"\nTotal: {total} classes merged")
