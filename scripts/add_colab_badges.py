#!/usr/bin/env python3
"""Add Open in Colab badges to all notebooks that are missing one."""

import json
import os
from pathlib import Path

REPO = "LegalIntermediaSL/InteligenciaArtificial"
BRANCH = "main"
NOTEBOOKS_ROOT = Path(__file__).parent.parent / "tutoriales" / "notebooks"

BADGE_TEMPLATE = (
    "[![Open in Colab]"
    "(https://colab.research.google.com/assets/colab-badge.svg)]"
    "(https://colab.research.google.com/github/{repo}/blob/{branch}/{rel_path})"
)


def has_colab_badge(nb: dict) -> bool:
    cells = nb.get("cells", [])
    if not cells:
        return False
    first = cells[0]
    if first.get("cell_type") != "markdown":
        return False
    source = "".join(first.get("source", []))
    return "colab.research.google.com" in source


def add_colab_badge(nb_path: Path) -> bool:
    with open(nb_path) as f:
        nb = json.load(f)

    if has_colab_badge(nb):
        return False

    rel_path = nb_path.relative_to(nb_path.parent.parent.parent.parent)
    badge_text = BADGE_TEMPLATE.format(repo=REPO, branch=BRANCH, rel_path=rel_path)

    badge_cell = {
        "cell_type": "markdown",
        "id": "colab_badge",
        "metadata": {},
        "source": [badge_text],
    }
    nb["cells"].insert(0, badge_cell)

    with open(nb_path, "w") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    return True


def main():
    added = []
    skipped = []

    for nb_path in sorted(NOTEBOOKS_ROOT.rglob("*.ipynb")):
        if add_colab_badge(nb_path):
            added.append(nb_path)
        else:
            skipped.append(nb_path)

    print(f"Badges added: {len(added)}")
    for p in added:
        print(f"  + {p.relative_to(NOTEBOOKS_ROOT.parent.parent)}")

    print(f"\nAlready had badge (skipped): {len(skipped)}")


if __name__ == "__main__":
    main()
