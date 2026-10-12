# test_install.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import re
import subprocess

import pytest

from pathlib import Path

from tangermeme._skills import install as install_mod
from tangermeme._skills.install import _bundled_skill_dir
from tangermeme._skills.install import _default_dest
from tangermeme._skills.install import install_skill


###
# Helpers


def _ref_mentions(text):
	"""Return the basenames of every backticked reference path in `text`.

	The skill writes cross-references as backticked paths, ``references/x.md``,
	rather than as Markdown links, because nothing that reads a skill renders
	Markdown and a link spends twice the characters on the same path.
	"""

	return [m.split("/")[-1]
		for m in re.findall(r'`(references/[\w.-]+\.md)`', text)]


def _seed_fake_skill(root):
	"""Create a minimal but well-formed skill directory at `root`."""

	(root / "references").mkdir(parents=True)
	(root / "SKILL.md").write_text("---\nname: x\ndescription: y\n---\n\nbody\n")
	(root / "references" / "foo.md").write_text("# foo\n")
	return root


###
# Bundled-skill integrity (run against the data that actually ships)


def test_bundled_skill_dir_is_well_formed():
	data = _bundled_skill_dir()
	assert data.is_dir()
	assert (data / "SKILL.md").is_file()

	refs = list((data / "references").glob("*.md"))
	assert len(refs) > 0


def test_skill_frontmatter():
	text = (_bundled_skill_dir() / "SKILL.md").read_text()
	assert text.startswith("---\n")

	frontmatter = text.split("---\n", 2)[1]
	fields = dict(re.findall(r'^(\w+):\s*(.*)$', frontmatter, flags=re.MULTILINE))

	assert fields.get("name") == "tangermeme"
	# 1024 is the spec maximum for a description, and the limit enforced by
	# skill-creator's quick_validate.py. This asserted 1536 until 2026-09-10,
	# so a description between the two passed here and failed validation.
	assert 0 < len(fields.get("description", "")) <= 1024


def test_internal_references_all_resolve():
	data = _bundled_skill_dir()
	refs = data / "references"
	existing = {p.name for p in refs.glob("*.md")}

	broken = []
	for f in [data / "SKILL.md", *refs.glob("*.md")]:
		for target in _ref_mentions(f.read_text()):
			if target not in existing:
				broken.append("{} -> {}".format(f.name, target))

	assert broken == [], "broken internal references: {}".format(broken)


def test_no_markdown_links_to_reference_files():
	# Backticked paths only. A Markdown link would also make the resolution and
	# orphan checks above pass vacuously, since neither one looks for links.
	data = _bundled_skill_dir()

	found = []
	for f in [data / "SKILL.md", *(data / "references").glob("*.md")]:
		for m in re.findall(r'\[[^\]]*\]\([^)]*\.md\)', f.read_text()):
			found.append("{}: {}".format(f.name, m))

	assert found == [], "use `references/x.md`, not a link: {}".format(found)


def test_every_reference_is_linked_from_the_router():
	data = _bundled_skill_dir()
	refs = data / "references"

	routed = set(_ref_mentions((data / "SKILL.md").read_text()))
	orphans = sorted(p.name for p in refs.glob("*.md") if p.name not in routed)

	assert orphans == [], "not reachable from SKILL.md: {}".format(orphans)


###
# install_skill behavior (always into a temporary destination)


def test_install_into_fresh_dest(tmp_path):
	dest = tmp_path / "tangermeme"
	result = install_skill(dest=dest)

	assert result == dest
	assert (dest / "SKILL.md").is_file()

	bundled = {p.name for p in (_bundled_skill_dir() / "references").glob("*.md")}
	installed = {p.name for p in (dest / "references").glob("*.md")}
	assert installed == bundled


def test_install_existing_dest_raises_without_force(tmp_path):
	dest = tmp_path / "tangermeme"
	install_skill(dest=dest)

	with pytest.raises(FileExistsError):
		install_skill(dest=dest)


def test_install_force_replaces_stale_content(tmp_path):
	dest = tmp_path / "tangermeme"
	install_skill(dest=dest)

	junk = dest / "references" / "stale.md"
	junk.write_text("remove me")

	install_skill(dest=dest, force=True)

	assert not junk.exists()
	assert (dest / "SKILL.md").is_file()


def test_install_missing_bundle_raises(tmp_path, monkeypatch):
	empty = tmp_path / "empty"
	empty.mkdir()
	monkeypatch.setattr(install_mod, "_bundled_skill_dir", lambda: empty)

	with pytest.raises(FileNotFoundError):
		install_skill(dest=tmp_path / "tangermeme")


def test_install_excludes_checkpoints_and_pycache(tmp_path, monkeypatch):
	src = _seed_fake_skill(tmp_path / "src")
	(src / ".ipynb_checkpoints").mkdir()
	(src / ".ipynb_checkpoints" / "SKILL-checkpoint.md").write_text("nope")
	(src / "__pycache__").mkdir()
	(src / "__pycache__" / "x.pyc").write_text("nope")
	monkeypatch.setattr(install_mod, "_bundled_skill_dir", lambda: src)

	dest = install_skill(dest=tmp_path / "tangermeme")

	assert not (dest / ".ipynb_checkpoints").exists()
	assert not (dest / "__pycache__").exists()
	assert (dest / "references" / "foo.md").is_file()


def test_default_dest_is_under_home_and_isolated(tmp_path, monkeypatch):
	# Patch home so the default-destination path never touches the real ~/.claude.
	monkeypatch.setattr(Path, "home", lambda: tmp_path)

	result = install_skill()

	assert result == _default_dest()
	assert result == tmp_path / ".claude" / "skills" / "tangermeme"
	assert (result / "SKILL.md").is_file()


###
# Console-script entry point (opt-in; does not install anything)


@pytest.mark.cmd
def test_console_script_print_path():
	result = subprocess.run(["tangermeme-install-skills", "--print-path"],
		capture_output=True, text=True)

	assert result.returncode == 0
	assert result.stdout.strip().endswith("data")
