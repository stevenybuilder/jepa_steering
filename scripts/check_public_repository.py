"""Check the published file set and Markdown navigation, excluding private drafts."""
from pathlib import Path
import posixpath
import re
import subprocess
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r'!?\[[^\]]*\]\(([^\s)]+)(?:\s+"[^"]*")?\)')
MANUSCRIPT_HOSTS = ('arxiv.org', 'openreview.net', 'aclanthology.org', 'jmlr.org',
                    'proceedings.mlr.press', 'papers.nips.cc', 'proceedings.neurips.cc')


def check(root=ROOT):
    files = set(subprocess.check_output(
        ['git', 'ls-files', '-z'], cwd=root).decode().rstrip('\0').split('\0'))
    errors = []
    links = 0
    for name in sorted(files):
        path = Path(name)
        if name.startswith('paper/') and not name.startswith('paper/data/'):
            errors.append(f'Manuscript directory is public: {name}')
        if path.suffix.lower() in ('.tex', '.bib', '.sty', '.log', '.out', '.err'):
            errors.append(f'Private draft or log is public: {name}')
        if path.suffix.lower() == '.pdf' and not name.startswith('docs/figures/'):
            errors.append(f'PDF outside the figure assets: {name}')
        if path.suffix.lower() != '.md':
            continue
        for match in LINK.finditer((root / name).read_text()):
            target = match.group(1).strip('<>')
            parsed = urlsplit(target)
            links += 1
            if parsed.scheme or parsed.netloc:
                if any(parsed.netloc == h or parsed.netloc.endswith('.' + h)
                       for h in MANUSCRIPT_HOSTS):
                    errors.append(f'Manuscript link in {name}: {target}')
                if parsed.path.lower().endswith('.pdf'):
                    errors.append(f'External PDF link in {name}: {target}')
                continue
            if not parsed.path:
                continue
            resolved = posixpath.normpath(str(path.parent / unquote(parsed.path)))
            if resolved not in files and not any(
                    f.startswith(resolved.rstrip('/') + '/') for f in files):
                errors.append(f'Unpublished link target in {name}: {target}')
    if errors:
        raise AssertionError('\n'.join(errors))
    return {'tracked_files': len(files), 'markdown_links': links}


if __name__ == '__main__':
    counts = check()
    print(f"PASS: {counts['tracked_files']} public paths; {counts['markdown_links']} "
          'Markdown links; no manuscripts, external paper links, or log files')
