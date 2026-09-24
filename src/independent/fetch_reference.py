"""Fetch and hash the author's published reproduction fixture, never ATP test data."""
import hashlib
import json
import urllib.request
from pathlib import Path

from src.util import DATA, utcnow, write_json

REPO = 'martiningram/tennis_bayes_point_based'
OUT = DATA / 'independent_ingram'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / 'MANIFEST.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for f in manifest['files']:
            assert hashlib.sha256((OUT / f['name']).read_bytes()).hexdigest() == f['sha256']
        print('Author reference hashes verified', flush=True)
        return
    req = urllib.request.Request(f'https://api.github.com/repos/{REPO}/commits/master',
                                 headers={'User-Agent': 'independent-tennis-reproduction'})
    with urllib.request.urlopen(req) as response:
        sha = json.load(response)['sha']
    files = []
    for name in ['dataset.csv', 'stan_model.stan', 'bayes_point_model.py', 'winning_prob.py', 'README.md']:
        url = f'https://raw.githubusercontent.com/{REPO}/{sha}/{name}'
        with urllib.request.urlopen(url) as response:
            data = response.read()
        (OUT / name).write_bytes(data)
        files.append({'name': name, 'url': url, 'sha256': hashlib.sha256(data).hexdigest()})
        print(name, len(data), flush=True)
    write_json(manifest_path, {'utc': utcnow(), 'repo': REPO, 'sha': sha, 'files': files,
                              'purpose': 'Published author fixture, separate from Sackmann corpus; attribution Martin Ingram. No 2025 data.'})


if __name__ == '__main__':
    main()
