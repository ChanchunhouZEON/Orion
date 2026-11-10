#!/usr/bin/env python3
# Copyright (c) Chanchunhou. All rights reserved.
# Licensed under the MIT License.
"""Create one subject's bearer credential outside the data root; print no secret."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import secrets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', required=True)
    parser.add_argument('--directory', type=Path, required=True, help='New private directory; refuses to overwrite')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', args.subject):
        parser.error('subject must be 1..64 ASCII letters, digits, underscore or hyphen')
    args.directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    token = secrets.token_urlsafe(32)
    for name, contents in {
        'token.txt': token + '\n',
        'credentials.json': json.dumps([{'subject': args.subject, 'token_sha256': hashlib.sha256(token.encode()).hexdigest()}], indent=2)+'\n',
    }.items():
        with os.fdopen(os.open(args.directory / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as file:
            file.write(contents)
    print(f'Created private token.txt and credentials.json in {args.directory.resolve()}')


if __name__ == '__main__':
    main()
