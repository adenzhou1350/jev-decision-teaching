"""Publish explicitly selected local training artifacts into the demo directory."""
import argparse
from pathlib import Path
import shutil

from .paths import ARTIFACTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sokoban-run',type=Path)
    parser.add_argument('--browser-run',type=Path)
    parser.add_argument('--dest',type=Path,default=ARTIFACTS,
                        help='Artifact root to create, with sokoban/ and browser/ subdirectories')
    parser.add_argument('--include-bundled-sokoban',action='store_true',
                        help='Copy bundled game artifacts alongside a newly trained browser head')
    parser.add_argument('--overwrite',action='store_true')
    args = parser.parse_args()
    if args.include_bundled_sokoban and args.sokoban_run:
        parser.error('Choose --sokoban-run or --include-bundled-sokoban, not both.')
    mapping = []
    if args.sokoban_run:
        mapping += [(args.sokoban_run/a,args.dest/'sokoban'/b) for a,b in (
            ('policy.pt','sokoban-demo.pt'),('levels.json','sokoban-levels.json'),('report.json','sokoban-results.json'))]
    if args.include_bundled_sokoban:
        mapping += [(ARTIFACTS/'sokoban'/name,args.dest/'sokoban'/name) for name in
                    ('sokoban-demo.pt','sokoban-levels.json','sokoban-results.json')]
    if args.browser_run:
        mapping += [(args.browser_run/a,args.dest/'browser'/b) for a,b in (
            ('heads.pt','browser-demo.pt'),('report.json','browser-results.json'))]
    if not mapping:
        parser.error('Choose --sokoban-run, --browser-run and/or --include-bundled-sokoban.')
    for source,target in mapping:
        if not source.is_file():
            parser.error(f'Missing artifact: {source}')
        if source.resolve()==target.resolve():
            parser.error(f'Source and destination are the same: {target}. Use --dest runs/my-demo.')
        if target.exists() and not args.overwrite:
            parser.error(f'{target.name} exists; use --overwrite only for the checkpoint you want to demonstrate.')
    for source,target in mapping:
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        print(f'{source} -> {target}')


if __name__=='__main__':
    main()
