from pathlib import Path
import pandas as pd
import argparse


def main(path: str | None = None) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    default = repo_root / 'data' / 'processed' / 'games_details_sample.csv'
    p = Path(path) if path else default
    if not p.exists():
        print(f'missing: {p}')
        raise SystemExit(1)

    df = pd.read_csv(p, usecols=['TEAM_ID', 'TEAM_CITY', 'TEAM_ABBREVIATION'])
    df = df.drop_duplicates()
    df['TEAM_NAME'] = df['TEAM_CITY'].fillna('') + ' ' + df['TEAM_ABBREVIATION'].fillna('')
    for tid, name in sorted({int(row['TEAM_ID']): row['TEAM_NAME'] for _, row in df.iterrows()}.items()):
        print(tid, name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Print TEAM_ID -> TEAM_NAME mapping from sample CSV')
    parser.add_argument('csv', nargs='?', help='Optional path to games_details_sample.csv')
    args = parser.parse_args()
    main(args.csv)
