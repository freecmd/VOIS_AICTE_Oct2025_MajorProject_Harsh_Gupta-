from pathlib import Path
import sys

try:
    import pandas as pd
    import numpy as np
except Exception as e:
    print("Required packages not installed. Please run:\n    pip install -r requirements.txt")
    raise

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_CSV = BASE_DIR / 'Netflix Dataset.csv'
OUT_DIR = BASE_DIR / 'data'
OUT_DIR.mkdir(parents=True, exist_ok=True)
CLEANED_CSV = OUT_DIR / 'netflix_cleaned.csv'

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    if 'date_added' in df.columns:
        df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
        df['year_added'] = df['date_added'].dt.year
    return df


def clean_duration(df: pd.DataFrame) -> pd.DataFrame:
    # Create numeric duration for movies and seasons for TV shows
    if 'duration' in df.columns and 'type' in df.columns:
        # movies: '90 min' -> 90
        df['duration_minutes'] = df.apply(
            lambda r: int(r['duration'].split()[0]) if pd.notna(r['duration']) and r['type'].strip().lower()=='movie' else np.nan,
            axis=1
        )
        # tv shows: '3 Seasons' -> 3
        def parse_seasons(x):
            try:
                if pd.isna(x):
                    return np.nan
                parts = x.split()
                return int(parts[0])
            except:
                return np.nan
        df['seasons'] = df.apply(lambda r: parse_seasons(r['duration']) if pd.notna(r['duration']) and r['type'].strip().lower()=='tv show' else np.nan, axis=1)
    return df


def explode_genres(df: pd.DataFrame) -> pd.DataFrame:
    if 'listed_in' in df.columns:
        df['listed_in'] = df['listed_in'].fillna('Unknown')
        df['genre_list'] = df['listed_in'].apply(lambda s: [g.strip() for g in s.split(',')])
        df = df.explode('genre_list').rename(columns={'genre_list':'genre'})
        df['genre'] = df['genre'].fillna('Unknown')
    return df


def normalize_countries(df: pd.DataFrame) -> pd.DataFrame:
    if 'country' in df.columns:
        df['country'] = df['country'].fillna('Unknown')
        df['country_list'] = df['country'].apply(lambda s: [c.strip() for c in s.split(',')] if pd.notna(s) else ['Unknown'])
        df = df.explode('country_list').rename(columns={'country_list':'country_normalized'})
    return df


def main():
    if not RAW_CSV.exists():
        print(f"Raw CSV not found at {RAW_CSV}. Make sure 'Netflix Dataset.csv' is in the project root.")
        sys.exit(1)

    df = pd.read_csv(RAW_CSV)
    print('Loaded rows:', len(df))

    df = normalize_column_names(df)
    df = parse_dates(df)
    df = clean_duration(df)
    df = explode_genres(df)
    df = normalize_countries(df)

    # Basic dedupe on title + release_year
    if 'title' in df.columns and 'release_year' in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=['title', 'release_year'])
        print(f'Dropped duplicates: {before - len(df)}')

    df.to_csv(CLEANED_CSV, index=False)
    print('Cleaned CSV written to', CLEANED_CSV)


if __name__ == '__main__':
    main()
