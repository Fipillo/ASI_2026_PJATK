# ASI_2026_PJATK — NBA Game Winner Prediction

Aplikacja do przewidywania wyniku meczu NBA, skoncentrowany na przygotowaniu cech przedmeczowych i trenowaniu modeli maszynowych.

## O projekcie

Aplikacja wykorzystuje:
- dane meczowe i szczegółowe statystyki zawodników,
- inżynierię cech: średnie ruchome, Elo, różnice statystyk,
- pipeline Kedro dla powtarzalności,
- MLflow do monitorowania przebiegów treningu,
- FastAPI do udostępniania predykcji.

Projekt zawiera:
1. **Badania i analizy**: notebook Jupyter/Google Colab z EDA i modelami AutoGluon
2. **Pipeline produkcyjny**: Kedro dla powtarzalnej inżynierii cech i treningu modelu
3. **Demo i wizualizację**: aplikacja Streamlit do interaktywnych predykcji
4. **API i monitoring**: FastAPI z endpoint'ami predykcji i monitorowaniem driftu danych

## Środowisko

### Szybki start z Conda (zalecane)

Conda jest zalecana na Windows, aby uniknąć problemów z kompilacją.

#### 1. Zainstaluj Minicondę (jeśli jeszcze nie masz)

Pobierz z: https://docs.conda.io/en/latest/miniconda.html

#### 2. Utwórz środowisko

```powershell
# Z katalogu głównego projektu
conda env create -f environment.yml
```

To tworzy:
- Środowisko conda o nazwie `nba-prediction`
- Python 3.10
- Wszystkie pakiety do nauki statystycznej (pandas, numpy, scikit-learn, itp.)
- Narzędzia ML (Kedro, MLflow, Streamlit, XGBoost, LightGBM)
- AutoML (AutoGluon) i FastAPI

#### 3. Aktywuj środowisko

```powershell
conda activate nba-prediction
```

### Alternatywa (venv + pip)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

##
└→ Kedro: node load_games, load_details
```

### Warstwa inżynierii cech

```
Kedro: node build_features
├─ agregacja statystyk drużyn
├─ obliczanie dni odpoczynku
├─ średnie ruchome (5 i 15 meczów)
├─ sezonowy wskaźnik wygranych
├─ ranking Elo (chronologiczny)
└─ różnice cech (gospodarze vs goście)

Output: data/processed/nba_features_pregame.csv
```

### Warstwa treningu modelu

```
Chronologiczny podział Train/Test

Kedro: node train_baseline_model
├─ RandomForest (baseline)
└─ Porównawcze modele (XGBoost, LightGBM) z GridSearchCV

MLflow Tracking:
├─ logowanie parametrów
├─ logowanie wyników walidacji
└─ zapis modeli jako artefakty

Output: models/baseline_random_forest.pkl
```

### Warstwa ewaluacji modelu

```
Kedro: node evaluate_model
├─ Accuracy
├─ Precision / Recall / F1
├─ ROC-AUC
└─ Macierz pomyłek

MLflow Logging: zapis metryk

Output: reports/baseline_metrics.yml
```

### Warstwa prezentacji i deploymentu

```
├─ Aplikacja Streamlit (streamlit_app.py)
│  └─ interaktywne predykcje na przykładowych danych
│
├─ FastAPI + Monitoring (src/api/)
│  ├─ GET /health — sprawdzenie statusu
│  ├─ POST /api/v1/predict — predykcja + logowanie driftu
│  └─ GET /api/v1/drift — status detekcji zmian
│
├─ Drift Detector (src/monitoring/)
│  ├─ porównanie cech z danymi treningowymi
│  └─ ostrzeżenia o zmianach w rozkładzie
│
├─ MLflow UI (mlflow ui)
│  └─ przegląd przebiegów treningowych i porównanie modeli
│
└─ Notebook (notebooks/nba_modeling.ipynb)
   └─ EDA, analiza AutoGluon, wizualizacje
```

## Przepływ pipeline Kedro (DAG)

```
load_games ─┐
            ├─→ build_features ─→ train_baseline_model ─→ evaluate_baseline
load_details┘                  ↓
                               ├─→ train_comparison_models
                               └─→ train_autogluon_model ─→ evaluate_autogluon
```

### Węzły pipeline

1. **load_games_node** — załaduj i przetwórz `games.csv`
2. **load_details_node** — załaduj i przetwórz `games_details.csv`
3. **build_features_node** — inżynieria cech (agregacja, średnie ruchome, Elo)
4. **train_baseline_model_node** — trenowanie modelu RandomForest z podziałem chronologicznym
5. **train_comparison_models_node** — porównanie modeli z GridSearchCV
6. **train_autogluon_model_node** — trenowanie AutoGluon z podziałem chronologicznym
7. **evaluate_baseline_node** — obliczenie metryk dla modelu bazowego
8. **evaluate_autogluon_node** — obliczenie metryk dla AutoGluon

## Konfiguracja

### `src/nba_kedro/conf/parameters.yml`
- `test_size`: stosunek podziału testowego
- `random_state`: ziarno losowe
- `baseline_rf_params`: parametry Random Forest
- `autogluon_time_limit`: limit czasu treningu AutoGluon
- `autogluon_presets`: preset jakości AutoGluon
- `mlflow_tracking_uri`: lokalizacja serwera MLflow
- `mlflow_experiment_name`: nazwa przebiegu treningowego

### `src/nba_kedro/conf/catalog.yml`
- definicje źródeł i celów danych
- ścieżki plików dla wejść i wyjść
- formaty serializacji (CSV, PKL, YAML)

## Kluczowe założenia projektowe

### 1. Chronologiczny podział Train/Test
- model trenuje się na starszych meczach, a testuje na nowszych
- zapobiega to wyciekowi danych
- odzwierciedla rzeczywisty scenariusz wdrożenia

### 2. Przetwarzanie cech przed treningiem
- wszystkie cechy są obliczane przed treningiem
- przesunięcia czasowe zapobiegają wyciekowi informacji
- wyniki są ujednolicone dla wszystkich przebiegów treningowych

### 3. Śledzenie przebiegów treningowych MLflow
- każdy trening jest logowany parametry i metryki
- umożliwia porównywanie przebiegów treningowych i reprodukcję
- lokalne ustawienie MLflow może być przy kolejnych etapach rozbudowy skalowane

### 4. Modularność węzłów
- każdy węzeł pipeline to funkcja z oddzielnym zadaniem
- łatwiej testować i utrzymywać kod
- węzły można ponownie użyć w innych pipeline'ach

### 5. Monitoring driftu danych
- detekuje zmiany w rozkładzie cech wejściowych
- porównuje bieżące dane z referencyjnymi statystykami treningowymi
- loguje ostrzeżenia o anomaliach w każdej predykcji
- endpoint `GET /api/v1/drift` umożliwia przegląd stanu monitoringu

### 6. Automatyzacja MLOps (opcje B)
- CI z GitHub Actions jest wdrożone w `.github/workflows/ci.yml`
- Continuous Training jest wdrożone w `.github/workflows/continuous-training.yml`
- proces CD nie jest obecnie realizowany, bo projekt ma charakter lokalny/demo
- Monitoring driftu danych jest zintegrowany z API

## Przykład przepływu danych

```
Games:
  2003-10-05: HOME_TEAM_ID=1 vs VISITOR_TEAM_ID=2 → HOME_TEAM_WINS=1

Details:
  GAME_ID=1, TEAM_ID=1: PTS=95, AST=22, REB=42, ...
  GAME_ID=1, TEAM_ID=2: PTS=88, AST=20, REB=38, ...

Wygenerowane cechy:
  HOME_REST_DAYS=7            (pierwszy mecz sezonu)
  HOME_LAST_GAME_WON=0.5      (brak poprzedniego meczu, wartość domyślna)
  HOME_SEASON_WIN_RATE=0.5    (0 wygranych / 0 meczów przed)
  HOME_ELO=1500               (początkowy ranking)
  VISITOR_ELO=1500
  ... (średnie ruchome, różnice, itp.)
```

## Dane

W katalogu `data/raw/` powinny się znaleźć pliki źródłowe:
- `games.csv`
- `games_details.csv`

W katalogu `data/processed/` zapisywane są dane przetworzone i cechy.

## Struktura projektu

```
.
├── conf/                    # konfiguracja projektu
├── data/                    # dane wejściowe i przetworzone
├── docs/                    # dokumentacja
├── models/                  # zapisane modele
├── notebooks/               # analiza i badania
├── reports/                 # metryki i wyniki
├── scripts/                 # pomocnicze narzędzia
├── src/
│   ├── api/                 # FastAPI endpointy
│   ├── nba_kedro/           # pipeline Kedro
│   ├── monitoring/          # monitoring driftu danych
│   ├── features.py          # funkcje inżynierii cech
│   ├── pipeline.py          # prosty runner
│   ├── train.py             # trening modelu
│   └── model_loader.py      # ładowanie i cachowanie modelu
├── streamlit_app.py         # aplikacja Streamlit
├── environment.yml          # środowisko Conda
├── requirements.txt         # zależności pip
└── README.md                # dokumentacja główna
```

### Struktura Kedro

```
src/nba_kedro/
├── nodes/
│   ├── __init__.py
│   ├── data_loading.py        # ładowanie surowych danych
│   ├── feature_engineering.py # tworzenie cech
│   ├── model_training.py      # trenowanie modeli
│   └── model_evaluation.py    # ewaluacja modeli
├── conf/
│   ├── parameters.yml         # parametry pipeline
│   └── catalog.yml            # katalog danych
├── pipeline.py                # definicja DAG
├── runner.py                  # wykonanie pipeline
└── README.md                  # dokumentacja pipeline
```

## Kluczowe cechy

- obliczenia dni odpoczynku i momentum,
- średnie ruchome 5- i 15-meczowe,
- sezonowy wskaźnik wygranych,
- ranking Elo,
- cechy różnicowe między gospodarzami a gośćmi.

## Modele

Najważniejsze modele:
- Random Forest (baseline),
- XGBoost,
- LightGBM,
- AutoGluon (podejście analityczne).

Model bazowy powinien być zapisany w katalogu `models/`.

## Technologie

- Python
- pandas, numpy
- scikit-learn
- xgboost, lightgbm, catboost
- kedro
- mlflow
- streamlit
- fastapi, uvicorn
- autogluon

## Wymagania

Zainstaluj zależności z `requirements.txt` lub `environment.yml`.

