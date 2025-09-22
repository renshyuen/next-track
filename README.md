# Next-Track: Music Recommendation API

Next-Track is a stateless, privacy-respecting music recommendation API. Given a list of Spotify track identifiers (and optional preference parameters), the system returns a single recommended track along with a textual explanation of why it was chosen. 

<aside>
💡

Next-Track does not track users or store personal data. All requests are independent, making the system lightweight and privacy-respecting.

</aside>

# Features

- Content-based recommendations using Spotify audio features
- Stateless design — no tracking, no persistent user profiles
- Explainability module — each recommendation comes with a human-readable rationale
- Recommendation strategies — Weighted Average, Recent Weighted, and Momentum
- API-first architecture with strict schema validation (using `FastAPI` & `Pydantic`)

# Getting Started

## Prerequisites

- Python 3.9+
- Python Virtual Environment (recommended)

<aside>
⚠️

**Important:**

You must provide **Spotify track** **identifiers** as input to the API. These identifiers can be copied directly from the Spotify desktop/mobile app (Right-click track → Share → Copy Song Link) or retrieved via the Spotify Web API.

</aside>

### Example Spotify Track ID

```bash
7MLK2wbiAXLc6xLH6Qbf3p
```

## Installation

Clone the repository and install the required dependencies:

```bash
git clone [copy and paste the web-url from above]
cd next-track
pip install -r requirements.txt
```

# API Usage

## Running the API

Start the `FastAPI` server with `uvicorn`:

```bash
uvicorn application.main:app --reload
```

By default, the server runs locally at:

```bash
http://127.0.0.1:8000
```

`OpenAPI` documentation is automatically available at:

```bash
http://127.0.0.1:8000/docs
```

## How to find Spotify Track ID

1. Download and save the csv dataset [spotify_tracks.csv](https://drive.google.com/file/d/1_PPD5VLmK9NPDrtFegn0fWuPVtCZzpa8/view?usp=sharing)
2. Search for the song track you want to use as input
3. Copy the track id from the dataset

In this example, the ID is:

```bash
0ygAYQQy7bfDN6gH640bbl
```

4. Paste this ID into the JSON body of your API request

## Request Body Example

```bash
{
  "track_ids": [
    "0ygAYQQy7bfDN6gH640bbl",
    "track_id_2",
    "track_id_3"
  ],
  "preferences": {
    "valence": 0.8,
    "energy": 0.7,
    "tempo": 120
  },
  "strategy": "momentum"
}

```

## Response Example

```bash
{
  "recommended_track": {
    "track_id": "5PhgD39pwGrRl2JV6BAYQc",
    "track_title": "Koccintós",
    "artists": "DR BRS, Varga Viktor, Heincz Gábor 'Biga'"
  },
  "explanation": "This upbeat pop track features similar high energy and positive mood, while picking up the pace, following your listening progression, honouring your upbeat preference (strong match).",
  "confidence_score": 0.87,
  "strategy_used": "momentum"
}

```

# Testing

Open up the terminal, activate your Python virtual environment (recommended), and run the following commands:

## Run all tests

```bash
# run complete test suite
python tests/test_runner.py
```

## Run specific test categories

```bash
# unit tests only
python tests/test_runner.py --unit

# integration tests only  
python tests/test_runner.py --integration

# evaluation metrics only
python tests/test_runner.py --evaluation --k-values 1 3 5 10

# performance benchmarks only
python tests/test_runner.py --performance
```

## Using `pytest` directly

```bash
# run specific test file
pytest tests/test_recommender.py -v
```

### Expected output

The test runner generates:

- **Timestamped results folder**: `test_results/YYYYMMDD_HHMMSS/`
- **HTML test reports**: For visual inspection
- **JSON metrics**: For programmatic analysis
- **Markdown report**: Human-readable summary