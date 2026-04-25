# Football Player Potential Analyzer

An AI-powered full-stack web application that predicts a football player's overall rating and position using machine learning, with a production-grade backend featuring caching, authentication, and cloud deployment.

**Live Demo:** [ml-workshop-mini-project.vercel.app](https://ml-workshop-mini-project.vercel.app)  
**API Docs:** [ml-workshop-miniproject.onrender.com/api/docs](https://ml-workshop-miniproject.onrender.com/api/docs) *(available in development mode)*  
**Backend API:** [ml-workshop-miniproject.onrender.com](https://ml-workshop-miniproject.onrender.com)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client (Browser)                         │
│                   Next.js + React + Recharts                    │
│              Deployed on Vercel (ml-workshop-mini-project)      │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTPS REST API
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                            │
│                   Deployed on Render                            │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │   /api/v1   │    │  Rate Limit │    │    JWT Auth         │ │
│  │ predictions │───▶│  slowapi    │───▶│  python-jose        │ │
│  │    auth     │    │ 10req/min   │    │  bcrypt             │ │
│  │   health    │    └─────────────┘    └─────────────────────┘ │
│  └──────┬──────┘                                               │
│         │                                                       │
│    ┌────▼────┐         ┌──────────┐         ┌───────────────┐  │
│    │  Redis  │◀───────▶│ Predict  │────────▶│  ML Pipeline  │  │
│    │  Cache  │  cache  │ Service  │         │  XGBoost      │  │
│    │ (1hr TTL│  hit/   │          │         │  RandomForest │  │
│    │  Keys)  │  miss   └────┬─────┘         │  Scikit-learn │  │
│    └─────────┘              │               └───────────────┘  │
│                             │                                   │
│                    ┌────────▼────────┐                          │
│                    │   PostgreSQL    │                          │
│                    │ prediction_     │                          │
│                    │ history table   │                          │
│                    │ users table     │                          │
│                    └─────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │   GitHub    │
                    │   Actions   │
                    │  CI/CD      │
                    │  Tests →    │
                    │  Deploy     │
                    └─────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 16, React, TypeScript, Tailwind CSS, Recharts, shadcn/ui |
| Backend | FastAPI, Python 3.11 |
| ML | XGBoost, Scikit-learn, RandomForest, Pandas, NumPy |
| Database | PostgreSQL (SQLAlchemy ORM) |
| Cache | Redis (1hr TTL, cache-aside pattern) |
| Auth | JWT (python-jose, bcrypt) |
| Rate Limiting | slowapi (10 req/min per IP) |
| Containerization | Docker, docker-compose |
| CI/CD | GitHub Actions |
| Deployment | Render (API + PostgreSQL + Redis), Vercel (Frontend) |

---

## Features

- **ML Prediction** — Predicts player overall rating and position (FWD/MID/DEF/GK) using a multi-stage XGBoost + RandomForest pipeline trained on FIFA 23 data
- **Similar Players** — Finds top 3 real-world players with similar attribute profiles using Euclidean distance
- **Redis Caching** — Identical prediction requests served from cache, reducing ML inference load
- **Prediction History** — All predictions stored in PostgreSQL with timestamps
- **JWT Authentication** — Register/login endpoints with bcrypt password hashing
- **Rate Limiting** — 10 requests/minute per IP on prediction endpoints
- **Health Check** — `/api/health` endpoint reporting DB and Redis status
- **Swagger Docs** — Auto-generated API documentation via FastAPI
- **CI/CD Pipeline** — GitHub Actions runs tests on every push, deploys to Render only if tests pass
- **Containerized** — Full stack runs locally with `docker-compose up`

---

## ML Pipeline

The prediction uses a 4-stage pipeline:

```
Input Stats
    │
    ▼
Stage 1: GK Classifier (RandomForest)
    Determines: GK or OUTFIELD
    │
    ▼
Stage 2: Group Classifier (XGBoost)
    Determines: FWD / MID / DEF
    │
    ▼
Stage 3: Exact Position Classifier (RandomForest)
    Determines: ST, CAM, CB, LB, etc.
    │
    ▼
Stage 4: Rating Regressor (RandomForest)
    Predicts: Overall rating (0-99)
```

**Model Performance:**
- GK Classifier: ~99% accuracy
- Group Position Classifier (XGBoost): ~93% accuracy
- Exact Position Classifiers: 80-90% accuracy per group
- Rating Regressors: R² > 0.95, MAE < 2.0

---

## API Endpoints

| Method | Endpoint | Description | Auth |
|---|---|---|---|
| GET | `/api/health` | Service health check | None |
| POST | `/api/v1/auth/register` | Register new user | None |
| POST | `/api/v1/auth/login` | Login, returns JWT | None |
| POST | `/api/v1/predictions` | Predict player rating | None |
| GET | `/api/v1/predictions/history` | Last 20 predictions | None |

### Example Request

```bash
curl -X POST https://ml-workshop-miniproject.onrender.com/api/v1/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "Shooting Total": 88,
    "Dribbling Total": 85,
    "Pace Total": 80,
    "Passing Total": 75,
    "Defending Total": 40,
    "Physicality Total": 70
  }'
```

### Example Response

```json
{
  "predicted_rating": 85,
  "predicted_group": "FWD",
  "predicted_exact_position": "ST",
  "similar_players": [
    {
      "Full Name": "K. Benzema",
      "Overall": 91,
      "Best Position": "CF",
      "Club Name": "Real Madrid CF"
    }
  ],
  "ideal_profile": {
    "Pace Total": 79.2,
    "Shooting Total": 82.1
  },
  "cached": false
}
```

---

## Local Development

### Prerequisites
- Docker Desktop
- Node.js 18+
- Python 3.11+

### Running with Docker

```bash
# Clone the repo
git clone https://github.com/kilqwe/ML-workshop-miniProject
cd ML-workshop-miniProject

# Add environment variables
cp backend/.env.example backend/.env
# Edit .env with your values

# Start all services
cd backend
docker-compose up --build
```

Services will be available at:
- API: `http://localhost:8000`
- Swagger Docs: `http://localhost:8000/api/docs`
- Health Check: `http://localhost:8000/api/health`

### Running Frontend Locally

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:3000`

### Environment Variables

```env
DATABASE_URL=postgresql://postgres:password@postgres:5432/football_analyzer
REDIS_URL=redis://redis:6379
SECRET_KEY=your-secret-key-here
DEBUG=True
```

---

## Project Structure

```
ML-workshop-miniProject/
├── backend/
│   ├── app/
│   │   ├── api/v1/routes/
│   │   │   ├── predictions.py   # ML prediction endpoint + caching
│   │   │   ├── auth.py          # JWT register/login
│   │   │   └── health.py        # Health check
│   │   ├── core/
│   │   │   ├── config.py        # Settings + env vars
│   │   │   ├── security.py      # JWT + bcrypt
│   │   │   └── cache.py         # Redis client
│   │   ├── db/
│   │   │   ├── models.py        # SQLAlchemy tables
│   │   │   └── session.py       # DB connection
│   │   ├── services/
│   │   │   └── prediction_service.py  # ML pipeline
│   │   └── main.py              # FastAPI app entry point
│   ├── models/                  # Trained .pkl files (gitignored)
│   ├── data/                    # CSV data (gitignored)
│   ├── tests/
│   │   └── test_api.py          # pytest test suite
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
├── frontend/
│   ├── src/app/
│   │   └── page.tsx             # Main analyzer UI
│   ├── src/components/          # Recharts visualizations
│   └── package.json
└── .github/
    └── workflows/
        └── deploy.yml           # CI/CD pipeline
```

---

## CI/CD Pipeline

Every push to `main` triggers:

```
Push to main
     │
     ▼
GitHub Actions
     │
     ├── Spin up PostgreSQL + Redis services
     ├── Install Python dependencies  
     ├── Run pytest test suite (6 tests)
     │
     ├── Tests pass? ──No──▶ Pipeline stops, no deploy
     │
     └── Tests pass? ──Yes─▶ Trigger Render deployment
                                      │
                                      ▼
                              Live API updated
```

---

## Running Tests

```bash
cd backend
pytest tests/ -v
```

---

*Built with FastAPI, Next.js, and scikit-learn. Trained on FIFA 23 dataset.*