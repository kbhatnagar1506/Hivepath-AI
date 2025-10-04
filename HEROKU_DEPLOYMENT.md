# 🚀 SwarmAura Heroku Deployment Guide

## Prerequisites

1. **Heroku CLI** installed: https://devcenter.heroku.com/articles/heroku-cli
2. **Git** configured with your GitHub repository
3. **API Keys** ready:
   - Google Maps API Key
   - OpenAI API Key

## Quick Deployment

### Option 1: Automated Script
```bash
./deploy_heroku.sh
```

### Option 2: Manual Deployment

#### Backend Deployment

1. **Create Heroku App:**
```bash
cd swarmaura
heroku create your-swarmaura-backend
```

2. **Add Redis:**
```bash
heroku addons:create heroku-redis:mini
```

3. **Set Environment Variables:**
```bash
heroku config:set SERVICE_NAME=routeloom
heroku config:set LOG_LEVEL=INFO
heroku config:set VLM_MODEL=gpt-4o-mini
heroku config:set GOOGLE_MAPS_API_KEY=your_google_maps_key
heroku config:set OPENAI_API_KEY=your_openai_key
```

4. **Deploy:**
```bash
git subtree push --prefix=swarmaura heroku main
```

#### Frontend Deployment

1. **Create Heroku App:**
```bash
cd geographic_intelligence
heroku create your-swarmaura-frontend
```

2. **Set Environment Variables:**
```bash
heroku config:set NODE_ENV=production
heroku config:set VITE_API_URL=https://your-swarmaura-backend.herokuapp.com
```

3. **Deploy:**
```bash
git subtree push --prefix=geographic_intelligence heroku main
```

## Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │
│   (Svelte)      │◄──►│   (FastAPI)     │
│   Heroku App    │    │   Heroku App    │
└─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Redis         │
                       │   (Heroku Addon)│
                       └─────────────────┘
```

## Features Deployed

### Backend (FastAPI)
- ✅ AI-powered routing optimization
- ✅ ML models for service time prediction
- ✅ Risk assessment and accessibility evaluation
- ✅ Google Maps integration
- ✅ Redis caching
- ✅ RESTful API endpoints

### Frontend (Svelte)
- ✅ 3D geographic visualization
- ✅ Real-time routing dashboard
- ✅ Interactive agent coordination
- ✅ Accessibility evaluation interface
- ✅ Responsive design

## API Endpoints

- `GET /` - Health check
- `POST /api/v1/optimize` - Route optimization
- `POST /api/v1/multi-location` - Multi-location routing
- `GET /api/v1/agents/coordinate` - Agent coordination
- `POST /api/v1/streetscout/analyze` - Street analysis

## Monitoring

- **Logs:** `heroku logs --tail -a your-app-name`
- **Metrics:** Heroku dashboard
- **Redis:** `heroku redis:cli -a your-app-name`

## Troubleshooting

1. **Build Failures:**
   - Check Python/Node.js versions
   - Verify all dependencies in requirements.txt/package.json

2. **Runtime Errors:**
   - Check environment variables
   - Verify API keys are set correctly
   - Check Redis connection

3. **CORS Issues:**
   - Update VITE_API_URL in frontend
   - Add CORS middleware if needed

## Scaling

- **Backend:** `heroku ps:scale web=2 -a your-backend-app`
- **Frontend:** `heroku ps:scale web=2 -a your-frontend-app`
- **Redis:** Upgrade to higher tier if needed

## Security

- All API keys are stored as environment variables
- No secrets in code
- HTTPS enforced by Heroku
- Redis connection encrypted

## Cost Estimation

- **Basic Dyno:** $7/month per app
- **Redis Mini:** $3/month
- **Total:** ~$17/month for both apps + Redis

## Next Steps

1. Set up custom domains
2. Configure CI/CD with GitHub Actions
3. Add monitoring with New Relic or DataDog
4. Set up staging environment
5. Implement database backup strategy
