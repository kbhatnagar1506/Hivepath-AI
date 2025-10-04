#!/bin/bash

echo "🚀 Deploying SwarmAura to Heroku..."

# Check if Heroku CLI is installed
if ! command -v heroku &> /dev/null; then
    echo "❌ Heroku CLI not found. Please install it first:"
    echo "   https://devcenter.heroku.com/articles/heroku-cli"
    exit 1
fi

# Login to Heroku
echo "🔐 Logging into Heroku..."
heroku login

# Create backend app
echo "📦 Creating backend app..."
cd swarmaura
heroku create swarmaura-backend-$(date +%s) --region us

# Add Redis addon
echo "🔴 Adding Redis addon..."
heroku addons:create heroku-redis:mini -a swarmaura-backend-$(date +%s)

# Set environment variables
echo "⚙️  Setting environment variables..."
heroku config:set SERVICE_NAME=routeloom -a swarmaura-backend-$(date +%s)
heroku config:set LOG_LEVEL=INFO -a swarmaura-backend-$(date +%s)
heroku config:set VLM_MODEL=gpt-4o-mini -a swarmaura-backend-$(date +%s)

echo "🔑 Please set your API keys:"
echo "   heroku config:set GOOGLE_MAPS_API_KEY=your_key_here"
echo "   heroku config:set OPENAI_API_KEY=your_key_here"

# Deploy backend
echo "🚀 Deploying backend..."
git subtree push --prefix=swarmaura heroku main

# Create frontend app
echo "📦 Creating frontend app..."
cd ../geographic_intelligence
heroku create swarmaura-frontend-$(date +%s) --region us

# Set frontend environment
echo "⚙️  Setting frontend environment..."
heroku config:set NODE_ENV=production -a swarmaura-frontend-$(date +%s)
heroku config:set VITE_API_URL=https://swarmaura-backend-$(date +%s).herokuapp.com -a swarmaura-frontend-$(date +%s)

# Deploy frontend
echo "🚀 Deploying frontend..."
git subtree push --prefix=geographic_intelligence heroku main

echo "✅ Deployment complete!"
echo "🌐 Backend: https://swarmaura-backend-$(date +%s).herokuapp.com"
echo "🌐 Frontend: https://swarmaura-frontend-$(date +%s).herokuapp.com"
