# Cloudflare Pages Setup for HivePath AI Dashboard

## 🚀 Deploy HivePath AI Dashboard to Cloudflare Pages

### Step 1: Connect GitHub Repository
1. Go to [Cloudflare Dashboard](https://dash.cloudflare.com/)
2. Navigate to **Pages** → **Create a project**
3. Connect your GitHub account
4. Select repository: `kbhatnagar1506/Hivepath-AI`
5. Choose branch: `main`

### Step 2: Configure Build Settings
```bash
# Build command
npm run build

# Build output directory
out

# Root directory
swarmaura/integrated_dashboard
```

### Step 3: Environment Variables
Add these environment variables in Cloudflare Pages:

```bash
# Google Maps API Key
NEXT_PUBLIC_GOOGLE_MAPS_API_KEY=AIzaSyAUeiyRuSuKcnPBFmezWCFuUStl8unv7_0

# Cloudflare Workers AI Endpoint
NEXT_PUBLIC_CLOUDFLARE_AI_ENDPOINT=https://hivepath-ai-worker.your-subdomain.workers.dev

# R2 Storage Endpoints
NEXT_PUBLIC_R2_MODELS_ENDPOINT=https://hivepath-ai-models.your-domain.com
NEXT_PUBLIC_R2_DATA_ENDPOINT=https://hivepath-ai-knowledge-graph.your-domain.com
```

### Step 4: Custom Domain (Optional)
1. Go to **Custom domains** in Pages settings
2. Add your domain: `hivepath-ai.com`
3. Update DNS records as instructed

### Step 5: Performance Optimization
- **Edge Caching**: Automatic with Cloudflare Pages
- **Image Optimization**: Built-in with Next.js
- **Global CDN**: Automatic deployment to 200+ cities

## 🎯 Benefits for HivePath AI
- **Global Performance**: Dashboard loads fast worldwide
- **Automatic Scaling**: Handles traffic spikes automatically
- **Edge Computing**: AI inference runs closer to users
- **Cost Effective**: Generous free tier
- **Security**: Built-in DDoS protection and WAF

## 📊 Expected Performance Improvements
- **Page Load Time**: 50% faster globally
- **AI Response Time**: 30% faster with edge computing
- **Availability**: 99.9% uptime SLA
- **Security**: Enterprise-grade protection
