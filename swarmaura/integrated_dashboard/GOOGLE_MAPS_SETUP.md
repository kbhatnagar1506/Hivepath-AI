# Google Maps Embed API Setup Guide

## 🗺️ Setting up Google Maps Integration

This guide will help you set up the Google Maps Embed API for the HivePath AI dashboard.

### 1. Get a Google Maps API Key

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the **Maps Embed API**:
   - Go to "APIs & Services" > "Library"
   - Search for "Maps Embed API"
   - Click "Enable"

### 2. Create API Credentials

1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "API Key"
3. Copy your API key
4. (Optional) Restrict the API key to your domain for security

### 3. Configure Environment Variables

Create a `.env.local` file in your dashboard directory:

```bash
# Google Maps API Key
NEXT_PUBLIC_GOOGLE_MAPS_API_KEY=your_api_key_here
```

### 4. API Key Restrictions (Recommended)

For security, restrict your API key:

1. Go to "APIs & Services" > "Credentials"
2. Click on your API key
3. Under "Application restrictions":
   - Select "HTTP referrers (web sites)"
   - Add your domain: `localhost:3000/*`
   - Add your production domain: `yourdomain.com/*`
4. Under "API restrictions":
   - Select "Restrict key"
   - Choose "Maps Embed API"

### 5. Features Available

The Google Maps integration includes:

- **Real Google Maps** with street view, satellite, and hybrid modes
- **Interactive controls** for zoom, map type, traffic, and transit
- **Location markers** with risk-based coloring
- **Route visualization** with vehicle assignments
- **Click-to-navigate** to Google Maps for detailed directions
- **Real-time data overlay** with AI predictions

### 6. Map Modes

- **Roadmap**: Standard street map view
- **Satellite**: Aerial imagery
- **Hybrid**: Satellite with street labels
- **Terrain**: Topographical view

### 7. Interactive Features

- **Zoom Controls**: +/- buttons for map zoom
- **Layer Toggles**: Traffic and transit information
- **Location Details**: Click markers for detailed information
- **External Navigation**: Links to full Google Maps experience

### 8. Troubleshooting

**Map not loading?**
- Check your API key is correct
- Ensure Maps Embed API is enabled
- Verify API key restrictions allow your domain

**Locations not showing?**
- Check that locations have valid lat/lng coordinates
- Verify the map center is set correctly
- Ensure locations are within the map bounds

### 9. Cost Considerations

- Maps Embed API has generous free tier limits
- First 28,000 map loads per month are free
- Additional loads cost $7 per 1,000 requests
- Monitor usage in Google Cloud Console

### 10. Production Deployment

For production deployment:

1. Update API key restrictions to include your production domain
2. Set up billing alerts in Google Cloud Console
3. Monitor API usage and costs
4. Consider implementing caching for frequently accessed maps

## 🚀 Ready to Go!

Once configured, your HivePath AI dashboard will have a fully interactive Google Maps integration with real street data, satellite imagery, and all the power of Google Maps!
