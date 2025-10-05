# v0 Integration Guide for SwarmAura API

## 🚀 API Endpoints for v0 Frontend Generation

### Base URL
```
http://localhost:8001
```

## 📊 Available Data Endpoints

### 1. **System Overview**
```javascript
// Get all system data at once
const systemData = await fetch('http://localhost:8001/api/bulk/all')
  .then(res => res.json());

// Get analytics overview
const analytics = await fetch('http://localhost:8001/api/analytics/overview')
  .then(res => res.json());
```

### 2. **Locations Data**
```javascript
// All locations
const locations = await fetch('http://localhost:8001/api/locations')
  .then(res => res.json());

// Locations by type
const stops = await fetch('http://localhost:8001/api/locations/type/stop')
  .then(res => res.json());
```

### 3. **Vehicles Data**
```javascript
// All vehicles
const vehicles = await fetch('http://localhost:8001/api/vehicles')
  .then(res => res.json());

// Vehicles by type
const trucks = await fetch('http://localhost:8001/api/vehicles/type/truck')
  .then(res => res.json());
```

### 4. **AI Predictions**
```javascript
// Service time predictions
const predictions = await fetch('http://localhost:8001/api/predictions/service-times')
  .then(res => res.json());

// Risk assessment
const riskData = await fetch('http://localhost:8001/api/predictions/risk-assessment')
  .then(res => res.json());
```

### 5. **Environmental Data**
```javascript
// Weather data
const weather = await fetch('http://localhost:8001/api/environmental/weather')
  .then(res => res.json());

// Traffic data
const traffic = await fetch('http://localhost:8001/api/environmental/traffic')
  .then(res => res.json());
```

### 6. **Accessibility Data**
```javascript
// Accessibility analysis
const accessibility = await fetch('http://localhost:8001/api/accessibility')
  .then(res => res.json());
```

## 🎨 v0 Prompt Examples

### Example 1: Fleet Management Dashboard
```
Create a modern fleet management dashboard using this data:

API: http://localhost:8001/api/bulk/all

The data includes:
- 6 locations (1 depot + 5 stops)
- 4 vehicles (2 trucks + 2 vans)
- AI predictions for service times
- Real-time analytics
- Accessibility scores

Create a responsive dashboard with:
- Location map with markers
- Vehicle status cards
- Analytics charts
- AI prediction display
- Real-time updates
```

### Example 2: Route Optimization Interface
```
Build a route optimization interface using:

API: http://localhost:8001/api/routes/current

Features needed:
- Interactive map showing routes
- Vehicle assignment display
- Performance metrics
- Real-time optimization
- Mobile responsive design

Use Tailwind CSS and modern UI components.
```

### Example 3: Analytics Dashboard
```
Create an analytics dashboard with:

Data sources:
- http://localhost:8001/api/analytics/overview
- http://localhost:8001/api/predictions/service-times
- http://localhost:8001/api/environmental/weather

Display:
- System metrics cards
- Performance charts
- AI prediction confidence scores
- Weather impact analysis
- Real-time status indicators
```

## 🔧 v0 Integration Code Examples

### React Hook for Data Fetching
```javascript
// Custom hook for SwarmAura data
const useSwarmAuraData = (endpoint) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`http://localhost:8001/api/${endpoint}`)
      .then(response => response.json())
      .then(data => {
        setData(data);
        setLoading(false);
      })
      .catch(error => {
        setError(error);
        setLoading(false);
      });
  }, [endpoint]);

  return { data, loading, error };
};
```

### Component Examples
```javascript
// Location Map Component
const LocationMap = () => {
  const { data: locations, loading } = useSwarmAuraData('locations');
  
  if (loading) return <div>Loading locations...</div>;
  
  return (
    <div className="w-full h-96 bg-gray-100 rounded-lg">
      {/* Map implementation with location markers */}
      {locations?.data?.map(location => (
        <div key={location.id} className="absolute">
          {/* Location marker */}
        </div>
      ))}
    </div>
  );
};

// Vehicle Status Cards
const VehicleCards = () => {
  const { data: vehicles, loading } = useSwarmAuraData('vehicles');
  
  if (loading) return <div>Loading vehicles...</div>;
  
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {vehicles?.data?.map(vehicle => (
        <div key={vehicle.id} className="bg-white p-4 rounded-lg shadow">
          <h3 className="font-semibold">{vehicle.id}</h3>
          <p className="text-gray-600">{vehicle.type}</p>
          <p className="text-sm">Capacity: {vehicle.capacity}</p>
          <div className="mt-2">
            {vehicle.capabilities.map(cap => (
              <span key={cap} className="bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded mr-1">
                {cap}
              </span>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
};

// Analytics Dashboard
const AnalyticsDashboard = () => {
  const { data: analytics } = useSwarmAuraData('analytics/overview');
  const { data: predictions } = useSwarmAuraData('predictions/service-times');
  
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">System Overview</h3>
        <div className="space-y-2">
          <p>Locations: {analytics?.data?.total_locations}</p>
          <p>Vehicles: {analytics?.data?.total_vehicles}</p>
          <p>Demand: {analytics?.data?.total_demand} units</p>
          <p>Utilization: {analytics?.data?.capacity_utilization}%</p>
        </div>
      </div>
      
      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">AI Predictions</h3>
        <div className="space-y-2">
          {predictions?.data?.map(pred => (
            <div key={pred.location_id} className="flex justify-between">
              <span className="text-sm">{pred.location_name}</span>
              <span className="text-sm font-medium">
                {pred.predicted_time}min ({Math.round(pred.confidence * 100)}%)
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
```

## 🎯 v0 Prompt Templates

### Template 1: Complete Dashboard
```
Create a comprehensive fleet management dashboard with:

Data API: http://localhost:8001/api/bulk/all

Requirements:
- Modern, responsive design
- Real-time data updates
- Interactive components
- Mobile-friendly
- Dark/light mode toggle
- Performance optimized

Components needed:
- Header with navigation
- Sidebar with menu
- Main content area
- Location map
- Vehicle cards
- Analytics charts
- AI predictions display
- Status indicators
```

### Template 2: Mobile App Interface
```
Build a mobile-first fleet management app using:

API: http://localhost:8001/api/bulk/all

Mobile features:
- Bottom navigation
- Swipe gestures
- Pull-to-refresh
- Offline support
- Push notifications
- Touch-friendly UI

Screens needed:
- Dashboard overview
- Vehicle list
- Location map
- Analytics
- Settings
```

### Template 3: Admin Panel
```
Create an admin panel for fleet management with:

Data sources:
- http://localhost:8001/api/locations
- http://localhost:8001/api/vehicles
- http://localhost:8001/api/analytics/overview
- http://localhost:8001/api/predictions/service-times

Admin features:
- Data tables with sorting/filtering
- CRUD operations
- Bulk actions
- Export functionality
- User management
- System settings
- Audit logs
```

## 🚀 Quick Start for v0

1. **Copy this prompt to v0:**
```
Create a modern fleet management dashboard using data from http://localhost:8001/api/bulk/all

The API returns:
- 6 locations with coordinates, demand, accessibility scores
- 4 vehicles with capacity, capabilities, costs
- AI predictions for service times
- Real-time analytics and metrics
- Environmental data (weather, traffic)

Build a responsive dashboard with:
- Interactive map showing locations
- Vehicle status cards with capabilities
- Analytics overview with key metrics
- AI prediction display with confidence scores
- Real-time status indicators
- Modern UI with Tailwind CSS
```

2. **Test the integration:**
- Make sure the API is running on http://localhost:8001
- Use the provided React hooks for data fetching
- Implement error handling and loading states

3. **Customize as needed:**
- Add more specific components
- Implement real-time updates
- Add user authentication
- Customize the UI theme

## 📱 API Response Structure

All API responses follow this format:
```json
{
  "status": "success",
  "count": 6,
  "data": [...],
  "timestamp": "2024-01-01T12:00:00"
}
```

Error responses:
```json
{
  "detail": "Error message",
  "status_code": 500
}
```

## 🔗 CORS Support

The API supports CORS from any origin, so v0 can access it directly from the browser.

## 📚 Documentation

- Interactive API docs: http://localhost:8001/docs
- ReDoc documentation: http://localhost:8001/redoc
- Complete endpoint list: http://localhost:8001/
