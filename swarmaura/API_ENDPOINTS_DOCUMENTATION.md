# SwarmAura Data Extraction API Endpoints

## 🚀 API Base URL
```
http://localhost:8001
```

## 📊 Complete API Endpoints for Frontend Data Extraction

### 🏠 **Root Endpoint**
```
GET /
```
**Description**: API information and available endpoints
**Response**: List of all available endpoints

---

### 📍 **Location Data Endpoints**

#### Get All Locations
```
GET /api/locations
```
**Description**: Get all locations with complete data
**Response**:
```json
{
  "status": "success",
  "count": 6,
  "data": [
    {
      "id": "D",
      "name": "Downtown Boston Depot",
      "type": "depot",
      "lat": 42.3601,
      "lng": -71.0589,
      "demand": 0,
      "priority": 1,
      "access_score": 0.95,
      "features": [],
      "hazards": [],
      "weather_risk": 0.10,
      "traffic_risk": 0.20,
      "crime_risk": 0.30,
      "congestion_score": 0.40,
      "sidewalk_width": 2.5,
      "curb_cuts": 4,
      "parking_spaces": 50,
      "loading_docks": 3,
      "lighting_score": 0.90
    }
  ],
  "timestamp": "2024-01-01T12:00:00"
}
```

#### Get Location by ID
```
GET /api/locations/{location_id}
```
**Description**: Get specific location by ID
**Parameters**: `location_id` (string) - Location ID
**Response**: Single location object

#### Get Locations by Type
```
GET /api/locations/type/{location_type}
```
**Description**: Get locations by type (depot, stop, etc.)
**Parameters**: `location_type` (string) - Type filter
**Response**: Array of locations matching type

---

### 🚛 **Vehicle Data Endpoints**

#### Get All Vehicles
```
GET /api/vehicles
```
**Description**: Get all vehicles with complete data
**Response**:
```json
{
  "status": "success",
  "count": 4,
  "data": [
    {
      "id": "V1",
      "type": "truck",
      "capacity": 400,
      "capabilities": ["lift_gate", "refrigeration", "hazmat"],
      "max_weight": 2000,
      "max_volume": 10,
      "fuel_type": "diesel",
      "year": 2020,
      "avg_speed_kmph": 40,
      "fuel_efficiency": 8.5,
      "hourly_rate": 25.0
    }
  ],
  "timestamp": "2024-01-01T12:00:00"
}
```

#### Get Vehicle by ID
```
GET /api/vehicles/{vehicle_id}
```
**Description**: Get specific vehicle by ID
**Parameters**: `vehicle_id` (string) - Vehicle ID
**Response**: Single vehicle object

#### Get Vehicles by Type
```
GET /api/vehicles/type/{vehicle_type}
```
**Description**: Get vehicles by type (truck, van, etc.)
**Parameters**: `vehicle_type` (string) - Type filter
**Response**: Array of vehicles matching type

---

### 🛣️ **Routing Data Endpoints**

#### Optimize Routes
```
POST /api/routes/optimize
```
**Description**: Optimize routes with given parameters
**Request Body**:
```json
{
  "depot": {
    "id": "D",
    "name": "Depot",
    "lat": 42.3601,
    "lng": -71.0589
  },
  "stops": [
    {
      "id": "S1",
      "name": "Stop 1",
      "lat": 42.3700,
      "lng": -71.0500,
      "demand": 100,
      "priority": 1
    }
  ],
  "vehicles": [
    {
      "id": "V1",
      "type": "truck",
      "capacity": 200
    }
  ],
  "time_limit_sec": 10,
  "drop_penalty_per_priority": 2000,
  "use_access_scores": true
}
```
**Response**:
```json
{
  "status": "success",
  "data": {
    "routes": [
      {
        "vehicle_id": "V1",
        "stops": ["D", "S1", "D"],
        "distance_km": 5.2,
        "time_min": 12.5,
        "load": 100
      }
    ],
    "summary": {
      "total_distance_km": 5.2,
      "total_time_min": 12.5,
      "served_stops": 1,
      "served_rate": 1.0
    }
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

#### Get Current Routes
```
GET /api/routes/current
```
**Description**: Get current active routes using system data
**Response**: Same as optimize routes response

---

### 📊 **Analytics Endpoints**

#### Get Analytics Overview
```
GET /api/analytics/overview
```
**Description**: Get system analytics overview
**Response**:
```json
{
  "status": "success",
  "data": {
    "total_locations": 6,
    "total_vehicles": 4,
    "total_demand": 725,
    "total_capacity": 1100,
    "capacity_utilization": 65.9,
    "system_status": "operational",
    "last_updated": "2024-01-01T12:00:00"
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

#### Get Performance Metrics
```
GET /api/analytics/performance
```
**Description**: Get performance metrics
**Response**:
```json
{
  "status": "success",
  "data": {
    "solve_time": 0.245,
    "total_routes": 4,
    "total_distance": 15.12,
    "total_time": 45.5,
    "served_stops": 5,
    "served_rate": 1.0,
    "vehicle_efficiency": 100.0,
    "average_route_length": 2.2
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

---

### 🧠 **AI Predictions Endpoints**

#### Get Service Time Predictions
```
GET /api/predictions/service-times
```
**Description**: Get AI-powered service time predictions
**Response**:
```json
{
  "status": "success",
  "count": 5,
  "data": [
    {
      "location_id": "S_A",
      "location_name": "Back Bay Station",
      "predicted_time": 14.3,
      "historical_avg": 8.6,
      "confidence": 0.33,
      "model_type": "gnn",
      "factors": {
        "demand": 150,
        "access_score": 0.72,
        "weather_risk": 0.20,
        "traffic_risk": 0.40,
        "peak_hour": 1.0
      }
    }
  ],
  "timestamp": "2024-01-01T12:00:00"
}
```

#### Get Risk Assessment
```
GET /api/predictions/risk-assessment
```
**Description**: Get AI-powered risk assessment for all routes
**Response**:
```json
{
  "status": "success",
  "count": 30,
  "data": [
    {
      "src_id": "S_A",
      "src_name": "Back Bay Station",
      "dst_id": "S_B",
      "dst_name": "North End",
      "base_time": 15.8,
      "risk_multiplier": 0.330,
      "adjusted_time": 21.0,
      "time_increase": 5.2,
      "risk_level": "high"
    }
  ],
  "statistics": {
    "average_risk": 0.313,
    "max_risk": 0.330,
    "high_risk_routes": 27
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

---

### 🌤️ **Environmental Data Endpoints**

#### Get Weather Data
```
GET /api/environmental/weather
```
**Description**: Get current weather conditions
**Response**:
```json
{
  "status": "success",
  "data": {
    "temperature": -2.5,
    "condition": "partly_cloudy",
    "humidity": 65,
    "wind_speed": 12,
    "visibility": 15,
    "precipitation": 0.1,
    "pressure": 1013.25,
    "uv_index": 0,
    "timestamp": "2024-01-01T12:00:00"
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

#### Get Traffic Data
```
GET /api/environmental/traffic
```
**Description**: Get current traffic conditions
**Response**:
```json
{
  "status": "success",
  "data": {
    "overall_congestion": 0.30,
    "incidents": 2,
    "construction_zones": 1,
    "rush_hour_multiplier": 1.4,
    "average_speed": 30,
    "timestamp": "2024-01-01T12:00:00"
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

---

### ♿ **Accessibility Data Endpoints**

#### Get Accessibility Data
```
GET /api/accessibility
```
**Description**: Get accessibility data for all locations
**Response**:
```json
{
  "status": "success",
  "count": 5,
  "data": [
    {
      "location_id": "S_A",
      "base_score": 72,
      "feature_bonus": 25,
      "sidewalk_bonus": 4.0,
      "curb_bonus": 4,
      "hazard_penalty": 10,
      "final_score": 95,
      "features": ["elevator", "ramp"],
      "hazards": ["construction_zone"],
      "sidewalk_width": 2.0,
      "curb_cuts": 2,
      "parking_spaces": 20,
      "loading_docks": 1,
      "lighting_score": 0.70,
      "accessibility_level": "excellent"
    }
  ],
  "statistics": {
    "average_score": 85.3,
    "excellent_locations": 3,
    "good_locations": 1,
    "poor_locations": 1
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

---

### 🏥 **Health Check Endpoint**

#### System Health Check
```
GET /api/health
```
**Description**: Check system health and service availability
**Response**:
```json
{
  "status": "healthy",
  "backend_available": true,
  "timestamp": "2024-01-01T12:00:00",
  "services": {
    "unified_data_system": true,
    "service_time_predictor": true,
    "risk_shaper": true,
    "warmstart_clusterer": true
  }
}
```

---

### 📦 **Bulk Data Endpoint**

#### Get All Data
```
GET /api/bulk/all
```
**Description**: Get all data in one request (locations, vehicles, environmental, accessibility, predictions)
**Response**:
```json
{
  "status": "success",
  "data": {
    "locations": [...],
    "vehicles": [...],
    "environmental": {...},
    "accessibility": [...],
    "service_predictions": [...],
    "system_info": {
      "total_locations": 6,
      "total_vehicles": 4,
      "total_demand": 725,
      "total_capacity": 1100
    }
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

---

## 🚀 **Frontend Integration Examples**

### JavaScript/Fetch Examples

```javascript
// Get all locations
const locations = await fetch('http://localhost:8001/api/locations')
  .then(response => response.json());

// Get vehicles by type
const trucks = await fetch('http://localhost:8001/api/vehicles/type/truck')
  .then(response => response.json());

// Optimize routes
const routeResult = await fetch('http://localhost:8001/api/routes/optimize', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    depot: { id: 'D', name: 'Depot', lat: 42.3601, lng: -71.0589 },
    stops: [...],
    vehicles: [...]
  })
}).then(response => response.json());

// Get analytics overview
const analytics = await fetch('http://localhost:8001/api/analytics/overview')
  .then(response => response.json());

// Get all data at once
const allData = await fetch('http://localhost:8001/api/bulk/all')
  .then(response => response.json());
```

### React/React Native Examples

```jsx
// React hook for data fetching
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

// Usage
const { data: locations, loading } = useSwarmAuraData('locations');
const { data: analytics } = useSwarmAuraData('analytics/overview');
```

---

## 🔧 **Error Handling**

All endpoints return consistent error responses:

```json
{
  "detail": "Error message",
  "status_code": 500
}
```

Common status codes:
- `200`: Success
- `404`: Not found
- `500`: Internal server error
- `503`: Service unavailable (backend not available)

---

## 🌐 **CORS Support**

All endpoints support CORS for frontend integration from any origin.

---

## 📚 **API Documentation**

Interactive API documentation available at:
- Swagger UI: `http://localhost:8001/docs`
- ReDoc: `http://localhost:8001/redoc`
