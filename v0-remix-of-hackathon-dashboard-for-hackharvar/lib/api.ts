const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001"

export interface Location {
  id: string
  name: string
  type: "depot" | "stop"
  lat: number
  lng: number
  demand: number
  priority: number
  access_score: number
  service_time_base: number
  weather_risk: number
  traffic_risk: number
  crime_risk: number
  lighting_score: number
  congestion_score: number
  accessibility_features: string[]
  parking_spaces: number
  loading_docks: number
  ev_charging: boolean
  traffic_signals: number
  streetlights: number
  sidewalk_width: number
  curb_cuts: number
  hazards: string[]
  time_windows: {
    open: string
    close: string
  }
  special_requirements: string[]
  historical_service_times: number[]
  peak_hours: number[]
  weather_impact: {
    rain: number
    snow: number
    clear: number
  }
  traffic_patterns: {
    morning: number
    afternoon: number
    evening: number
    night: number
  }
}

export interface Vehicle {
  id: string
  type: "truck" | "van"
  capacity: number
  max_weight: number
  dimensions: {
    length: number
    width: number
    height: number
  }
  capabilities: string[]
  fuel_type: string
  efficiency: number
  driver_skill: number
  maintenance_status: string
  cost_per_km: number
  cost_per_hour: number
  availability: {
    start: string
    end: string
  }
  rest_requirements: {
    max_hours: number
    break_interval: number
  }
}

export interface Prediction {
  location_id: string
  location_name: string
  predicted_time: number
  historical_avg: number
  confidence: number
  model_type: string
  factors: {
    demand: number
    access_score: number
    weather_risk: number
    traffic_risk: number
    peak_hour: number
  }
}

export interface Analytics {
  total_locations: number
  total_vehicles: number
  total_demand: number
  total_capacity: number
  capacity_utilization: number
  system_status?: string
  last_updated?: string
}

const MOCK_LOCATIONS: Location[] = [
  {
    id: "D",
    name: "Downtown Boston Depot",
    type: "depot",
    lat: 42.3601,
    lng: -71.0589,
    demand: 0,
    priority: 1,
    access_score: 0.95,
    service_time_base: 5.0,
    weather_risk: 0.1,
    traffic_risk: 0.2,
    crime_risk: 0.3,
    lighting_score: 0.9,
    congestion_score: 0.4,
    accessibility_features: ["elevator", "ramp", "wide_doors"],
    parking_spaces: 50,
    loading_docks: 3,
    ev_charging: true,
    traffic_signals: 2,
    streetlights: 15,
    sidewalk_width: 2.5,
    curb_cuts: 4,
    hazards: [],
    time_windows: { open: "06:00", close: "22:00" },
    special_requirements: [],
    historical_service_times: [4.5, 5.2, 4.8, 5.1, 4.9],
    peak_hours: [8, 17, 18],
    weather_impact: { rain: 1.2, snow: 1.5, clear: 1.0 },
    traffic_patterns: { morning: 1.3, afternoon: 1.1, evening: 1.4, night: 0.8 },
  },
  {
    id: "S_A",
    name: "Back Bay Station",
    type: "stop",
    lat: 42.37,
    lng: -71.05,
    demand: 150,
    priority: 2,
    access_score: 0.72,
    service_time_base: 8.5,
    weather_risk: 0.2,
    traffic_risk: 0.4,
    crime_risk: 0.3,
    lighting_score: 0.7,
    congestion_score: 0.6,
    accessibility_features: ["elevator", "ramp"],
    parking_spaces: 20,
    loading_docks: 1,
    ev_charging: true,
    traffic_signals: 3,
    streetlights: 12,
    sidewalk_width: 2.0,
    curb_cuts: 2,
    hazards: ["construction_zone"],
    time_windows: { open: "07:00", close: "19:00" },
    special_requirements: ["lift_gate"],
    historical_service_times: [8.2, 9.1, 8.5, 8.8, 8.3],
    peak_hours: [8, 12, 17],
    weather_impact: { rain: 1.3, snow: 1.6, clear: 1.0 },
    traffic_patterns: { morning: 1.4, afternoon: 1.2, evening: 1.5, night: 0.9 },
  },
  {
    id: "S_B",
    name: "North End",
    type: "stop",
    lat: 42.34,
    lng: -71.1,
    demand: 140,
    priority: 1,
    access_score: 0.61,
    service_time_base: 12.0,
    weather_risk: 0.3,
    traffic_risk: 0.5,
    crime_risk: 0.4,
    lighting_score: 0.6,
    congestion_score: 0.7,
    accessibility_features: ["ramp"],
    parking_spaces: 8,
    loading_docks: 0,
    ev_charging: false,
    traffic_signals: 1,
    streetlights: 8,
    sidewalk_width: 1.5,
    curb_cuts: 1,
    hazards: ["narrow_streets", "pedestrian_heavy"],
    time_windows: { open: "08:00", close: "18:00" },
    special_requirements: ["small_vehicle"],
    historical_service_times: [11.5, 12.8, 11.9, 12.3, 12.1],
    peak_hours: [9, 13, 18],
    weather_impact: { rain: 1.4, snow: 1.8, clear: 1.0 },
    traffic_patterns: { morning: 1.5, afternoon: 1.3, evening: 1.6, night: 0.7 },
  },
  {
    id: "S_C",
    name: "Harvard Square",
    type: "stop",
    lat: 42.39,
    lng: -71.02,
    demand: 145,
    priority: 2,
    access_score: 0.55,
    service_time_base: 10.5,
    weather_risk: 0.2,
    traffic_risk: 0.3,
    crime_risk: 0.2,
    lighting_score: 0.8,
    congestion_score: 0.5,
    accessibility_features: ["elevator", "ramp", "wide_doors"],
    parking_spaces: 30,
    loading_docks: 2,
    ev_charging: true,
    traffic_signals: 4,
    streetlights: 18,
    sidewalk_width: 3.0,
    curb_cuts: 6,
    hazards: [],
    time_windows: { open: "06:30", close: "21:00" },
    special_requirements: [],
    historical_service_times: [10.2, 11.1, 10.6, 10.9, 10.4],
    peak_hours: [7, 11, 16],
    weather_impact: { rain: 1.2, snow: 1.4, clear: 1.0 },
    traffic_patterns: { morning: 1.2, afternoon: 1.1, evening: 1.3, night: 0.8 },
  },
  {
    id: "S_D",
    name: "Beacon Hill",
    type: "stop",
    lat: 42.33,
    lng: -71.06,
    demand: 150,
    priority: 1,
    access_score: 0.65,
    service_time_base: 9.0,
    weather_risk: 0.2,
    traffic_risk: 0.4,
    crime_risk: 0.3,
    lighting_score: 0.7,
    congestion_score: 0.6,
    accessibility_features: ["ramp"],
    parking_spaces: 15,
    loading_docks: 1,
    ev_charging: false,
    traffic_signals: 2,
    streetlights: 10,
    sidewalk_width: 2.2,
    curb_cuts: 3,
    hazards: ["steep_hills"],
    time_windows: { open: "08:00", close: "20:00" },
    special_requirements: ["low_clearance"],
    historical_service_times: [8.8, 9.5, 9.1, 9.3, 8.9],
    peak_hours: [8, 14, 19],
    weather_impact: { rain: 1.3, snow: 1.5, clear: 1.0 },
    traffic_patterns: { morning: 1.4, afternoon: 1.2, evening: 1.5, night: 0.8 },
  },
  {
    id: "S_E",
    name: "South End",
    type: "stop",
    lat: 42.41,
    lng: -71.03,
    demand: 140,
    priority: 2,
    access_score: 0.7,
    service_time_base: 7.5,
    weather_risk: 0.1,
    traffic_risk: 0.2,
    crime_risk: 0.2,
    lighting_score: 0.9,
    congestion_score: 0.3,
    accessibility_features: ["elevator", "ramp", "wide_doors"],
    parking_spaces: 25,
    loading_docks: 2,
    ev_charging: true,
    traffic_signals: 3,
    streetlights: 16,
    sidewalk_width: 2.8,
    curb_cuts: 5,
    hazards: [],
    time_windows: { open: "07:00", close: "22:00" },
    special_requirements: [],
    historical_service_times: [7.2, 8.1, 7.6, 7.9, 7.4],
    peak_hours: [7, 12, 17],
    weather_impact: { rain: 1.1, snow: 1.3, clear: 1.0 },
    traffic_patterns: { morning: 1.1, afternoon: 1.0, evening: 1.2, night: 0.9 },
  },
]

const MOCK_VEHICLES: Vehicle[] = [
  {
    id: "V1",
    type: "truck",
    capacity: 400,
    max_weight: 2000,
    dimensions: { length: 6.0, width: 2.5, height: 3.0 },
    capabilities: ["lift_gate", "refrigeration", "hazmat"],
    fuel_type: "diesel",
    efficiency: 0.8,
    driver_skill: 0.9,
    maintenance_status: "excellent",
    cost_per_km: 0.15,
    cost_per_hour: 25.0,
    availability: { start: "06:00", end: "22:00" },
    rest_requirements: { max_hours: 10, break_interval: 4 },
  },
  {
    id: "V2",
    type: "van",
    capacity: 200,
    max_weight: 1000,
    dimensions: { length: 4.5, width: 2.0, height: 2.5 },
    capabilities: ["lift_gate"],
    fuel_type: "gasoline",
    efficiency: 0.9,
    driver_skill: 0.8,
    maintenance_status: "good",
    cost_per_km: 0.12,
    cost_per_hour: 20.0,
    availability: { start: "07:00", end: "21:00" },
    rest_requirements: { max_hours: 8, break_interval: 3 },
  },
  {
    id: "V3",
    type: "truck",
    capacity: 350,
    max_weight: 1800,
    dimensions: { length: 5.5, width: 2.3, height: 2.8 },
    capabilities: ["lift_gate", "refrigeration"],
    fuel_type: "diesel",
    efficiency: 0.85,
    driver_skill: 0.85,
    maintenance_status: "good",
    cost_per_km: 0.14,
    cost_per_hour: 23.0,
    availability: { start: "06:30", end: "21:30" },
    rest_requirements: { max_hours: 9, break_interval: 4 },
  },
  {
    id: "V4",
    type: "van",
    capacity: 150,
    max_weight: 800,
    dimensions: { length: 4.0, width: 1.8, height: 2.2 },
    capabilities: [],
    fuel_type: "electric",
    efficiency: 0.95,
    driver_skill: 0.75,
    maintenance_status: "excellent",
    cost_per_km: 0.08,
    cost_per_hour: 18.0,
    availability: { start: "08:00", end: "20:00" },
    rest_requirements: { max_hours: 7, break_interval: 3 },
  },
]

const MOCK_ANALYTICS: Analytics = {
  total_locations: 6,
  total_vehicles: 4,
  total_demand: 725,
  total_capacity: 1100,
  capacity_utilization: 65.91,
  system_status: "operational",
}

const MOCK_PREDICTIONS: Prediction[] = [
  {
    location_id: "S_A",
    location_name: "Back Bay Station",
    predicted_time: 14.3,
    historical_avg: 8.6,
    confidence: 0.33,
    model_type: "gnn",
    factors: { demand: 150, access_score: 0.72, weather_risk: 0.2, traffic_risk: 0.4, peak_hour: 1.0 },
  },
  {
    location_id: "S_B",
    location_name: "North End",
    predicted_time: 13.9,
    historical_avg: 12.1,
    confidence: 0.86,
    model_type: "gnn",
    factors: { demand: 140, access_score: 0.61, weather_risk: 0.3, traffic_risk: 0.5, peak_hour: 1.0 },
  },
  {
    location_id: "S_C",
    location_name: "Harvard Square",
    predicted_time: 14.1,
    historical_avg: 10.6,
    confidence: 0.68,
    model_type: "gnn",
    factors: { demand: 145, access_score: 0.55, weather_risk: 0.2, traffic_risk: 0.3, peak_hour: 1.0 },
  },
  {
    location_id: "S_D",
    location_name: "Beacon Hill",
    predicted_time: 14.3,
    historical_avg: 9.1,
    confidence: 0.43,
    model_type: "gnn",
    factors: { demand: 150, access_score: 0.65, weather_risk: 0.2, traffic_risk: 0.4, peak_hour: 1.2 },
  },
  {
    location_id: "S_E",
    location_name: "South End",
    predicted_time: 13.9,
    historical_avg: 7.6,
    confidence: 0.18,
    model_type: "gnn",
    factors: { demand: 140, access_score: 0.7, weather_risk: 0.1, traffic_risk: 0.2, peak_hour: 1.0 },
  },
]

export async function fetchLocations(): Promise<Location[]> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/locations`, {
      signal: AbortSignal.timeout(3000),
    })
    if (!response.ok) throw new Error("API request failed")
    const result = await response.json()
    return result.data || []
  } catch (error) {
    console.log("[v0] Using mock location data (API unavailable)")
    return MOCK_LOCATIONS
  }
}

export async function fetchVehicles(): Promise<Vehicle[]> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/vehicles`, {
      signal: AbortSignal.timeout(3000),
    })
    if (!response.ok) throw new Error("API request failed")
    const result = await response.json()
    return result.data || []
  } catch (error) {
    console.log("[v0] Using mock vehicle data (API unavailable)")
    return MOCK_VEHICLES
  }
}

export async function fetchPredictions(): Promise<Prediction[]> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/predictions/service-times`, {
      signal: AbortSignal.timeout(3000),
    })
    if (!response.ok) throw new Error("API request failed")
    const result = await response.json()
    return result.data || []
  } catch (error) {
    console.log("[v0] Using mock prediction data (API unavailable)")
    return MOCK_PREDICTIONS
  }
}

export async function fetchAnalytics(): Promise<Analytics> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/analytics/overview`, {
      signal: AbortSignal.timeout(3000),
    })
    if (!response.ok) throw new Error("API request failed")
    const result = await response.json()
    return result.data || MOCK_ANALYTICS
  } catch (error) {
    console.log("[v0] Using mock analytics data (API unavailable)")
    return MOCK_ANALYTICS
  }
}

export async function fetchBulkData() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/bulk/all`, {
      signal: AbortSignal.timeout(5000),
    })
    if (!response.ok) throw new Error("API request failed")
    const result = await response.json()
    return result.data || { locations: MOCK_LOCATIONS, vehicles: MOCK_VEHICLES, analytics: MOCK_ANALYTICS }
  } catch (error) {
    console.log("[v0] Using mock bulk data (API unavailable)")
    return {
      locations: MOCK_LOCATIONS,
      vehicles: MOCK_VEHICLES,
      analytics: MOCK_ANALYTICS,
      service_predictions: MOCK_PREDICTIONS,
    }
  }
}
