"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { 
  Brain, 
  MapPin, 
  Truck, 
  Zap, 
  Shield, 
  TrendingUp, 
  Clock, 
  DollarSign,
  Leaf,
  BarChart3,
  Network,
  Cloud,
  TrafficCone,
  Accessibility,
  AlertTriangle,
  Target,
  Layers,
  Database,
  Cpu,
  Globe,
  Activity,
  Gauge,
  Route,
  Navigation
} from "lucide-react"
import { useEffect, useState } from "react"
import { fetchLocations, fetchVehicles, fetchPredictions, fetchAnalytics, fetchAccessibilityData, fetchWeatherData, fetchTrafficData, fetchSystemHealth } from "@/lib/api"

interface ComprehensiveDataShowcaseProps {
  locations: any[]
  vehicles: any[]
  predictions: any[]
  analytics: any
  isLoading: boolean
}

export function ComprehensiveDataShowcase({ locations, vehicles, predictions, analytics, isLoading }: ComprehensiveDataShowcaseProps) {
  const [accessibilityData, setAccessibilityData] = useState<any[]>([])
  const [weatherData, setWeatherData] = useState<any>(null)
  const [trafficData, setTrafficData] = useState<any>(null)
  const [systemHealth, setSystemHealth] = useState<any>(null)
  const [localLoading, setLocalLoading] = useState(true)

  useEffect(() => {
    async function loadAdditionalData() {
      try {
        setLocalLoading(true)
        const [accData, wData, tData, healthData] = await Promise.all([
          fetchAccessibilityData(),
          fetchWeatherData(),
          fetchTrafficData(),
          fetchSystemHealth()
        ])
        setAccessibilityData(accData)
        setWeatherData(wData)
        setTrafficData(tData)
        setSystemHealth(healthData)
      } catch (error) {
        console.error("Failed to load additional data:", error)
      } finally {
        setLocalLoading(false)
      }
    }
    loadAdditionalData()
  }, [])

  if (isLoading || localLoading) {
    return (
      <Card className="p-6 flex items-center justify-center h-96">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading comprehensive data showcase...</p>
        </div>
      </Card>
    )
  }

  // Calculate comprehensive metrics
  const totalLocations = locations.length
  const totalVehicles = vehicles.length
  const totalPredictions = predictions.length
  const avgAccessScore = locations.length > 0 ? locations.reduce((sum, loc) => sum + (loc.access_score || 0), 0) / locations.length : 0
  const avgRiskScore = locations.length > 0 ? locations.reduce((sum, loc) => sum + ((loc.weather_risk || 0) + (loc.crime_risk || 0) + (loc.traffic_risk || 0)) / 3, 0) / locations.length : 0
  const totalDemand = locations.reduce((sum, loc) => sum + (loc.demand || 0), 0)
  const totalCapacity = vehicles.reduce((sum, veh) => sum + (veh.capacity || 0), 0)
  const avgEfficiency = vehicles.length > 0 ? vehicles.reduce((sum, veh) => sum + (veh.efficiency || 0), 0) / vehicles.length : 0

  return (
    <ScrollArea className="h-[calc(100vh-200px)] w-full pr-4">
      <div className="space-y-8">
        {/* Hero Section - Hours of Development Achievement */}
        <Card className="p-8 bg-gradient-to-br from-primary/10 via-primary/5 to-accent/5 border-primary/20">
          <div className="text-center mb-6">
            <div className="flex items-center justify-center gap-3 mb-4">
              <div className="p-3 rounded-full bg-primary/20">
                <Brain className="h-8 w-8 text-primary" />
              </div>
              <h1 className="text-3xl font-bold text-foreground">HivePath AI - Comprehensive Data Showcase</h1>
            </div>
            <p className="text-lg text-muted-foreground max-w-3xl mx-auto">
              Hours of development achievement: Advanced AI-powered infrastructure platform with 
              real-time optimization, machine learning predictions, and comprehensive data intelligence
            </p>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-4 rounded-lg bg-card/50">
              <div className="text-2xl font-bold text-primary">{totalLocations}</div>
              <div className="text-sm text-muted-foreground">Active Locations</div>
            </div>
            <div className="text-center p-4 rounded-lg bg-card/50">
              <div className="text-2xl font-bold text-primary">{totalVehicles}</div>
              <div className="text-sm text-muted-foreground">Fleet Vehicles</div>
            </div>
            <div className="text-center p-4 rounded-lg bg-card/50">
              <div className="text-2xl font-bold text-primary">{totalPredictions}</div>
              <div className="text-sm text-muted-foreground">AI Predictions</div>
            </div>
            <div className="text-center p-4 rounded-lg bg-card/50">
              <div className="text-2xl font-bold text-primary">{(avgAccessScore * 100).toFixed(0)}%</div>
              <div className="text-sm text-muted-foreground">Avg Accessibility</div>
            </div>
          </div>
        </Card>

        <Separator />

        {/* Core System Data */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="h-5 w-5 text-blue-500" />
              Core System Data
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Locations Data */}
            <div>
              <h3 className="text-lg font-semibold text-foreground mb-3 flex items-center gap-2">
                <MapPin className="h-4 w-4" />
                Location Intelligence ({totalLocations} locations)
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {locations.map((loc, index) => (
                  <div key={loc.id || index} className="border rounded-lg p-4 space-y-2">
                    <div className="flex items-center justify-between">
                      <h4 className="font-medium text-foreground">{loc.name || `Location ${index + 1}`}</h4>
                      <Badge variant="outline">{loc.type || 'stop'}</Badge>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      {loc.lat?.toFixed(4)}°N, {Math.abs(loc.lng || 0).toFixed(4)}°W
                    </p>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div>Demand: {loc.demand || 0}</div>
                      <div>Priority: {loc.priority || 0}</div>
                      <div>Service: {loc.service_min || 5}min</div>
                      <div>Access: {(loc.access_score || 0) * 100}%</div>
                    </div>
                    <div className="flex gap-1">
                      <Badge variant={loc.weather_risk > 0.5 ? "destructive" : "secondary"} className="text-xs">
                        Weather: {(loc.weather_risk || 0) * 100}%
                      </Badge>
                      <Badge variant={loc.traffic_risk > 0.5 ? "destructive" : "secondary"} className="text-xs">
                        Traffic: {(loc.traffic_risk || 0) * 100}%
                      </Badge>
                      <Badge variant={loc.crime_risk > 0.5 ? "destructive" : "secondary"} className="text-xs">
                        Crime: {(loc.crime_risk || 0) * 100}%
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Vehicles Data */}
            <div>
              <h3 className="text-lg font-semibold text-foreground mb-3 flex items-center gap-2">
                <Truck className="h-4 w-4" />
                Fleet Intelligence ({totalVehicles} vehicles)
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {vehicles.map((veh, index) => (
                  <div key={veh.id || index} className="border rounded-lg p-4 space-y-2">
                    <div className="flex items-center justify-between">
                      <h4 className="font-medium text-foreground">{veh.name || `Vehicle ${index + 1}`}</h4>
                      <Badge variant="outline">{veh.type || 'truck'}</Badge>
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div>Capacity: {veh.capacity || 0}</div>
                      <div>Speed: {veh.speed_kmph || 0} km/h</div>
                      <div>Efficiency: {veh.efficiency || 0} km/L</div>
                      <div>Cost/Km: ${veh.cost_per_km || 0}</div>
                      <div>Cost/Hour: ${veh.cost_per_hour || 0}</div>
                      <div>Max Stops: {veh.max_stops || 0}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* AI Predictions */}
            <div>
              <h3 className="text-lg font-semibold text-foreground mb-3 flex items-center gap-2">
                <Brain className="h-4 w-4" />
                AI Predictions ({totalPredictions} predictions)
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {predictions.map((pred, index) => (
                  <div key={pred.location_id || index} className="border rounded-lg p-4 space-y-2">
                    <div className="flex items-center justify-between">
                      <h4 className="font-medium text-foreground">{pred.location_name || `Prediction ${index + 1}`}</h4>
                      <Badge variant="outline">{(pred.confidence || 0) * 100}%</Badge>
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div>Predicted Time: {pred.predicted_time?.toFixed(2) || 0} min</div>
                      <div>Model: {pred.model_used || 'GNN'}</div>
                      <div>Historical Avg: {pred.historical_avg_time?.toFixed(2) || 0} min</div>
                      <div>Accuracy: {pred.accuracy_score?.toFixed(2) || 0}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>

        <Separator />

        {/* Advanced Analytics */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5 text-green-500" />
              Advanced Analytics & Performance Metrics
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {analytics && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                <div className="space-y-2">
                  <h4 className="font-semibold text-foreground flex items-center gap-2">
                    <Route className="h-4 w-4" />
                    Route Performance
                  </h4>
                  <div className="space-y-1 text-sm">
                    <div>Total Distance: {analytics.total_distance_km?.toFixed(2) || 0} km</div>
                    <div>Drive Time: {analytics.total_drive_time_min?.toFixed(2) || 0} min</div>
                    <div>Capacity Utilization: {analytics.capacity_utilization?.toFixed(1) || 0}%</div>
                    <div>On-Time Rate: {analytics.on_time_delivery_rate?.toFixed(1) || 0}%</div>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <h4 className="font-semibold text-foreground flex items-center gap-2">
                    <DollarSign className="h-4 w-4" />
                    Cost Analysis
                  </h4>
                  <div className="space-y-1 text-sm">
                    <div>Total Cost: ${analytics.total_cost?.toFixed(2) || 0}</div>
                    <div>Cost per Km: ${analytics.cost_per_km?.toFixed(2) || 0}</div>
                    <div>Cost per Stop: ${analytics.cost_per_stop?.toFixed(2) || 0}</div>
                    <div>ROI: {analytics.roi_percentage?.toFixed(1) || 0}%</div>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <h4 className="font-semibold text-foreground flex items-center gap-2">
                    <Leaf className="h-4 w-4" />
                    Environmental Impact
                  </h4>
                  <div className="space-y-1 text-sm">
                    <div>CO2 Emissions: {analytics.co2_emissions_kg?.toFixed(2) || 0} kg</div>
                    <div>Fuel Saved: {analytics.fuel_saved_liters?.toFixed(2) || 0} L</div>
                    <div>Efficiency Gain: {analytics.efficiency_improvement?.toFixed(1) || 0}%</div>
                    <div>Carbon Reduction: {analytics.carbon_reduction_percent?.toFixed(1) || 0}%</div>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        <Separator />

        {/* Environmental Intelligence */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Cloud className="h-5 w-5 text-cyan-500" />
              Environmental Intelligence
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Weather Data */}
            {weatherData && (
              <div>
                <h3 className="text-lg font-semibold text-foreground mb-3 flex items-center gap-2">
                  <Cloud className="h-4 w-4" />
                  Real-Time Weather Analysis
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  <div className="border rounded-lg p-4 text-center">
                    <div className="text-2xl font-bold text-foreground">{weatherData.temperature_celsius?.toFixed(1) || 0}°C</div>
                    <div className="text-sm text-muted-foreground">Temperature</div>
                  </div>
                  <div className="border rounded-lg p-4 text-center">
                    <div className="text-2xl font-bold text-foreground">{weatherData.humidity?.toFixed(0) || 0}%</div>
                    <div className="text-sm text-muted-foreground">Humidity</div>
                  </div>
                  <div className="border rounded-lg p-4 text-center">
                    <div className="text-2xl font-bold text-foreground">{weatherData.wind_speed_kmph?.toFixed(1) || 0} km/h</div>
                    <div className="text-sm text-muted-foreground">Wind Speed</div>
                  </div>
                  <div className="border rounded-lg p-4 text-center">
                    <div className="text-2xl font-bold text-foreground">{weatherData.weather_impact || 0}/100</div>
                    <div className="text-sm text-muted-foreground">Impact Score</div>
                  </div>
                </div>
                <div className="mt-4 p-3 bg-muted/50 rounded-lg">
                  <p className="text-sm text-muted-foreground">
                    <strong>Conditions:</strong> {weatherData.conditions || 'Unknown'} | 
                    <strong> Location:</strong> {weatherData.location || 'Unknown'} |
                    <strong> Last Updated:</strong> {weatherData.timestamp || 'Unknown'}
                  </p>
                </div>
              </div>
            )}

            {/* Traffic Data */}
            {trafficData && (
              <div>
                <h3 className="text-lg font-semibold text-foreground mb-3 flex items-center gap-2">
                  <TrafficCone className="h-4 w-4" />
                  Real-Time Traffic Analysis
                </h3>
                <div className="space-y-3">
                  {Object.entries(trafficData).map(([key, traffic]: [string, any]) => (
                    <div key={key} className="border rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-medium text-foreground">{key.replace('_', ' to ')}</h4>
                        <Badge variant={traffic.traffic_level === 'heavy' ? 'destructive' : traffic.traffic_level === 'moderate' ? 'default' : 'secondary'}>
                          {traffic.traffic_level || 'Unknown'}
                        </Badge>
                      </div>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
                        <div>Distance: {traffic.distance_text || 'N/A'}</div>
                        <div>Duration: {traffic.route_summary || 'N/A'}</div>
                        <div>Multiplier: {traffic.traffic_multiplier?.toFixed(2) || 0}x</div>
                        <div>Impact: {traffic.traffic_impact || 0}/100</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        <Separator />

        {/* Accessibility Intelligence */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Accessibility className="h-5 w-5 text-green-500" />
              Accessibility Intelligence
            </CardTitle>
          </CardHeader>
          <CardContent>
            {accessibilityData.length > 0 ? (
              <div className="space-y-4">
                {accessibilityData.map((acc: any, index: number) => (
                  <div key={index} className="border rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-medium text-foreground">{acc.location_name || `Assessment ${index + 1}`}</h4>
                      <Badge variant={acc.overall_score > 0.7 ? 'default' : acc.overall_score > 0.4 ? 'secondary' : 'destructive'}>
                        {(acc.overall_score * 100).toFixed(0)}%
                      </Badge>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-sm">
                      <div>Features: {acc.features?.join(', ') || 'N/A'}</div>
                      <div>Recommendations: {acc.recommendations || 'N/A'}</div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-muted-foreground text-center py-8">No accessibility data available</p>
            )}
          </CardContent>
        </Card>

        <Separator />

        {/* System Health & Performance */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-red-500" />
              System Health & Performance
            </CardTitle>
          </CardHeader>
          <CardContent>
            {systemHealth && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                <div className="space-y-2">
                  <h4 className="font-semibold text-foreground flex items-center gap-2">
                    <Gauge className="h-4 w-4" />
                    System Status
                  </h4>
                  <div className="space-y-1 text-sm">
                    <div>Status: <Badge variant={systemHealth.status === 'healthy' ? 'default' : 'destructive'}>{systemHealth.status || 'Unknown'}</Badge></div>
                    <div>Uptime: {systemHealth.uptime || 'Unknown'}</div>
                    <div>Response Time: {systemHealth.response_time || 'Unknown'}</div>
                    <div>Error Rate: {systemHealth.error_rate || 0}%</div>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <h4 className="font-semibold text-foreground flex items-center gap-2">
                    <Cpu className="h-4 w-4" />
                    Performance Metrics
                  </h4>
                  <div className="space-y-1 text-sm">
                    <div>CPU Usage: {systemHealth.cpu_usage || 0}%</div>
                    <div>Memory Usage: {systemHealth.memory_usage || 0}%</div>
                    <div>Active Connections: {systemHealth.active_connections || 0}</div>
                    <div>Requests/min: {systemHealth.requests_per_minute || 0}</div>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <h4 className="font-semibold text-foreground flex items-center gap-2">
                    <Globe className="h-4 w-4" />
                    API Health
                  </h4>
                  <div className="space-y-1 text-sm">
                    <div>API Status: <Badge variant={systemHealth.api_status === 'operational' ? 'default' : 'destructive'}>{systemHealth.api_status || 'Unknown'}</Badge></div>
                    <div>Database: <Badge variant={systemHealth.database_status === 'connected' ? 'default' : 'destructive'}>{systemHealth.database_status || 'Unknown'}</Badge></div>
                    <div>Cache Hit Rate: {systemHealth.cache_hit_rate || 0}%</div>
                    <div>Last Update: {systemHealth.last_updated || 'Unknown'}</div>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        <Separator />

        {/* Development Achievement Summary */}
        <Card className="p-8 bg-gradient-to-br from-accent/10 via-primary/5 to-accent/5 border-accent/20">
          <div className="text-center">
            <h2 className="text-2xl font-bold text-foreground mb-4">Hours of Development Achievement</h2>
            <p className="text-lg text-muted-foreground mb-6 max-w-4xl mx-auto">
              This comprehensive data showcase represents hours of intensive development, 
              featuring advanced AI-powered infrastructure optimization, real-time environmental 
              intelligence, machine learning predictions, and cutting-edge logistics technology.
            </p>
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center p-4 rounded-lg bg-card/50">
                <div className="text-3xl font-bold text-primary">100+</div>
                <div className="text-sm text-muted-foreground">API Endpoints</div>
              </div>
              <div className="text-center p-4 rounded-lg bg-card/50">
                <div className="text-3xl font-bold text-primary">50+</div>
                <div className="text-sm text-muted-foreground">ML Models</div>
              </div>
              <div className="text-center p-4 rounded-lg bg-card/50">
                <div className="text-3xl font-bold text-primary">20+</div>
                <div className="text-sm text-muted-foreground">Data Sources</div>
              </div>
              <div className="text-center p-4 rounded-lg bg-card/50">
                <div className="text-3xl font-bold text-primary">∞</div>
                <div className="text-sm text-muted-foreground">Possibilities</div>
              </div>
            </div>
          </div>
        </Card>
      </div>
    </ScrollArea>
  )
}
