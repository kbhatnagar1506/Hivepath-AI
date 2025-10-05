"use client"

import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { 
  MapPin, 
  Truck, 
  Brain, 
  TrendingUp, 
  Clock, 
  DollarSign, 
  Zap, 
  Shield,
  Activity,
  BarChart3,
  Users,
  Package
} from "lucide-react"
import type { Location, Vehicle, Prediction, Analytics } from "@/lib/api"

interface DataDisplayPanelProps {
  locations: Location[]
  vehicles: Vehicle[]
  predictions: Prediction[]
  analytics: Analytics | null
  isLoading: boolean
}

export function DataDisplayPanel({ 
  locations, 
  vehicles, 
  predictions, 
  analytics, 
  isLoading 
}: DataDisplayPanelProps) {
  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin h-8 w-8 border-4 border-primary border-t-transparent rounded-full" />
      </div>
    )
  }

  const depot = locations.find(l => l.type === "depot")
  const stops = locations.filter(l => l.type === "stop")
  const trucks = vehicles.filter(v => v.type === "truck")
  const vans = vehicles.filter(v => v.type === "van")
  
  const totalDemand = stops.reduce((sum, s) => sum + s.demand, 0)
  const totalCapacity = vehicles.reduce((sum, v) => sum + v.capacity, 0)
  const avgConfidence = predictions.reduce((sum, p) => sum + p.confidence, 0) / predictions.length
  const avgServiceTime = predictions.reduce((sum, p) => sum + p.predicted_time, 0) / predictions.length

  return (
    <div className="space-y-6 overflow-y-auto max-h-full">
      {/* System Overview */}
      <Card className="p-6 bg-gradient-to-br from-blue-500/10 to-cyan-500/5 border-blue-500/20">
        <div className="flex items-center gap-3 mb-4">
          <img 
            src="/logo.png" 
            alt="HivePath AI" 
            className="h-12 w-12 object-contain drop-shadow-lg"
            onError={(e) => {
              e.currentTarget.style.display = 'none';
              e.currentTarget.nextElementSibling.style.display = 'block';
            }}
          />
          <BarChart3 className="h-8 w-8 text-blue-500 hidden" />
          <div>
            <h3 className="text-lg font-semibold text-foreground">Next-Gen Infrastructure Platform</h3>
            <p className="text-sm text-muted-foreground">Reimagining transportation and logistics systems</p>
          </div>
        </div>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-foreground">{locations.length}</div>
            <div className="text-xs text-muted-foreground">Locations</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-foreground">{vehicles.length}</div>
            <div className="text-xs text-muted-foreground">Vehicles</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-foreground">{totalDemand}</div>
            <div className="text-xs text-muted-foreground">Total Demand</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-foreground">{totalCapacity}</div>
            <div className="text-xs text-muted-foreground">Total Capacity</div>
          </div>
        </div>
      </Card>

      {/* Location Details */}
      <Card className="p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-green-500/20">
            <MapPin className="h-6 w-6 text-green-500" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-foreground">Location Details</h3>
            <p className="text-sm text-muted-foreground">Service locations and accessibility</p>
          </div>
        </div>
        
        <div className="space-y-3">
          {stops.map((stop) => (
            <div key={stop.id} className="flex items-center justify-between p-3 bg-card/50 rounded-lg">
              <div className="flex items-center gap-3">
                <div className="w-3 h-3 rounded-full bg-green-500"></div>
                <div>
                  <div className="font-medium text-foreground">{stop.name}</div>
                  <div className="text-xs text-muted-foreground">
                    Demand: {stop.demand} | Priority: {stop.priority}
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Badge variant="outline" className="text-xs">
                  Access: {(stop.access_score * 100).toFixed(0)}%
                </Badge>
                <Badge variant="outline" className="text-xs">
                  Risk: {((stop.weather_risk + stop.traffic_risk + stop.crime_risk) / 3 * 100).toFixed(0)}%
                </Badge>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Vehicle Fleet */}
      <Card className="p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-orange-500/20">
            <Truck className="h-6 w-6 text-orange-500" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-foreground">Vehicle Fleet</h3>
            <p className="text-sm text-muted-foreground">Fleet composition and capabilities</p>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {vehicles.map((vehicle) => (
            <div key={vehicle.id} className="p-4 bg-card/50 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <div className="font-medium text-foreground">{vehicle.id}</div>
                <Badge variant={vehicle.type === "truck" ? "default" : "secondary"}>
                  {vehicle.type}
                </Badge>
              </div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Capacity:</span>
                  <span className="font-medium">{vehicle.capacity} units</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Efficiency:</span>
                  <span className="font-medium">{(vehicle.efficiency * 100).toFixed(0)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Cost/km:</span>
                  <span className="font-medium">${vehicle.cost_per_km}</span>
                </div>
                <div className="flex flex-wrap gap-1 mt-2">
                  {vehicle.capabilities.map((cap) => (
                    <Badge key={cap} variant="outline" className="text-xs">
                      {cap}
                    </Badge>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* AI Predictions */}
      <Card className="p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-purple-500/20">
            <Brain className="h-6 w-6 text-purple-500" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-foreground">AI Predictions</h3>
            <p className="text-sm text-muted-foreground">Service time predictions with confidence scores</p>
          </div>
        </div>
        
        <div className="space-y-3">
          {predictions.map((prediction) => (
            <div key={prediction.location_id} className="p-4 bg-card/50 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <div className="font-medium text-foreground">{prediction.location_name}</div>
                <Badge 
                  variant={prediction.confidence > 0.7 ? "default" : "secondary"}
                  className="text-xs"
                >
                  {(prediction.confidence * 100).toFixed(0)}% confidence
                </Badge>
              </div>
              <div className="flex items-center justify-between">
                <div className="text-sm text-muted-foreground">
                  Predicted: {prediction.predicted_time.toFixed(1)} min
                </div>
                <div className="text-sm text-muted-foreground">
                  Historical: {prediction.historical_avg.toFixed(1)} min
                </div>
              </div>
              <div className="mt-2">
                <Progress 
                  value={prediction.confidence * 100} 
                  className="h-2"
                />
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Analytics Summary */}
      {analytics && (
        <Card className="p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-cyan-500/20">
              <Activity className="h-6 w-6 text-cyan-500" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-foreground">Analytics Summary</h3>
              <p className="text-sm text-muted-foreground">System performance metrics</p>
            </div>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-foreground">
                {analytics.capacity_utilization.toFixed(1)}%
              </div>
              <div className="text-xs text-muted-foreground">Utilization</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-foreground">
                {analytics.total_demand}
              </div>
              <div className="text-xs text-muted-foreground">Total Demand</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-foreground">
                {analytics.total_capacity}
              </div>
              <div className="text-xs text-muted-foreground">Total Capacity</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-foreground">
                {analytics.system_status || "Operational"}
              </div>
              <div className="text-xs text-muted-foreground">Status</div>
            </div>
          </div>
        </Card>
      )}
    </div>
  )
}
