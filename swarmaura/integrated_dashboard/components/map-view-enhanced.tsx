"use client"

import React, { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Loader2, MapPin, Truck, Route, Clock, Users, Package, Navigation } from "lucide-react"
import { type Location, type Vehicle, type Prediction, type Analytics } from "@/lib/api"
import { fetchLocations, fetchVehicles, fetchPredictions, fetchAnalytics } from "@/lib/api"

interface MapViewEnhancedProps {
  activeRun: "baseline" | "replan"
  riskShaper: boolean
  warmStart: boolean
  serviceTimeGNN: boolean
  locations?: Location[]
  isLoading?: boolean
}

export function MapViewEnhanced({ 
  activeRun, 
  riskShaper, 
  warmStart, 
  serviceTimeGNN, 
  locations: propLocations, 
  isLoading: propIsLoading 
}: MapViewEnhancedProps) {
  const [locations, setLocations] = useState<Location[]>([])
  const [vehicles, setVehicles] = useState<Vehicle[]>([])
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [analytics, setAnalytics] = useState<Analytics | null>(null)
  const [loading, setLoading] = useState(true)
  const [selectedLocation, setSelectedLocation] = useState<Location | null>(null)
  const [mapCenter, setMapCenter] = useState<[number, number]>([42.3601, -71.0589]) // Boston default
  const [zoom, setZoom] = useState(12)

  useEffect(() => {
    async function loadData() {
      try {
        // Use passed locations if available, otherwise fetch
        if (propLocations && propLocations.length > 0) {
          setLocations(propLocations)
        } else {
          const locationsData = await fetchLocations()
          setLocations(locationsData)
        }
        
        const [vehiclesData, predictionsData, analyticsData] = await Promise.all([
          fetchVehicles(),
          fetchPredictions(),
          fetchAnalytics()
        ])
        
        setVehicles(vehiclesData)
        setPredictions(predictionsData)
        setAnalytics(analyticsData)
        
        // Set map center based on locations
        if (locations.length > 0) {
          const avgLat = locations.reduce((sum, loc) => sum + loc.lat, 0) / locations.length
          const avgLng = locations.reduce((sum, loc) => sum + loc.lng, 0) / locations.length
          setMapCenter([avgLat, avgLng])
        }
      } catch (error) {
        console.error("Failed to load data:", error)
      } finally {
        setLoading(false)
      }
    }
    loadData()
  }, [propLocations])

  // Calculate statistics
  const totalStops = locations.length
  const totalVehicles = vehicles.length
  const avgServiceTime = predictions.length > 0 
    ? predictions.reduce((sum, pred) => sum + pred.predicted_time, 0) / predictions.length 
    : 0
  const avgRisk = locations.length > 0 
    ? locations.reduce((sum, loc) => sum + (loc.weather_risk + loc.crime_risk + loc.traffic_risk) / 3, 0) / locations.length 
    : 0
  const highRiskStops = locations.filter(loc => (loc.weather_risk + loc.crime_risk + loc.traffic_risk) / 3 > 0.7).length
  const avgAccessScore = locations.length > 0 
    ? locations.reduce((sum, loc) => sum + loc.access_score, 0) / locations.length 
    : 0

  // Route colors for different vehicles
  const routeColors = [
    '#ef4444', // Red
    '#3b82f6', // Blue
    '#10b981', // Green
    '#f59e0b', // Yellow
    '#8b5cf6', // Purple
    '#ec4899', // Pink
    '#06b6d4', // Cyan
    '#84cc16', // Lime
  ]

  if (loading || propIsLoading) {
    return (
      <div className="flex flex-col h-full overflow-hidden">
        <div className="border-b-2 border-border bg-gradient-to-r from-primary/5 via-primary/10 to-primary/5 p-6 flex-shrink-0">
          <div className="flex items-center gap-4">
            <img src="/logo.png" alt="HivePath AI" className="h-14 w-14" />
            <div>
              <h1 className="text-2xl font-bold text-foreground">Next-Gen Infrastructure Network</h1>
              <p className="text-muted-foreground">Reimagining transportation with smarter grids, stronger infrastructure, and faster logistics</p>
            </div>
          </div>
        </div>
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <img src="/logo.png" alt="HivePath AI" className="h-24 w-24 mx-auto mb-4 animate-pulse" />
            <h2 className="text-xl font-semibold text-foreground mb-2">HivePath AI</h2>
            <p className="text-muted-foreground">Loading intelligent infrastructure network...</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="border-b-2 border-border bg-gradient-to-r from-primary/5 via-primary/10 to-primary/5 p-6 flex-shrink-0">
        <div className="flex items-center gap-4">
          <img src="/logo.png" alt="HivePath AI" className="h-14 w-14" />
          <div>
            <h1 className="text-2xl font-bold text-foreground">Next-Gen Infrastructure Network</h1>
            <p className="text-muted-foreground">Reimagining transportation with smarter grids, stronger infrastructure, and faster logistics</p>
          </div>
        </div>
        
        {/* Statistics */}
        <div className="flex items-center gap-6 mt-4">
          <div className="flex items-center gap-2">
            <div className="h-3 w-3 rounded-full bg-primary" />
            <span className="text-sm font-medium text-foreground">{totalStops} Stops</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="h-3 w-3 rounded-full bg-blue-500" />
            <span className="text-sm font-medium text-foreground">{totalVehicles} Vehicles</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="h-3 w-3 rounded-full bg-green-500" />
            <span className="text-sm font-medium text-foreground">{avgServiceTime.toFixed(1)}m Avg</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="h-3 w-3 rounded-full bg-red-500" />
            <span className="text-sm font-medium text-foreground">{highRiskStops} High Risk</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="h-3 w-3 rounded-full bg-green-500" />
            <span className="text-sm font-medium text-foreground">{(avgAccessScore * 100).toFixed(0)}% Access</span>
          </div>
        </div>
      </div>

      {/* Map Content */}
      <div className="flex-1 relative bg-gradient-to-br from-slate-50 to-blue-50">
        {locations.length === 0 ? (
          <div className="h-full flex items-center justify-center">
            <Card className="max-w-md border-slate-200">
              <CardHeader>
                <CardTitle className="text-center flex items-center gap-2">
                  <MapPin className="h-5 w-5" />
                  No Locations Available
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-center text-slate-600">
                  There are no locations with coordinates available. 
                  Locations will appear here once they have been configured with proper coordinates.
                </p>
              </CardContent>
            </Card>
          </div>
        ) : (
          <>
            {/* Simulated Map Area */}
            <div className="h-full w-full relative bg-gradient-to-br from-blue-100 to-green-100">
              {/* Map Grid Pattern */}
              <div className="absolute inset-0 opacity-20">
                <div className="grid grid-cols-12 grid-rows-8 h-full w-full">
                  {Array.from({ length: 96 }).map((_, i) => (
                    <div key={i} className="border border-slate-300/30" />
                  ))}
                </div>
              </div>

              {/* Location Markers */}
              {locations.map((location, index) => {
                const prediction = predictions.find(p => p.location_id === location.id)
                const riskScore = (location.weather_risk + location.crime_risk + location.traffic_risk) / 3
                const markerColor = riskScore > 0.7 ? '#ef4444' : riskScore > 0.4 ? '#f59e0b' : '#10b981'
                
                // Simulate map coordinates (in a real implementation, these would be actual lat/lng)
                const x = 10 + (index * 15) % 80
                const y = 15 + (index * 20) % 70
                
                return (
                  <div
                    key={location.id}
                    className="absolute transform -translate-x-1/2 -translate-y-1/2 cursor-pointer group"
                    style={{ left: `${x}%`, top: `${y}%` }}
                    onClick={() => setSelectedLocation(location)}
                  >
                    <button className="relative">
                      <MapPin className="h-6 w-6 text-white drop-shadow-lg" style={{ color: markerColor }} />
                      
                      {/* Stop number badge */}
                      <div className="absolute -top-1 -right-1 h-5 w-5 rounded-full bg-background border-2 border-current flex items-center justify-center">
                        <span className="text-[10px] font-bold">{index + 1}</span>
                      </div>
                      
                      {/* Demand indicator */}
                      {location.demand > 5 && (
                        <div className="absolute -bottom-1 -right-1 h-4 w-4 rounded-full bg-blue-500 border-2 border-background flex items-center justify-center">
                          <span className="text-[8px] font-bold text-white">{location.demand}</span>
                        </div>
                      )}
                      
                      {/* Access score indicator */}
                      <div 
                        className="absolute -top-1 -left-1 h-4 w-4 rounded-full border-2 border-background flex items-center justify-center"
                        style={{
                          backgroundColor: location.access_score > 0.7 ? '#10b981' : location.access_score > 0.4 ? '#f59e0b' : '#ef4444'
                        }}
                      >
                        <span className="text-[8px] font-bold text-white">A</span>
                      </div>
                    </button>
                    
                    {/* Hover tooltip */}
                    <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none">
                      <div className="bg-background border border-border rounded-lg p-2 shadow-lg whitespace-nowrap">
                        <p className="text-sm font-medium">{location.name}</p>
                        <p className="text-xs text-muted-foreground">
                          Risk: {(riskScore * 100).toFixed(0)}% | Access: {(location.access_score * 100).toFixed(0)}%
                        </p>
                      </div>
                    </div>
                  </div>
                )
              })}

              {/* Vehicle Routes (Simulated) */}
              {vehicles.map((vehicle, vehicleIndex) => {
                const vehicleLocations = locations.slice(vehicleIndex * 2, (vehicleIndex + 1) * 2)
                if (vehicleLocations.length < 2) return null
                
                const startX = 10 + (vehicleIndex * 15) % 80
                const startY = 15 + (vehicleIndex * 20) % 70
                const endX = 10 + ((vehicleIndex + 1) * 15) % 80
                const endY = 15 + ((vehicleIndex + 1) * 20) % 70
                
                return (
                  <div key={vehicle.id} className="absolute inset-0 pointer-events-none">
                    <svg className="w-full h-full">
                      <path
                        d={`M ${startX}% ${startY}% Q ${(startX + endX) / 2}% ${(startY + endY) / 2 - 10}% ${endX}% ${endY}%`}
                        stroke={routeColors[vehicleIndex % routeColors.length]}
                        strokeWidth="3"
                        fill="none"
                        strokeDasharray="5,5"
                        className="animate-pulse"
                      />
                    </svg>
                  </div>
                )
              })}
            </div>

            {/* Map Legend */}
            <Card className="absolute top-6 right-6 z-10 max-w-xs shadow-lg border-slate-200">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-semibold text-slate-900 flex items-center gap-2">
                  <Navigation className="h-4 w-4" />
                  Map Legend
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="space-y-2">
                  <p className="text-xs font-medium text-slate-700">Risk Levels</p>
                  <div className="flex items-center gap-2">
                    <div className="h-3 w-3 rounded-full bg-green-500" />
                    <span className="text-xs text-slate-600">Low Risk (0-40%)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="h-3 w-3 rounded-full bg-yellow-500" />
                    <span className="text-xs text-slate-600">Medium Risk (40-70%)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="h-3 w-3 rounded-full bg-red-500" />
                    <span className="text-xs text-slate-600">High Risk (70%+)</span>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <p className="text-xs font-medium text-slate-700">Indicators</p>
                  <div className="flex items-center gap-2">
                    <div className="h-3 w-3 rounded-full bg-blue-500 border-2 border-background flex items-center justify-center">
                      <span className="text-[6px] font-bold text-white">5</span>
                    </div>
                    <span className="text-xs text-slate-600">High Demand</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="h-3 w-3 rounded-full bg-green-500 border-2 border-background flex items-center justify-center">
                      <span className="text-[6px] font-bold text-white">A</span>
                    </div>
                    <span className="text-xs text-slate-600">Accessibility Score</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Location Details Panel */}
            {selectedLocation && (
              <Card className="absolute bottom-6 left-6 z-10 max-w-sm shadow-lg border-slate-200">
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-sm font-semibold text-slate-900 flex items-center gap-2">
                      <MapPin className="h-4 w-4" />
                      Location Details
                    </CardTitle>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setSelectedLocation(null)}
                      className="h-6 w-6 p-0"
                    >
                      ×
                    </Button>
                  </div>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div>
                    <h3 className="font-medium text-slate-900">{selectedLocation.name}</h3>
                    <p className="text-xs text-slate-500">
                      {selectedLocation.lat.toFixed(4)}° N, {Math.abs(selectedLocation.lng).toFixed(4)}° W
                    </p>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <p className="text-xs text-slate-500">Demand</p>
                      <p className="text-sm font-medium">{selectedLocation.demand}</p>
                    </div>
                    <div>
                      <p className="text-xs text-slate-500">Priority</p>
                      <p className="text-sm font-medium">{selectedLocation.priority}</p>
                    </div>
                    <div>
                      <p className="text-xs text-slate-500">Service Time</p>
                      <p className="text-sm font-medium">{selectedLocation.service_min} min</p>
                    </div>
                    <div>
                      <p className="text-xs text-slate-500">Access Score</p>
                      <p className="text-sm font-medium">{(selectedLocation.access_score * 100).toFixed(0)}%</p>
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between text-xs">
                      <span>Weather Risk</span>
                      <span>{(selectedLocation.weather_risk * 100).toFixed(0)}%</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span>Traffic Risk</span>
                      <span>{(selectedLocation.traffic_risk * 100).toFixed(0)}%</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span>Crime Risk</span>
                      <span>{(selectedLocation.crime_risk * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                  
                  {predictions.find(p => p.location_id === selectedLocation.id) && (
                    <div className="pt-2 border-t border-slate-200">
                      <p className="text-xs text-slate-500">AI Prediction</p>
                      <p className="text-sm font-medium">
                        {predictions.find(p => p.location_id === selectedLocation.id)?.predicted_time.toFixed(1)} min
                        <span className="text-xs text-slate-500 ml-1">
                          ({(predictions.find(p => p.location_id === selectedLocation.id)?.confidence || 0) * 100}% confidence)
                        </span>
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}

            {/* Route Summary Panel */}
            <Card className="absolute top-6 left-6 z-10 max-w-sm shadow-lg border-slate-200">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-semibold text-slate-900 flex items-center gap-2">
                  <Route className="h-4 w-4" />
                  Route Summary
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="space-y-2">
                  {vehicles.map((vehicle, index) => (
                    <div key={vehicle.id} className="flex items-center gap-3 p-2 rounded-lg hover:bg-slate-50 transition-colors">
                      <div 
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: routeColors[index % routeColors.length] }}
                      />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-slate-900 truncate">{vehicle.name}</p>
                        <p className="text-xs text-slate-500">{vehicle.type} • {vehicle.capacity} capacity</p>
                      </div>
                      <Badge variant="outline" className="text-xs">
                        {Math.ceil(locations.length / vehicles.length)} stops
                      </Badge>
                    </div>
                  ))}
                </div>
                
                <div className="pt-2 border-t border-slate-200">
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div>
                      <p className="text-slate-500">Total Distance</p>
                      <p className="font-medium">{analytics?.total_distance_km?.toFixed(1) || '0.0'} km</p>
                    </div>
                    <div>
                      <p className="text-slate-500">Avg Time</p>
                      <p className="font-medium">{avgServiceTime.toFixed(1)} min</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </>
        )}
      </div>
    </div>
  )
}
