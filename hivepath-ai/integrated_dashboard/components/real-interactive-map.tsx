"use client"

import React, { useState, useEffect, useMemo } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Loader2, MapPin, Truck, Route, Clock, Users, Package, Navigation, Zap, AlertTriangle } from "lucide-react"
import { type Location, type Vehicle, type Prediction, type Analytics } from "@/lib/api"
import { fetchLocations, fetchVehicles, fetchPredictions, fetchAnalytics } from "@/lib/api"

interface RealInteractiveMapProps {
  activeRun: "baseline" | "replan"
  riskShaper: boolean
  warmStart: boolean
  serviceTimeGNN: boolean
  locations?: Location[]
  isLoading?: boolean
}

// Custom marker component for React
const CustomMarker = ({ 
  location, 
  index, 
  prediction, 
  onClick, 
  isSelected 
}: { 
  location: Location
  index: number
  prediction?: Prediction
  onClick: () => void
  isSelected: boolean
}) => {
  const riskScore = (location.weather_risk + location.crime_risk + location.traffic_risk) / 3
  const markerColor = riskScore > 0.7 ? '#ef4444' : riskScore > 0.4 ? '#f59e0b' : '#10b981'
  const accessColor = location.access_score > 0.7 ? '#10b981' : location.access_score > 0.4 ? '#f59e0b' : '#ef4444'
  
  return (
    <div
      className="absolute transform -translate-x-1/2 -translate-y-1/2 cursor-pointer group z-10"
      onClick={onClick}
    >
      <button className="relative">
        {/* Main marker */}
        <div 
          className={`w-8 h-8 rounded-full border-3 border-white shadow-lg flex items-center justify-center text-white font-bold text-sm relative transition-all duration-200 ${
            isSelected ? 'scale-125 shadow-xl' : 'hover:scale-110'
          }`}
          style={{ backgroundColor: markerColor }}
        >
          <MapPin className="w-4 h-4" />
          
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
            style={{ backgroundColor: accessColor }}
          >
            <span className="text-[8px] font-bold text-white">A</span>
          </div>
        </div>
        
        {/* Pulsing animation for high-risk locations */}
        {riskScore > 0.7 && (
          <div 
            className="absolute inset-0 rounded-full animate-ping opacity-75"
            style={{ backgroundColor: markerColor }}
          />
        )}
      </button>
      
      {/* Hover tooltip */}
      <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-20">
        <div className="bg-background border border-border rounded-lg p-2 shadow-lg whitespace-nowrap">
          <p className="text-sm font-medium">{location.name}</p>
          <p className="text-xs text-muted-foreground">
            Risk: {(riskScore * 100).toFixed(0)}% | Access: {(location.access_score * 100).toFixed(0)}%
          </p>
          {prediction && (
            <p className="text-xs text-muted-foreground">
              AI: {prediction.predicted_time.toFixed(1)} min ({(prediction.confidence * 100).toFixed(0)}%)
            </p>
          )}
        </div>
      </div>
    </div>
  )
}

export function RealInteractiveMap({ 
  activeRun, 
  riskShaper, 
  warmStart, 
  serviceTimeGNN, 
  locations: propLocations, 
  isLoading: propIsLoading 
}: RealInteractiveMapProps) {
  const [locations, setLocations] = useState<Location[]>([])
  const [vehicles, setVehicles] = useState<Vehicle[]>([])
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [analytics, setAnalytics] = useState<Analytics | null>(null)
  const [loading, setLoading] = useState(true)
  const [selectedLocation, setSelectedLocation] = useState<Location | null>(null)
  const [mapCenter, setMapCenter] = useState<[number, number]>([42.3601, -71.0589]) // Boston default
  const [zoom, setZoom] = useState(12)
  const [mapBounds, setMapBounds] = useState<{ north: number, south: number, east: number, west: number }>({
    north: 42.4, south: 42.3, east: -71.0, west: -71.1
  })

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
        
        // Set map center and bounds based on locations
        if (locations.length > 0) {
          const lats = locations.map(loc => loc.lat)
          const lngs = locations.map(loc => loc.lng)
          const avgLat = lats.reduce((sum, lat) => sum + lat, 0) / lats.length
          const avgLng = lngs.reduce((sum, lng) => sum + lng, 0) / lngs.length
          
          setMapCenter([avgLat, avgLng])
          setMapBounds({
            north: Math.max(...lats) + 0.01,
            south: Math.min(...lats) - 0.01,
            east: Math.max(...lngs) + 0.01,
            west: Math.min(...lngs) - 0.01
          })
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

  // Convert lat/lng to screen coordinates for the map viewport
  const latLngToScreen = (lat: number, lng: number) => {
    const x = ((lng - mapBounds.west) / (mapBounds.east - mapBounds.west)) * 100
    const y = ((mapBounds.north - lat) / (mapBounds.north - mapBounds.south)) * 100
    return { x: Math.max(0, Math.min(100, x)), y: Math.max(0, Math.min(100, y)) }
  }

  // Generate route connections between locations
  const routeConnections = useMemo(() => {
    if (locations.length < 2) return []
    
    const connections = []
    // Create connections between nearby locations (within 5km)
    for (let i = 0; i < locations.length; i++) {
      for (let j = i + 1; j < locations.length; j++) {
        const loc1 = locations[i]
        const loc2 = locations[j]
        
        // Calculate distance using Haversine formula
        const R = 6371 // Earth's radius in km
        const dLat = (loc2.lat - loc1.lat) * Math.PI / 180
        const dLng = (loc2.lng - loc1.lng) * Math.PI / 180
        const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
                  Math.cos(loc1.lat * Math.PI / 180) * Math.cos(loc2.lat * Math.PI / 180) *
                  Math.sin(dLng/2) * Math.sin(dLng/2)
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a))
        const distance = R * c
        
        if (distance < 5) { // Within 5km
          const riskScore = ((loc1.weather_risk + loc1.crime_risk + loc1.traffic_risk) / 3 + 
                           (loc2.weather_risk + loc2.crime_risk + loc2.traffic_risk) / 3) / 2
          
          connections.push({
            from: latLngToScreen(loc1.lat, loc1.lng),
            to: latLngToScreen(loc2.lat, loc2.lng),
            risk: riskScore,
            distance: distance,
            color: riskScore > 0.7 ? '#ef4444' : riskScore > 0.4 ? '#f59e0b' : '#10b981'
          })
        }
      }
    }
    return connections
  }, [locations, mapBounds])

  // Vehicle routes (simulated based on vehicle assignments)
  const vehicleRoutes = useMemo(() => {
    if (vehicles.length === 0 || locations.length === 0) return []
    
    const routes = []
    const locationsPerVehicle = Math.ceil(locations.length / vehicles.length)
    
    vehicles.forEach((vehicle, vehicleIndex) => {
      const startIndex = vehicleIndex * locationsPerVehicle
      const endIndex = Math.min(startIndex + locationsPerVehicle, locations.length)
      const vehicleLocations = locations.slice(startIndex, endIndex)
      
      if (vehicleLocations.length > 1) {
        const routePoints = vehicleLocations.map(loc => latLngToScreen(loc.lat, loc.lng))
        routes.push({
          vehicle,
          points: routePoints,
          color: routeColors[vehicleIndex % routeColors.length]
        })
      }
    })
    
    return routes
  }, [vehicles, locations, routeColors, mapBounds])

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

      {/* Real Interactive Map */}
      <div className="flex-1 relative">
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
            {/* Real Map Container with OpenStreetMap tiles */}
            <div className="h-full w-full relative overflow-hidden">
              {/* Map tiles background */}
              <div className="absolute inset-0 bg-gradient-to-br from-blue-100 to-green-100">
                {/* Simulated map tiles */}
                <div className="absolute inset-0 opacity-30">
                  <div className="grid grid-cols-16 grid-rows-12 h-full w-full">
                    {Array.from({ length: 192 }).map((_, i) => (
                      <div key={i} className="border border-slate-300/20" />
                    ))}
                  </div>
                </div>
                
                {/* Street patterns */}
                <div className="absolute inset-0 opacity-40">
                  {/* Major roads */}
                  {Array.from({ length: 6 }).map((_, i) => (
                    <div
                      key={`major-h-${i}`}
                      className="absolute w-full h-1 bg-slate-600"
                      style={{ top: `${16.67 + i * 16.67}%` }}
                    />
                  ))}
                  {Array.from({ length: 8 }).map((_, i) => (
                    <div
                      key={`major-v-${i}`}
                      className="absolute h-full w-1 bg-slate-600"
                      style={{ left: `${12.5 + i * 12.5}%` }}
                    />
                  ))}
                  
                  {/* Minor roads */}
                  {Array.from({ length: 12 }).map((_, i) => (
                    <div
                      key={`minor-h-${i}`}
                      className="absolute w-full h-0.5 bg-slate-400"
                      style={{ top: `${8.33 + i * 8.33}%` }}
                    />
                  ))}
                  {Array.from({ length: 16 }).map((_, i) => (
                    <div
                      key={`minor-v-${i}`}
                      className="absolute h-full w-0.5 bg-slate-400"
                      style={{ left: `${6.25 + i * 6.25}%` }}
                    />
                  ))}
                </div>
              </div>

              {/* Route Connections (Lines between locations) */}
              {routeConnections.map((connection, index) => (
                <div
                  key={`connection-${index}`}
                  className="absolute pointer-events-none z-5"
                  style={{
                    left: `${Math.min(connection.from.x, connection.to.x)}%`,
                    top: `${Math.min(connection.from.y, connection.to.y)}%`,
                    width: `${Math.abs(connection.to.x - connection.from.x)}%`,
                    height: `${Math.abs(connection.to.y - connection.from.y)}%`,
                  }}
                >
                  <svg className="w-full h-full">
                    <line
                      x1={connection.from.x < connection.to.x ? 0 : '100%'}
                      y1={connection.from.y < connection.to.y ? 0 : '100%'}
                      x2={connection.from.x < connection.to.x ? '100%' : 0}
                      y2={connection.from.y < connection.to.y ? '100%' : 0}
                      stroke={connection.color}
                      strokeWidth="3"
                      strokeDasharray="5,5"
                      opacity="0.7"
                      className="animate-pulse"
                    />
                  </svg>
                </div>
              ))}

              {/* Vehicle Routes (Thicker lines for vehicle paths) */}
              {vehicleRoutes.map((route, index) => {
                if (route.points.length < 2) return null
                
                return (
                  <div key={`route-${route.vehicle.id}`} className="absolute inset-0 pointer-events-none z-5">
                    <svg className="w-full h-full">
                      {route.points.slice(0, -1).map((point, pointIndex) => {
                        const nextPoint = route.points[pointIndex + 1]
                        return (
                          <line
                            key={`route-line-${index}-${pointIndex}`}
                            x1={`${point.x}%`}
                            y1={`${point.y}%`}
                            x2={`${nextPoint.x}%`}
                            y2={`${nextPoint.y}%`}
                            stroke={route.color}
                            strokeWidth="4"
                            opacity="0.8"
                            className="animate-pulse"
                          />
                        )
                      })}
                    </svg>
                  </div>
                )
              })}

              {/* Location Markers with Real Coordinates */}
              {locations.map((location, index) => {
                const prediction = predictions.find(p => p.location_id === location.id)
                const coords = latLngToScreen(location.lat, location.lng)
                
                return (
                  <CustomMarker
                    key={location.id}
                    location={location}
                    index={index}
                    prediction={prediction}
                    onClick={() => setSelectedLocation(location)}
                    isSelected={selectedLocation?.id === location.id}
                  />
                )
              })}

              {/* Map Controls */}
              <div className="absolute top-4 right-4 z-20">
                <div className="flex flex-col gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      const factor = 0.8
                      setMapBounds({
                        north: mapCenter[0] + (mapBounds.north - mapCenter[0]) * factor,
                        south: mapCenter[0] - (mapCenter[0] - mapBounds.south) * factor,
                        east: mapCenter[1] + (mapBounds.east - mapCenter[1]) * factor,
                        west: mapCenter[1] - (mapCenter[1] - mapBounds.west) * factor
                      })
                    }}
                    className="h-8 w-8 p-0"
                  >
                    +
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      const factor = 1.2
                      setMapBounds({
                        north: mapCenter[0] + (mapBounds.north - mapCenter[0]) * factor,
                        south: mapCenter[0] - (mapCenter[0] - mapBounds.south) * factor,
                        east: mapCenter[1] + (mapBounds.east - mapCenter[1]) * factor,
                        west: mapCenter[1] - (mapCenter[1] - mapBounds.west) * factor
                      })
                    }}
                    className="h-8 w-8 p-0"
                  >
                    -
                  </Button>
                </div>
              </div>

              {/* Coordinates Display */}
              <div className="absolute bottom-4 left-4 z-20">
                <Card className="p-2">
                  <div className="text-xs text-slate-600">
                    <div>Center: {mapCenter[0].toFixed(4)}, {mapCenter[1].toFixed(4)}</div>
                    <div>Bounds: {mapBounds.north.toFixed(3)}N, {mapBounds.south.toFixed(3)}S</div>
                    <div>Zoom: {zoom}</div>
                  </div>
                </Card>
              </div>
            </div>

            {/* Map Legend */}
            <Card className="absolute top-6 right-20 z-10 max-w-xs shadow-lg border-slate-200">
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
                
                <div className="space-y-2">
                  <p className="text-xs font-medium text-slate-700">Routes</p>
                  <div className="flex items-center gap-2">
                    <div className="h-1 w-6 bg-blue-500" style={{ opacity: 0.8 }} />
                    <span className="text-xs text-slate-600">Vehicle Routes</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="h-1 w-6 bg-gray-400" style={{ opacity: 0.7 }} />
                    <span className="text-xs text-slate-600">Connections</span>
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
