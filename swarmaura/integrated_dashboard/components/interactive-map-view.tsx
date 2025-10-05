"use client"

import React, { useState, useEffect, useMemo } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Loader2, MapPin, Truck, Route, Clock, Users, Package, Navigation, Zap, AlertTriangle } from "lucide-react"
import { type Location, type Vehicle, type Prediction, type Analytics } from "@/lib/api"
import { fetchLocations, fetchVehicles, fetchPredictions, fetchAnalytics } from "@/lib/api"
import dynamic from "next/dynamic"

// Dynamically import Leaflet components to avoid SSR issues
const MapContainer = dynamic(() => import('react-leaflet').then((mod) => mod.MapContainer), { ssr: false })
const TileLayer = dynamic(() => import('react-leaflet').then((mod) => mod.TileLayer), { ssr: false })
const Marker = dynamic(() => import('react-leaflet').then((mod) => mod.Marker), { ssr: false })
const Popup = dynamic(() => import('react-leaflet').then((mod) => mod.Popup), { ssr: false })
const Polyline = dynamic(() => import('react-leaflet').then((mod) => mod.Polyline), { ssr: false })
const Tooltip = dynamic(() => import('react-leaflet').then((mod) => mod.Tooltip), { ssr: false })

interface InteractiveMapViewProps {
  activeRun: "baseline" | "replan"
  riskShaper: boolean
  warmStart: boolean
  serviceTimeGNN: boolean
  locations?: Location[]
  isLoading?: boolean
}

// Custom marker icons
const createCustomIcon = (color: string, size: number = 25) => {
  if (typeof window === 'undefined') return null
  
  const L = require('leaflet')
  
  return L.divIcon({
    className: 'custom-marker',
    html: `
      <div style="
        background-color: ${color};
        width: ${size}px;
        height: ${size}px;
        border-radius: 50%;
        border: 3px solid white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        color: white;
        font-size: 12px;
        position: relative;
      ">
        <div style="
          position: absolute;
          top: -8px;
          right: -8px;
          background: #1f2937;
          color: white;
          border-radius: 50%;
          width: 16px;
          height: 16px;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 8px;
          border: 2px solid white;
        "></div>
      </div>
    `,
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
    popupAnchor: [0, -size / 2]
  })
}

// Risk-based icon
const createRiskIcon = (riskScore: number, demand: number, accessScore: number) => {
  if (typeof window === 'undefined') return null
  
  const L = require('leaflet')
  
  const riskColor = riskScore > 0.7 ? '#ef4444' : riskScore > 0.4 ? '#f59e0b' : '#10b981'
  const accessColor = accessScore > 0.7 ? '#10b981' : accessScore > 0.4 ? '#f59e0b' : '#ef4444'
  
  return L.divIcon({
    className: 'custom-risk-marker',
    html: `
      <div style="
        background-color: ${riskColor};
        width: 30px;
        height: 30px;
        border-radius: 50%;
        border: 3px solid white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        color: white;
        font-size: 14px;
        position: relative;
      ">
        <MapPin style="width: 16px; height: 16px;" />
        ${demand > 5 ? `
          <div style="
            position: absolute;
            top: -6px;
            right: -6px;
            background: #3b82f6;
            color: white;
            border-radius: 50%;
            width: 18px;
            height: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            border: 2px solid white;
            font-weight: bold;
          ">${demand}</div>
        ` : ''}
        <div style="
          position: absolute;
          bottom: -6px;
          left: -6px;
          background: ${accessColor};
          color: white;
          border-radius: 50%;
          width: 18px;
          height: 18px;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 10px;
          border: 2px solid white;
          font-weight: bold;
        ">A</div>
      </div>
    `,
    iconSize: [30, 30],
    iconAnchor: [15, 15],
    popupAnchor: [0, -15]
  })
}

export function InteractiveMapView({ 
  activeRun, 
  riskShaper, 
  warmStart, 
  serviceTimeGNN, 
  locations: propLocations, 
  isLoading: propIsLoading 
}: InteractiveMapViewProps) {
  const [locations, setLocations] = useState<Location[]>([])
  const [vehicles, setVehicles] = useState<Vehicle[]>([])
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [analytics, setAnalytics] = useState<Analytics | null>(null)
  const [loading, setLoading] = useState(true)
  const [selectedLocation, setSelectedLocation] = useState<Location | null>(null)
  const [mapCenter, setMapCenter] = useState<[number, number]>([42.3601, -71.0589]) // Boston default
  const [zoom, setZoom] = useState(12)
  const [isMapLoaded, setIsMapLoaded] = useState(false)

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

  // Generate route connections between locations
  const routeConnections = useMemo(() => {
    if (locations.length < 2) return []
    
    const connections = []
    // Create connections between nearby locations (within 5km)
    for (let i = 0; i < locations.length; i++) {
      for (let j = i + 1; j < locations.length; j++) {
        const loc1 = locations[i]
        const loc2 = locations[j]
        
        // Calculate distance (simplified)
        const distance = Math.sqrt(
          Math.pow(loc1.lat - loc2.lat, 2) + Math.pow(loc1.lng - loc2.lng, 2)
        ) * 111 // Rough conversion to km
        
        if (distance < 5) { // Within 5km
          const riskScore = ((loc1.weather_risk + loc1.crime_risk + loc1.traffic_risk) / 3 + 
                           (loc2.weather_risk + loc2.crime_risk + loc2.traffic_risk) / 3) / 2
          
          connections.push({
            from: [loc1.lat, loc1.lng],
            to: [loc2.lat, loc2.lng],
            risk: riskScore,
            distance: distance,
            color: riskScore > 0.7 ? '#ef4444' : riskScore > 0.4 ? '#f59e0b' : '#10b981'
          })
        }
      }
    }
    return connections
  }, [locations])

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
        const routePoints = vehicleLocations.map(loc => [loc.lat, loc.lng] as [number, number])
        routes.push({
          vehicle,
          points: routePoints,
          color: routeColors[vehicleIndex % routeColors.length]
        })
      }
    })
    
    return routes
  }, [vehicles, locations, routeColors])

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

      {/* Interactive Map */}
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
            <div className="h-full w-full">
              <MapContainer
                center={mapCenter}
                zoom={zoom}
                style={{ height: '100%', width: '100%' }}
                className="z-0"
                whenReady={() => setIsMapLoaded(true)}
              >
                <TileLayer
                  attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />
                
                {/* Location Markers */}
                {locations.map((location, index) => {
                  const prediction = predictions.find(p => p.location_id === location.id)
                  const riskScore = (location.weather_risk + location.crime_risk + location.traffic_risk) / 3
                  const icon = createRiskIcon(riskScore, location.demand, location.access_score)
                  
                  return (
                    <Marker
                      key={location.id}
                      position={[location.lat, location.lng]}
                      icon={icon}
                      eventHandlers={{
                        click: () => setSelectedLocation(location)
                      }}
                    >
                      <Popup>
                        <div className="p-2 min-w-[200px]">
                          <h3 className="font-semibold text-sm mb-2">{location.name}</h3>
                          <div className="space-y-1 text-xs">
                            <div className="flex justify-between">
                              <span>Demand:</span>
                              <span className="font-medium">{location.demand}</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Priority:</span>
                              <span className="font-medium">{location.priority}</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Service Time:</span>
                              <span className="font-medium">{location.service_min} min</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Risk Score:</span>
                              <span className="font-medium">{(riskScore * 100).toFixed(0)}%</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Access Score:</span>
                              <span className="font-medium">{(location.access_score * 100).toFixed(0)}%</span>
                            </div>
                            {prediction && (
                              <div className="pt-1 border-t">
                                <div className="flex justify-between">
                                  <span>AI Prediction:</span>
                                  <span className="font-medium">{prediction.predicted_time.toFixed(1)} min</span>
                                </div>
                                <div className="flex justify-between">
                                  <span>Confidence:</span>
                                  <span className="font-medium">{(prediction.confidence * 100).toFixed(0)}%</span>
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      </Popup>
                      
                      <Tooltip direction="top" offset={[0, -10]} opacity={1}>
                        <div className="text-center">
                          <div className="font-semibold">{location.name}</div>
                          <div className="text-xs">Risk: {(riskScore * 100).toFixed(0)}%</div>
                        </div>
                      </Tooltip>
                    </Marker>
                  )
                })}
                
                {/* Route Connections */}
                {routeConnections.map((connection, index) => (
                  <Polyline
                    key={`connection-${index}`}
                    positions={[connection.from, connection.to]}
                    color={connection.color}
                    weight={3}
                    opacity={0.7}
                    dashArray="5, 5"
                  />
                ))}
                
                {/* Vehicle Routes */}
                {vehicleRoutes.map((route, index) => (
                  <Polyline
                    key={`route-${route.vehicle.id}`}
                    positions={route.points}
                    color={route.color}
                    weight={4}
                    opacity={0.8}
                  />
                ))}
              </MapContainer>
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
