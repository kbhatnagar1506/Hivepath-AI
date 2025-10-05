"use client"

import React, { useState, useEffect, useMemo } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Loader2, MapPin, Truck, Route, Clock, Users, Package, Navigation, Zap, AlertTriangle, ExternalLink, Settings } from "lucide-react"
import { type Location, type Vehicle, type Prediction, type Analytics } from "@/lib/api"
import { fetchLocations, fetchVehicles, fetchPredictions, fetchAnalytics } from "@/lib/api"

interface GoogleMapsFallbackProps {
  activeRun: "baseline" | "replan"
  riskShaper: boolean
  warmStart: boolean
  serviceTimeGNN: boolean
  locations?: Location[]
  isLoading?: boolean
}

export function GoogleMapsFallback({ 
  activeRun, 
  riskShaper, 
  warmStart, 
  serviceTimeGNN, 
  locations: propLocations, 
  isLoading: propIsLoading 
}: GoogleMapsFallbackProps) {
  const [locations, setLocations] = useState<Location[]>([])
  const [vehicles, setVehicles] = useState<Vehicle[]>([])
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [analytics, setAnalytics] = useState<Analytics | null>(null)
  const [loading, setLoading] = useState(true)
  const [selectedLocation, setSelectedLocation] = useState<Location | null>(null)
  const [mapCenter, setMapCenter] = useState<[number, number]>([42.3601, -71.0589]) // Boston default
  const [zoom, setZoom] = useState(12)
  const [mapMode, setMapMode] = useState<'roadmap' | 'satellite' | 'hybrid' | 'terrain'>('roadmap')
  const [showTraffic, setShowTraffic] = useState(false)
  const [showTransit, setShowTransit] = useState(false)
  const [apiKeyStatus, setApiKeyStatus] = useState<'checking' | 'valid' | 'invalid' | 'missing'>('checking')

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
          const lats = locations.map(loc => loc.lat)
          const lngs = locations.map(loc => loc.lng)
          const avgLat = lats.reduce((sum, lat) => sum + lat, 0) / lats.length
          const avgLng = lngs.reduce((sum, lng) => sum + lng, 0) / lngs.length
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

  // Check API key status
  useEffect(() => {
    const apiKey = process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY
    if (!apiKey || apiKey === 'YOUR_API_KEY_HERE') {
      setApiKeyStatus('missing')
    } else {
      // Test the API key with a simple request
      const testUrl = `https://www.google.com/maps/embed/v1/view?key=${apiKey}&center=42.3601,-71.0589&zoom=12`
      fetch(testUrl)
        .then(response => {
          if (response.ok) {
            setApiKeyStatus('valid')
          } else {
            setApiKeyStatus('invalid')
          }
        })
        .catch(() => {
          setApiKeyStatus('invalid')
        })
    }
  }, [])

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

  // Generate Google Maps embed URL
  const generateMapUrl = () => {
    const apiKey = process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY
    if (!apiKey || apiKey === 'YOUR_API_KEY_HERE') {
      return null
    }
    
    const baseUrl = "https://www.google.com/maps/embed/v1/view"
    const params = new URLSearchParams({
      key: apiKey,
      center: `${mapCenter[0]},${mapCenter[1]}`,
      zoom: zoom.toString(),
      maptype: mapMode,
    })

    if (showTraffic) params.append('traffic', 'on')
    if (showTransit) params.append('transit', 'on')

    return `${baseUrl}?${params.toString()}`
  }

  // Generate directions URL for vehicle routes
  const generateDirectionsUrl = (vehicle: Vehicle, vehicleLocations: Location[]) => {
    if (vehicleLocations.length < 2) return null
    
    const origin = `${vehicleLocations[0].lat},${vehicleLocations[0].lng}`
    const destination = `${vehicleLocations[vehicleLocations.length - 1].lat},${vehicleLocations[vehicleLocations.length - 1].lng}`
    const waypoints = vehicleLocations.slice(1, -1).map(loc => `${loc.lat},${loc.lng}`).join('|')
    
    const baseUrl = "https://www.google.com/maps/dir"
    const params = new URLSearchParams({
      api: '1',
      origin: origin,
      destination: destination,
    })
    
    if (waypoints) params.append('waypoints', waypoints)
    
    return `${baseUrl}?${params.toString()}`
  }

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
        routes.push({
          vehicle,
          locations: vehicleLocations,
          color: routeColors[vehicleIndex % routeColors.length],
          directionsUrl: generateDirectionsUrl(vehicle, vehicleLocations)
        })
      }
    })
    
    return routes
  }, [vehicles, locations, routeColors])

  // Convert lat/lng to screen coordinates for fallback map
  const latLngToScreen = (lat: number, lng: number) => {
    const bounds = {
      north: mapCenter[0] + 0.05,
      south: mapCenter[0] - 0.05,
      east: mapCenter[1] + 0.05,
      west: mapCenter[1] - 0.05
    }
    
    const x = ((lng - bounds.west) / (bounds.east - bounds.west)) * 100
    const y = ((bounds.north - lat) / (bounds.north - bounds.south)) * 100
    return { x: Math.max(0, Math.min(100, x)), y: Math.max(0, Math.min(100, y)) }
  }

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
            <p className="text-muted-foreground">Loading map integration...</p>
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

      {/* Map Integration */}
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
            {/* API Key Status Banner */}
            {apiKeyStatus === 'missing' && (
              <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-20">
                <Card className="bg-yellow-50 border-yellow-200">
                  <CardContent className="p-3">
                    <div className="flex items-center gap-2 text-yellow-800">
                      <Settings className="h-4 w-4" />
                      <span className="text-sm font-medium">Google Maps API Key Required</span>
                    </div>
                    <p className="text-xs text-yellow-700 mt-1">
                      Add NEXT_PUBLIC_GOOGLE_MAPS_API_KEY to .env.local for full Google Maps integration
                    </p>
                  </CardContent>
                </Card>
              </div>
            )}

            {apiKeyStatus === 'invalid' && (
              <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-20">
                <Card className="bg-red-50 border-red-200">
                  <CardContent className="p-3">
                    <div className="flex items-center gap-2 text-red-800">
                      <AlertTriangle className="h-4 w-4" />
                      <span className="text-sm font-medium">Invalid Google Maps API Key</span>
                    </div>
                    <p className="text-xs text-red-700 mt-1">
                      Please check your API key configuration and ensure Maps Embed API is enabled
                    </p>
                  </CardContent>
                </Card>
              </div>
            )}

            {/* Map Container */}
            <div className="h-full w-full relative">
              {apiKeyStatus === 'valid' && generateMapUrl() ? (
                // Real Google Maps
                <iframe
                  src={generateMapUrl()}
                  width="100%"
                  height="100%"
                  style={{ border: 0 }}
                  allowFullScreen
                  loading="lazy"
                  referrerPolicy="no-referrer-when-downgrade"
                  title="HivePath AI Infrastructure Network Map"
                  className="rounded-lg"
                />
              ) : (
                // Fallback Map with Street-like Background
                <div className="h-full w-full relative bg-gradient-to-br from-blue-100 to-green-100">
                  {/* Street-like background */}
                  <div className="absolute inset-0 opacity-30">
                    <div className="grid grid-cols-20 grid-rows-16 h-full w-full">
                      {Array.from({ length: 320 }).map((_, i) => (
                        <div key={i} className="border border-slate-300/30" />
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

                  {/* Fallback mode indicator */}
                  <div className="absolute top-4 left-4 z-10">
                    <Card className="bg-blue-50 border-blue-200">
                      <CardContent className="p-2">
                        <div className="flex items-center gap-2 text-blue-800">
                          <MapPin className="h-4 w-4" />
                          <span className="text-sm font-medium">Interactive Map View</span>
                        </div>
                        <p className="text-xs text-blue-700">
                          {apiKeyStatus === 'missing' ? 'Configure API key for Google Maps' : 'Using fallback map view'}
                        </p>
                      </CardContent>
                    </Card>
                  </div>
                </div>
              )}
              
              {/* Overlay Controls */}
              <div className="absolute top-4 right-4 z-10 flex flex-col gap-2">
                {/* Map Mode Controls */}
                <div className="flex gap-1 bg-white rounded-lg shadow-lg p-1">
                  {(['roadmap', 'satellite', 'hybrid', 'terrain'] as const).map((mode) => (
                    <Button
                      key={mode}
                      variant={mapMode === mode ? "default" : "ghost"}
                      size="sm"
                      onClick={() => setMapMode(mode)}
                      className="text-xs px-2 py-1 h-8"
                    >
                      {mode.charAt(0).toUpperCase() + mode.slice(1)}
                    </Button>
                  ))}
                </div>
                
                {/* Zoom Controls */}
                <div className="flex flex-col gap-1 bg-white rounded-lg shadow-lg p-1">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setZoom(Math.min(20, zoom + 1))}
                    className="h-8 w-8 p-0"
                  >
                    +
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setZoom(Math.max(1, zoom - 1))}
                    className="h-8 w-8 p-0"
                  >
                    -
                  </Button>
                </div>
                
                {/* Layer Controls */}
                <div className="flex flex-col gap-1 bg-white rounded-lg shadow-lg p-1">
                  <Button
                    variant={showTraffic ? "default" : "outline"}
                    size="sm"
                    onClick={() => setShowTraffic(!showTraffic)}
                    className="text-xs px-2 py-1 h-8"
                  >
                    Traffic
                  </Button>
                  <Button
                    variant={showTransit ? "default" : "outline"}
                    size="sm"
                    onClick={() => setShowTransit(!showTransit)}
                    className="text-xs px-2 py-1 h-8"
                  >
                    Transit
                  </Button>
                </div>
              </div>
            </div>

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
                  
                  <Button
                    variant="outline"
                    size="sm"
                    className="w-full"
                    onClick={() => {
                      const url = `https://www.google.com/maps/search/?api=1&query=${selectedLocation.lat},${selectedLocation.lng}`
                      window.open(url, '_blank')
                    }}
                  >
                    <ExternalLink className="h-4 w-4 mr-2" />
                    View in Google Maps
                  </Button>
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
                  {vehicleRoutes.map((route, index) => (
                    <div key={route.vehicle.id} className="flex items-center gap-3 p-2 rounded-lg hover:bg-slate-50 transition-colors">
                      <div 
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: route.color }}
                      />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-slate-900 truncate">{route.vehicle.name}</p>
                        <p className="text-xs text-slate-500">{route.vehicle.type} • {route.vehicle.capacity} capacity</p>
                      </div>
                      <div className="flex gap-1">
                        <Badge variant="outline" className="text-xs">
                          {route.locations.length} stops
                        </Badge>
                        {route.directionsUrl && (
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => window.open(route.directionsUrl, '_blank')}
                            className="h-6 w-6 p-0"
                          >
                            <ExternalLink className="h-3 w-3" />
                          </Button>
                        )}
                      </div>
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
                  <p className="text-xs font-medium text-slate-700">Map Features</p>
                  <div className="flex items-center gap-2">
                    <div className="h-3 w-3 rounded-full bg-blue-500" />
                    <span className="text-xs text-slate-600">Vehicle Routes</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="h-3 w-3 rounded-full bg-gray-400" />
                    <span className="text-xs text-slate-600">Location Connections</span>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <p className="text-xs font-medium text-slate-700">Controls</p>
                  <div className="text-xs text-slate-600">
                    • Click map modes to switch views<br/>
                    • Use +/- to zoom in/out<br/>
                    • Toggle traffic and transit layers<br/>
                    • Click locations for details
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Location Markers Overlay */}
            <div className="absolute inset-0 pointer-events-none z-5">
              {locations.map((location, index) => {
                const prediction = predictions.find(p => p.location_id === location.id)
                const riskScore = (location.weather_risk + location.crime_risk + location.traffic_risk) / 3
                const markerColor = riskScore > 0.7 ? '#ef4444' : riskScore > 0.4 ? '#f59e0b' : '#10b981'
                const accessColor = location.access_score > 0.7 ? '#10b981' : location.access_score > 0.4 ? '#f59e0b' : '#ef4444'
                
                // Convert lat/lng to screen position
                const coords = latLngToScreen(location.lat, location.lng)
                
                return (
                  <div
                    key={location.id}
                    className="absolute transform -translate-x-1/2 -translate-y-1/2 cursor-pointer group pointer-events-auto"
                    style={{ 
                      left: `${coords.x}%`, 
                      top: `${coords.y}%` 
                    }}
                    onClick={() => setSelectedLocation(location)}
                  >
                    <button className="relative">
                      {/* Main marker */}
                      <div 
                        className={`w-8 h-8 rounded-full border-3 border-white shadow-lg flex items-center justify-center text-white font-bold text-sm relative transition-all duration-200 ${
                          selectedLocation?.id === location.id ? 'scale-125 shadow-xl' : 'hover:scale-110'
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
              })}
            </div>
          </>
        )}
      </div>
    </div>
  )
}
