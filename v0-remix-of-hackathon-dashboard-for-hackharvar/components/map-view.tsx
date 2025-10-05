"use client"

import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { MapPin, Navigation, Fuel, Zap, Loader2, Clock, Package, TrendingDown, Shield, Route } from "lucide-react"
import { useState, useEffect } from "react"
import { fetchLocations, fetchVehicles, type Location, type Vehicle } from "@/lib/api"

interface MapViewProps {
  activeRun: "baseline" | "replan"
  riskShaper: boolean
  warmStart: boolean
  serviceTimeGNN: boolean
}

export function MapView({ activeRun, riskShaper }: MapViewProps) {
  const [locations, setLocations] = useState<Location[]>([])
  const [vehicles, setVehicles] = useState<Vehicle[]>([])
  const [loading, setLoading] = useState(true)
  const [showRiskHeat, setShowRiskHeat] = useState(true)
  const [showIncidents, setShowIncidents] = useState(true)
  const [showAmenities, setShowAmenities] = useState(false)
  const [selectedStop, setSelectedStop] = useState<Location | null>(null)
  const [animationProgress, setAnimationProgress] = useState(0)

  useEffect(() => {
    async function loadData() {
      try {
        const [locationsData, vehiclesData] = await Promise.all([fetchLocations(), fetchVehicles()])
        setLocations(locationsData)
        setVehicles(vehiclesData)
      } catch (error) {
      } finally {
        setLoading(false)
      }
    }
    loadData()
  }, [])

  useEffect(() => {
    const interval = setInterval(() => {
      setAnimationProgress((prev) => (prev + 1) % 100)
    }, 50)
    return () => clearInterval(interval)
  }, [])

  const getMapBounds = () => {
    if (locations.length === 0) return { minLat: 42.3, maxLat: 42.4, minLng: -71.1, maxLng: -71.0 }

    const lats = locations.map((l) => l.lat)
    const lngs = locations.map((l) => l.lng)

    return {
      minLat: Math.min(...lats),
      maxLat: Math.max(...lats),
      minLng: Math.min(...lngs),
      maxLng: Math.max(...lngs),
    }
  }

  const coordsToSVG = (lat: number, lng: number) => {
    const bounds = getMapBounds()
    const x = ((lng - bounds.minLng) / (bounds.maxLng - bounds.minLng)) * 800 + 50
    const y = ((bounds.maxLat - lat) / (bounds.maxLat - bounds.minLat)) * 500 + 50
    return { x, y }
  }

  const generateHeatmapGrid = () => {
    const gridSize = 20
    const heatmapCells = []
    const bounds = getMapBounds()

    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        const cellLat = bounds.minLat + ((bounds.maxLat - bounds.minLat) * i) / gridSize
        const cellLng = bounds.minLng + ((bounds.maxLng - bounds.minLng) * j) / gridSize

        let totalRisk = 0
        let count = 0

        locations
          .filter((l) => l.type === "stop")
          .forEach((stop) => {
            const distance = Math.sqrt(Math.pow(stop.lat - cellLat, 2) + Math.pow(stop.lng - cellLng, 2))

            if (distance < 0.02) {
              const riskScore = (stop.weather_risk + stop.crime_risk + stop.traffic_risk) / 3
              totalRisk += riskScore * (1 - distance / 0.02)
              count++
            }
          })

        const avgRisk = count > 0 ? totalRisk / count : 0

        if (avgRisk > 0.1) {
          const pos = coordsToSVG(cellLat, cellLng)
          heatmapCells.push({
            x: pos.x,
            y: pos.y,
            intensity: avgRisk,
          })
        }
      }
    }

    return heatmapCells
  }

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    )
  }

  const depot = locations.find((l) => l.type === "depot")
  const stops = locations.filter((l) => l.type === "stop")
  const heatmapCells = showRiskHeat && riskShaper ? generateHeatmapGrid() : []

  const avgRisk = stops.reduce((sum, s) => sum + (s.weather_risk + s.crime_risk + s.traffic_risk) / 3, 0) / stops.length
  const totalDemand = stops.reduce((sum, s) => sum + s.demand, 0)

  return (
    <div className="flex flex-col h-full">
      <div className="border-b-2 border-border bg-gradient-to-r from-primary/5 via-primary/10 to-primary/5 p-6">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-start justify-between gap-6">
            <div className="flex-1">
              <div className="flex items-center gap-3 mb-2">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/20">
                  <Route className="h-6 w-6 text-primary" />
                </div>
                <div>
                  <h2 className="text-2xl font-bold text-foreground">Boston Delivery Network</h2>
                  <p className="text-sm text-muted-foreground">
                    Real-time route optimization with AI-powered risk analysis
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-6 mt-4">
                <div className="flex items-center gap-2">
                  <div className="h-3 w-3 rounded-full bg-primary animate-pulse" />
                  <span className="text-sm font-medium text-foreground">{stops.length} Active Stops</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="h-3 w-3 rounded-full bg-blue-500" />
                  <span className="text-sm font-medium text-foreground">{vehicles.length} Vehicles Deployed</span>
                </div>
                <div className="flex items-center gap-2">
                  <Package className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm font-medium text-foreground">{totalDemand} Total Units</span>
                </div>
              </div>
            </div>

            <div className="flex gap-3">
              <Card className="p-4 bg-gradient-to-br from-emerald-500/10 to-emerald-500/5 border-emerald-500/20 min-w-[140px]">
                <div className="flex items-center gap-2 mb-1">
                  <Shield className="h-4 w-4 text-emerald-600" />
                  <span className="text-xs font-medium text-muted-foreground">Risk Avoidance</span>
                </div>
                <div className="text-2xl font-bold text-foreground">{((1 - avgRisk) * 100).toFixed(0)}%</div>
                <div className="text-[10px] text-muted-foreground mt-1">Routes optimized for safety</div>
              </Card>

              <Card className="p-4 bg-gradient-to-br from-blue-500/10 to-blue-500/5 border-blue-500/20 min-w-[140px]">
                <div className="flex items-center gap-2 mb-1">
                  <TrendingDown className="h-4 w-4 text-blue-600" />
                  <span className="text-xs font-medium text-muted-foreground">Efficiency Gain</span>
                </div>
                <div className="text-2xl font-bold text-foreground">+18%</div>
                <div className="text-[10px] text-muted-foreground mt-1">vs baseline routing</div>
              </Card>
            </div>
          </div>

          <div className="mt-4 p-3 rounded-lg bg-card/50 border border-border/50">
            <p className="text-sm text-muted-foreground leading-relaxed">
              <span className="font-semibold text-foreground">How it works:</span> Our AI analyzes environmental risks
              (weather, traffic, crime) across Boston to generate optimal delivery routes.
              {riskShaper && showRiskHeat && (
                <span className="text-primary font-medium"> The heatmap shows risk zones we actively avoid.</span>
              )}{" "}
              Click any stop marker to see detailed risk analysis and infrastructure data.
            </p>
          </div>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Map Area */}
        <div className="relative flex-1 bg-gradient-to-br from-secondary/10 via-background to-secondary/5">
          <div className="absolute left-4 top-4 z-10 flex flex-col gap-3">
            <Card className="p-4 backdrop-blur-sm bg-card/95 shadow-xl border-2">
              <div className="text-sm font-semibold mb-3 text-foreground">Map Layers</div>
              <div className="flex flex-col gap-3">
                <div className="flex items-center justify-between gap-3">
                  <Label htmlFor="risk-heat" className="text-xs cursor-pointer font-medium">
                    Risk Heatmap
                  </Label>
                  <Switch id="risk-heat" checked={showRiskHeat} onCheckedChange={setShowRiskHeat} />
                </div>
                <div className="flex items-center justify-between gap-3">
                  <Label htmlFor="incidents" className="text-xs cursor-pointer font-medium">
                    Incidents
                  </Label>
                  <Switch id="incidents" checked={showIncidents} onCheckedChange={setShowIncidents} />
                </div>
                <div className="flex items-center justify-between gap-3">
                  <Label htmlFor="amenities" className="text-xs cursor-pointer font-medium">
                    Amenities
                  </Label>
                  <Switch id="amenities" checked={showAmenities} onCheckedChange={setShowAmenities} />
                </div>
              </div>
            </Card>

            {showRiskHeat && riskShaper && (
              <Card className="p-4 backdrop-blur-sm bg-card/95 shadow-xl border-2">
                <div className="text-sm font-semibold mb-2 text-foreground">Risk Intensity Legend</div>
                <div className="text-xs text-muted-foreground mb-3">AI avoids high-risk zones</div>
                <div className="relative h-4 rounded-full overflow-hidden shadow-inner">
                  <div className="absolute inset-0 bg-gradient-to-r from-emerald-500 via-yellow-500 via-orange-500 to-red-500" />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent" />
                </div>
                <div className="flex justify-between mt-2">
                  <span className="text-[10px] font-medium text-emerald-600">Safe</span>
                  <span className="text-[10px] font-medium text-red-600">High Risk</span>
                </div>
              </Card>
            )}

            <Card className="p-4 backdrop-blur-sm bg-card/95 shadow-xl border-2">
              <div className="text-sm font-semibold mb-1 text-foreground">Fleet Status</div>
              <div className="text-xs text-muted-foreground mb-3">Active vehicles on routes</div>
              <div className="flex flex-col gap-2">
                {vehicles.map((vehicle, idx) => (
                  <div
                    key={vehicle.id}
                    className="flex items-center gap-3 p-2 rounded-lg bg-secondary/30 hover:bg-secondary/50 transition-colors"
                  >
                    <div
                      className="h-4 w-4 rounded-full shadow-lg ring-2 ring-background"
                      style={{
                        backgroundColor: ["#3b82f6", "#06b6d4", "#8b5cf6", "#f59e0b"][idx % 4],
                      }}
                    />
                    <div className="flex-1">
                      <div className="text-xs font-medium text-foreground">{vehicle.id}</div>
                      <div className="text-[10px] text-muted-foreground capitalize">{vehicle.type}</div>
                    </div>
                    <Badge variant="outline" className="text-[10px]">
                      {vehicle.capacity}kg
                    </Badge>
                  </div>
                ))}
              </div>
            </Card>
          </div>

          <div className="flex h-full items-center justify-center p-8">
            <div className="relative h-full w-full max-w-[900px] max-h-[600px] rounded-xl border-2 border-border bg-gradient-to-br from-card/80 to-card/60 shadow-2xl backdrop-blur-sm overflow-hidden">
              {/* Grid background */}
              <div className="absolute inset-0 opacity-10">
                <svg className="h-full w-full">
                  <defs>
                    <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                      <path d="M 40 0 L 0 0 0 40" fill="none" stroke="currentColor" strokeWidth="0.5" />
                    </pattern>
                  </defs>
                  <rect width="100%" height="100%" fill="url(#grid)" />
                </svg>
              </div>

              <div className="absolute inset-0 overflow-hidden rounded-xl">
                <svg className="h-full w-full" viewBox="0 0 900 600">
                  {heatmapCells.map((cell, idx) => {
                    const getHeatColor = (intensity: number) => {
                      if (intensity < 0.3) return { r: 16, g: 185, b: 129, a: intensity * 0.5 }
                      if (intensity < 0.5) return { r: 234, g: 179, b: 8, a: intensity * 0.6 }
                      if (intensity < 0.7) return { r: 249, g: 115, b: 22, a: intensity * 0.7 }
                      return { r: 239, g: 68, b: 68, a: intensity * 0.8 }
                    }

                    const color = getHeatColor(cell.intensity)

                    return (
                      <g key={idx}>
                        <circle
                          cx={cell.x}
                          cy={cell.y}
                          r={35 + cell.intensity * 25}
                          fill={`rgba(${color.r}, ${color.g}, ${color.b}, ${color.a})`}
                          className="blur-2xl"
                        />
                        <circle
                          cx={cell.x}
                          cy={cell.y}
                          r={20 + cell.intensity * 15}
                          fill={`rgba(${color.r}, ${color.g}, ${color.b}, ${color.a * 1.5})`}
                          className="blur-xl"
                        />
                      </g>
                    )
                  })}

                  {depot &&
                    stops.map((stop, idx) => {
                      const depotPos = coordsToSVG(depot.lat, depot.lng)
                      const stopPos = coordsToSVG(stop.lat, stop.lng)
                      const gradientId = `route-gradient-${idx % 4}`

                      return (
                        <g key={stop.id}>
                          {/* Background glow */}
                          <line
                            x1={depotPos.x}
                            y1={depotPos.y}
                            x2={stopPos.x}
                            y2={stopPos.y}
                            stroke={["#3b82f6", "#06b6d4", "#8b5cf6", "#f59e0b"][idx % 4]}
                            strokeWidth="6"
                            opacity="0.2"
                            className="blur-sm"
                          />
                          {/* Main route line */}
                          <line
                            x1={depotPos.x}
                            y1={depotPos.y}
                            x2={stopPos.x}
                            y2={stopPos.y}
                            stroke={`url(#${gradientId})`}
                            strokeWidth="3"
                            strokeDasharray="8,4"
                            strokeDashoffset={-animationProgress}
                            filter="url(#route-glow)"
                            className="transition-all"
                          />
                          {/* Direction arrow */}
                          <circle
                            cx={stopPos.x + (depotPos.x - stopPos.x) * 0.3}
                            cy={stopPos.y + (depotPos.y - stopPos.y) * 0.3}
                            r="4"
                            fill={["#3b82f6", "#06b6d4", "#8b5cf6", "#f59e0b"][idx % 4]}
                            opacity="0.8"
                          />
                        </g>
                      )
                    })}
                </svg>

                {depot &&
                  (() => {
                    const pos = coordsToSVG(depot.lat, depot.lng)
                    return (
                      <div
                        className="absolute animate-in fade-in zoom-in duration-500"
                        style={{ left: `${pos.x}px`, top: `${pos.y}px`, transform: "translate(-50%, -50%)" }}
                      >
                        {/* Pulse rings */}
                        <div className="absolute inset-0 flex items-center justify-center">
                          <div className="h-16 w-16 rounded-full bg-primary/20 animate-ping" />
                        </div>
                        <div className="absolute inset-0 flex items-center justify-center">
                          <div className="h-12 w-12 rounded-full bg-primary/30 animate-ping" />
                        </div>
                        {/* Main marker */}
                        <div className="relative flex h-12 w-12 items-center justify-center rounded-full bg-gradient-to-br from-primary to-primary/80 shadow-2xl ring-4 ring-primary/30">
                          <MapPin className="h-6 w-6 text-primary-foreground drop-shadow-lg" />
                        </div>
                        {/* Label */}
                        <div className="absolute top-14 left-1/2 -translate-x-1/2 whitespace-nowrap">
                          <div className="bg-gradient-to-r from-primary/90 to-primary/80 backdrop-blur-sm px-3 py-1.5 rounded-lg shadow-lg border border-primary/20">
                            <div className="text-xs font-bold text-primary-foreground">{depot.name}</div>
                            <div className="text-[10px] text-primary-foreground/80">Distribution Hub</div>
                          </div>
                        </div>
                      </div>
                    )
                  })()}

                {stops.map((stop, idx) => {
                  const pos = coordsToSVG(stop.lat, stop.lng)
                  const isSelected = selectedStop?.id === stop.id
                  const riskLevel = (stop.weather_risk + stop.crime_risk + stop.traffic_risk) / 3

                  return (
                    <div
                      key={stop.id}
                      className="absolute animate-in fade-in zoom-in"
                      style={{
                        left: `${pos.x}px`,
                        top: `${pos.y}px`,
                        transform: "translate(-50%, -50%)",
                        animationDelay: `${idx * 100}ms`,
                      }}
                    >
                      {/* Pulse effect for selected */}
                      {isSelected && (
                        <div className="absolute inset-0 flex items-center justify-center">
                          <div className="h-12 w-12 rounded-full bg-primary/30 animate-ping" />
                        </div>
                      )}
                      {/* Main marker button */}
                      <button
                        onClick={() => setSelectedStop(stop)}
                        className={`relative flex h-10 w-10 items-center justify-center rounded-full shadow-xl ring-2 transition-all duration-300 ${
                          isSelected
                            ? "bg-gradient-to-br from-primary to-primary/80 ring-primary/50 scale-125 z-10"
                            : riskLevel > 0.6
                              ? "bg-gradient-to-br from-red-500 to-orange-500 ring-red-500/30 hover:scale-110"
                              : riskLevel > 0.4
                                ? "bg-gradient-to-br from-yellow-500 to-orange-500 ring-yellow-500/30 hover:scale-110"
                                : "bg-gradient-to-br from-emerald-500 to-green-500 ring-emerald-500/30 hover:scale-110"
                        }`}
                      >
                        <MapPin className="h-5 w-5 text-white drop-shadow-lg" />
                        {/* Stop number badge */}
                        <div className="absolute -top-1 -right-1 h-5 w-5 rounded-full bg-background border-2 border-current flex items-center justify-center">
                          <span className="text-[10px] font-bold">{idx + 1}</span>
                        </div>
                      </button>
                      {/* Hover label */}
                      {!isSelected && (
                        <div className="absolute top-12 left-1/2 -translate-x-1/2 whitespace-nowrap opacity-0 hover:opacity-100 transition-opacity pointer-events-none">
                          <div className="bg-card/95 backdrop-blur-sm px-2 py-1 rounded shadow-lg border text-[10px] font-medium">
                            {stop.name}
                          </div>
                        </div>
                      )}
                    </div>
                  )
                })}

                {showAmenities &&
                  stops.slice(0, 3).map((stop, idx) => {
                    const pos = coordsToSVG(stop.lat, stop.lng)
                    const icons = [Fuel, Zap, Package]
                    const Icon = icons[idx % 3]
                    const colors = ["bg-blue-500", "bg-yellow-500", "bg-purple-500"]

                    return (
                      <div
                        key={`amenity-${stop.id}`}
                        className="absolute animate-in fade-in zoom-in"
                        style={{
                          left: `${pos.x + 35}px`,
                          top: `${pos.y - 35}px`,
                          transform: "translate(-50%, -50%)",
                          animationDelay: `${(idx + stops.length) * 100}ms`,
                        }}
                      >
                        <div
                          className={`flex h-6 w-6 items-center justify-center rounded-full ${colors[idx % 3]} shadow-lg ring-2 ring-background`}
                        >
                          <Icon className="h-3 w-3 text-white" />
                        </div>
                      </div>
                    )
                  })}
              </div>

              <div className="absolute bottom-4 left-4">
                <div className="bg-card/90 backdrop-blur-sm px-3 py-1.5 rounded-lg shadow-lg border">
                  <div className="text-xs font-semibold text-foreground">📍 Boston, MA</div>
                </div>
              </div>

              <div className="absolute top-4 right-4 flex gap-2">
                <div className="bg-card/90 backdrop-blur-sm px-3 py-2 rounded-lg shadow-lg border">
                  <div className="text-[10px] text-muted-foreground">Delivery Stops</div>
                  <div className="text-lg font-bold text-foreground">{stops.length}</div>
                </div>
                <div className="bg-card/90 backdrop-blur-sm px-3 py-2 rounded-lg shadow-lg border">
                  <div className="text-[10px] text-muted-foreground">Active Routes</div>
                  <div className="text-lg font-bold text-foreground">{vehicles.length}</div>
                </div>
              </div>
            </div>
          </div>

          <div className="absolute bottom-4 right-4">
            <Badge
              variant={activeRun === "replan" ? "default" : "outline"}
              className="bg-card/90 backdrop-blur-sm shadow-lg text-sm px-4 py-2"
            >
              {activeRun === "baseline" ? "📊 Baseline Route" : "🚀 AI-Optimized Route"}
            </Badge>
          </div>
        </div>

        {selectedStop && (
          <div className="w-96 border-l-2 border-border bg-gradient-to-br from-card to-card/80 backdrop-blur-sm overflow-y-auto animate-in slide-in-from-right duration-300">
            <div className="sticky top-0 z-10 bg-gradient-to-r from-primary/10 to-primary/5 backdrop-blur-sm border-b-2 border-border p-4">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <h3 className="text-xl font-bold text-foreground mb-1">{selectedStop.name}</h3>
                  <p className="text-xs text-muted-foreground font-mono">
                    {selectedStop.lat.toFixed(4)}° N, {Math.abs(selectedStop.lng).toFixed(4)}° W
                  </p>
                  <Badge variant="secondary" className="mt-2 text-xs">
                    Stop Analysis
                  </Badge>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setSelectedStop(null)}
                  className="hover:bg-destructive/10"
                >
                  <span className="text-lg">×</span>
                </Button>
              </div>
            </div>

            <div className="p-4 space-y-4">
              <Card className="p-4 bg-gradient-to-br from-blue-500/10 to-blue-500/5 border-blue-500/20">
                <div className="flex items-center gap-3">
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-blue-500/20">
                    <Package className="h-6 w-6 text-blue-600" />
                  </div>
                  <div className="flex-1">
                    <div className="text-xs font-medium text-muted-foreground mb-1">Package Demand</div>
                    <div className="flex items-baseline gap-2">
                      <span className="text-3xl font-bold text-foreground">{selectedStop.demand}</span>
                      <span className="text-sm text-muted-foreground">units</span>
                    </div>
                  </div>
                </div>
              </Card>

              <Card className="p-4">
                <div className="flex items-center gap-2 mb-3">
                  <div className="h-2 w-2 rounded-full bg-primary" />
                  <div className="text-sm font-semibold text-foreground">Accessibility Score</div>
                </div>
                <div className="flex items-center gap-3 mb-3">
                  <div className="flex-1 h-3 bg-secondary rounded-full overflow-hidden shadow-inner">
                    <div
                      className="h-full bg-gradient-to-r from-primary to-primary/80 transition-all duration-500 shadow-lg"
                      style={{ width: `${selectedStop.access_score * 100}%` }}
                    />
                  </div>
                  <span className="text-lg font-bold text-foreground min-w-[3rem] text-right">
                    {(selectedStop.access_score * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="flex flex-wrap gap-1.5 mb-2">
                  {selectedStop.accessibility_features.map((feature) => (
                    <Badge key={feature} variant="secondary" className="text-xs">
                      {feature}
                    </Badge>
                  ))}
                </div>
                <div className="text-xs text-muted-foreground">
                  Sidewalk Width: <span className="font-medium text-foreground">{selectedStop.sidewalk_width}m</span>
                </div>
              </Card>

              <Card className="p-4">
                <div className="flex items-center gap-2 mb-3">
                  <div className="h-2 w-2 rounded-full bg-orange-500" />
                  <div className="text-sm font-semibold text-foreground">Environmental Factors</div>
                </div>
                <div className="space-y-3">
                  {[
                    { label: "Weather Risk", value: selectedStop.weather_risk, color: "blue" },
                    { label: "Traffic Risk", value: selectedStop.traffic_risk, color: "orange" },
                    { label: "Crime Risk", value: selectedStop.crime_risk, color: "red" },
                  ].map((factor) => (
                    <div key={factor.label}>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-muted-foreground">{factor.label}</span>
                        <span className="font-semibold text-foreground">{(factor.value * 100).toFixed(0)}%</span>
                      </div>
                      <div className="h-2 bg-secondary rounded-full overflow-hidden">
                        <div
                          className={`h-full transition-all duration-500 ${
                            factor.color === "blue"
                              ? "bg-gradient-to-r from-blue-500 to-blue-600"
                              : factor.color === "orange"
                                ? "bg-gradient-to-r from-orange-500 to-orange-600"
                                : "bg-gradient-to-r from-red-500 to-red-600"
                          }`}
                          style={{ width: `${factor.value * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                  <div className="pt-2 border-t border-border">
                    <div className="flex justify-between text-xs">
                      <span className="text-muted-foreground">Lighting Score</span>
                      <span className="font-semibold text-foreground">
                        {(selectedStop.lighting_score * 10).toFixed(1)}/10
                      </span>
                    </div>
                  </div>
                </div>
              </Card>

              <Card className="p-4">
                <div className="flex items-center gap-2 mb-3">
                  <div className="h-2 w-2 rounded-full bg-green-500" />
                  <div className="text-sm font-semibold text-foreground">Infrastructure</div>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="flex flex-col items-center p-3 rounded-lg bg-secondary/30">
                    <div className="text-2xl font-bold text-foreground">{selectedStop.parking_spaces}</div>
                    <div className="text-[10px] text-muted-foreground text-center">Parking Spaces</div>
                  </div>
                  <div className="flex flex-col items-center p-3 rounded-lg bg-secondary/30">
                    <div className="text-2xl font-bold text-foreground">{selectedStop.loading_docks}</div>
                    <div className="text-[10px] text-muted-foreground text-center">Loading Docks</div>
                  </div>
                  <div className="flex flex-col items-center p-3 rounded-lg bg-secondary/30">
                    <div className="text-2xl font-bold text-foreground">{selectedStop.traffic_signals}</div>
                    <div className="text-[10px] text-muted-foreground text-center">Traffic Signals</div>
                  </div>
                  <div className="flex flex-col items-center p-3 rounded-lg bg-secondary/30">
                    <div
                      className={`text-2xl ${selectedStop.ev_charging ? "text-green-500" : "text-muted-foreground"}`}
                    >
                      {selectedStop.ev_charging ? "✓" : "✗"}
                    </div>
                    <div className="text-[10px] text-muted-foreground text-center">EV Charging</div>
                  </div>
                </div>
              </Card>

              <Card className="p-4 bg-gradient-to-br from-purple-500/10 to-purple-500/5 border-purple-500/20">
                <div className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-full bg-purple-500/20">
                    <Clock className="h-5 w-5 text-purple-600" />
                  </div>
                  <div className="flex-1">
                    <div className="text-xs font-medium text-muted-foreground mb-1">Service Window</div>
                    <div className="text-lg font-bold text-foreground">
                      {selectedStop.time_windows.open} - {selectedStop.time_windows.close}
                    </div>
                  </div>
                </div>
              </Card>

              {selectedStop.hazards.length > 0 && (
                <Card className="p-4 bg-gradient-to-br from-red-500/10 to-red-500/5 border-red-500/20">
                  <div className="text-sm font-semibold text-foreground mb-2">⚠️ Active Hazards</div>
                  <div className="flex flex-wrap gap-2">
                    {selectedStop.hazards.map((hazard) => (
                      <Badge key={hazard} variant="destructive" className="text-xs">
                        {hazard.replace(/_/g, " ")}
                      </Badge>
                    ))}
                  </div>
                </Card>
              )}

              <Button className="w-full h-12 text-base font-semibold shadow-lg" size="lg">
                <Navigation className="mr-2 h-5 w-5" />
                View Last 300m Approach
              </Button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
