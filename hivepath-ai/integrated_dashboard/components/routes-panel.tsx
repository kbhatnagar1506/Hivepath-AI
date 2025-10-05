"use client"

import { useEffect, useState } from "react"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Truck, MapPin, Clock, DollarSign, Leaf, TrendingDown, Navigation, AlertTriangle } from "lucide-react"
import { fetchLocations, fetchVehicles, type Location, type Vehicle } from "@/lib/api"

interface RouteStop {
  location: Location
  arrival_time: string
  departure_time: string
  service_time: number
  load_after: number
}

interface Route {
  id: string
  vehicle: Vehicle
  stops: RouteStop[]
  total_distance: number
  total_time: number
  total_cost: number
  co2_emissions: number
  efficiency_score: number
  risk_score: number
  baseline_comparison: {
    distance_saved: number
    time_saved: number
    cost_saved: number
    co2_saved: number
  }
}

export function RoutesPanel() {
  const [routes, setRoutes] = useState<Route[]>([])
  const [selectedRoute, setSelectedRoute] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function loadRoutes() {
      try {
        const [locations, vehicles] = await Promise.all([fetchLocations(), fetchVehicles()])

        // Generate mock optimized routes
        const mockRoutes: Route[] = [
          {
            id: "R1",
            vehicle: vehicles[0],
            stops: [
              {
                location: locations[1],
                arrival_time: "08:15",
                departure_time: "08:29",
                service_time: 14.3,
                load_after: 250,
              },
              {
                location: locations[3],
                arrival_time: "09:05",
                departure_time: "09:19",
                service_time: 14.1,
                load_after: 105,
              },
            ],
            total_distance: 18.5,
            total_time: 64,
            total_cost: 29.53,
            co2_emissions: 12.4,
            efficiency_score: 0.92,
            risk_score: 0.28,
            baseline_comparison: {
              distance_saved: 3.2,
              time_saved: 12,
              cost_saved: 5.8,
              co2_saved: 2.1,
            },
          },
          {
            id: "R2",
            vehicle: vehicles[1],
            stops: [
              {
                location: locations[2],
                arrival_time: "08:30",
                departure_time: "08:44",
                service_time: 13.9,
                load_after: 60,
              },
              {
                location: locations[4],
                arrival_time: "09:15",
                departure_time: "09:29",
                service_time: 14.3,
                load_after: 0,
              },
            ],
            total_distance: 15.2,
            total_time: 59,
            total_cost: 21.64,
            co2_emissions: 9.8,
            efficiency_score: 0.88,
            risk_score: 0.35,
            baseline_comparison: {
              distance_saved: 2.8,
              time_saved: 9,
              cost_saved: 4.2,
              co2_saved: 1.8,
            },
          },
          {
            id: "R3",
            vehicle: vehicles[2],
            stops: [
              {
                location: locations[5],
                arrival_time: "07:45",
                departure_time: "07:59",
                service_time: 13.9,
                load_after: 0,
              },
            ],
            total_distance: 12.8,
            total_time: 42,
            total_cost: 17.99,
            co2_emissions: 8.5,
            efficiency_score: 0.95,
            risk_score: 0.18,
            baseline_comparison: {
              distance_saved: 1.9,
              time_saved: 7,
              cost_saved: 3.1,
              co2_saved: 1.3,
            },
          },
        ]

        setRoutes(mockRoutes)
        setSelectedRoute(mockRoutes[0].id)
      } catch (error) {
        console.error("Failed to load routes:", error)
      } finally {
        setLoading(false)
      }
    }
    loadRoutes()
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
      </div>
    )
  }

  const selectedRouteData = routes.find((r) => r.id === selectedRoute)

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Routes List */}
      <div className="lg:col-span-1 space-y-4">
        <h3 className="text-lg font-semibold text-foreground mb-4">Optimized Routes</h3>
        {routes.map((route) => (
          <Card
            key={route.id}
            className={`p-4 cursor-pointer transition-all hover:shadow-lg ${
              selectedRoute === route.id
                ? "border-primary bg-primary/5 shadow-md"
                : "border-border hover:border-primary/50"
            }`}
            onClick={() => setSelectedRoute(route.id)}
          >
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center gap-2">
                <div className="p-2 rounded-lg bg-gradient-to-br from-blue-500 to-blue-600">
                  <Truck className="h-4 w-4 text-white" />
                </div>
                <div>
                  <h4 className="font-semibold text-foreground">{route.id}</h4>
                  <p className="text-xs text-muted-foreground">{route.vehicle.id}</p>
                </div>
              </div>
              <Badge variant="secondary" className="bg-emerald-500/10 text-emerald-600 border-emerald-500/20">
                {Math.round(route.efficiency_score * 100)}% Efficient
              </Badge>
            </div>

            <div className="space-y-2 text-sm">
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground flex items-center gap-1">
                  <MapPin className="h-3 w-3" />
                  {route.stops.length} stops
                </span>
                <span className="font-medium text-foreground">{route.total_distance.toFixed(1)} km</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground flex items-center gap-1">
                  <Clock className="h-3 w-3" />
                  Duration
                </span>
                <span className="font-medium text-foreground">{route.total_time} min</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground flex items-center gap-1">
                  <TrendingDown className="h-3 w-3" />
                  Saved
                </span>
                <span className="font-medium text-emerald-600">{route.baseline_comparison.time_saved} min</span>
              </div>
            </div>
          </Card>
        ))}

        {/* Summary Card */}
        <Card className="p-4 bg-gradient-to-br from-blue-500/10 to-purple-500/10 border-blue-500/20">
          <h4 className="font-semibold text-foreground mb-3">Total Savings</h4>
          <div className="space-y-2 text-sm">
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Distance</span>
              <span className="font-medium text-emerald-600">
                -{routes.reduce((sum, r) => sum + r.baseline_comparison.distance_saved, 0).toFixed(1)} km
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Time</span>
              <span className="font-medium text-emerald-600">
                -{routes.reduce((sum, r) => sum + r.baseline_comparison.time_saved, 0)} min
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Cost</span>
              <span className="font-medium text-emerald-600">
                -${routes.reduce((sum, r) => sum + r.baseline_comparison.cost_saved, 0).toFixed(2)}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">CO₂</span>
              <span className="font-medium text-emerald-600">
                -{routes.reduce((sum, r) => sum + r.baseline_comparison.co2_saved, 0).toFixed(1)} kg
              </span>
            </div>
          </div>
        </Card>
      </div>

      {/* Route Details */}
      {selectedRouteData && (
        <div className="lg:col-span-2 space-y-6">
          {/* Route Header */}
          <Card className="p-6 bg-gradient-to-br from-blue-500/5 to-purple-500/5 border-blue-500/20">
            <div className="flex items-start justify-between mb-4">
              <div>
                <h3 className="text-2xl font-bold text-foreground mb-1">{selectedRouteData.id}</h3>
                <p className="text-muted-foreground">
                  {selectedRouteData.vehicle.type.charAt(0).toUpperCase() + selectedRouteData.vehicle.type.slice(1)} •{" "}
                  {selectedRouteData.vehicle.id} • {selectedRouteData.vehicle.fuel_type}
                </p>
              </div>
              <div className="flex gap-2">
                <Badge className="bg-emerald-500/10 text-emerald-600 border-emerald-500/20">
                  <TrendingDown className="h-3 w-3 mr-1" />
                  {Math.round(selectedRouteData.efficiency_score * 100)}% Efficient
                </Badge>
                <Badge
                  className={`${
                    selectedRouteData.risk_score < 0.3
                      ? "bg-emerald-500/10 text-emerald-600 border-emerald-500/20"
                      : selectedRouteData.risk_score < 0.5
                        ? "bg-yellow-500/10 text-yellow-600 border-yellow-500/20"
                        : "bg-red-500/10 text-red-600 border-red-500/20"
                  }`}
                >
                  <AlertTriangle className="h-3 w-3 mr-1" />
                  {Math.round(selectedRouteData.risk_score * 100)}% Risk
                </Badge>
              </div>
            </div>

            {/* Metrics Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="p-3 rounded-lg bg-background/50 border border-border">
                <div className="flex items-center gap-2 mb-1">
                  <Navigation className="h-4 w-4 text-blue-500" />
                  <span className="text-xs text-muted-foreground">Distance</span>
                </div>
                <p className="text-xl font-bold text-foreground">{selectedRouteData.total_distance.toFixed(1)} km</p>
                <p className="text-xs text-emerald-600">
                  -{selectedRouteData.baseline_comparison.distance_saved.toFixed(1)} km saved
                </p>
              </div>

              <div className="p-3 rounded-lg bg-background/50 border border-border">
                <div className="flex items-center gap-2 mb-1">
                  <Clock className="h-4 w-4 text-purple-500" />
                  <span className="text-xs text-muted-foreground">Time</span>
                </div>
                <p className="text-xl font-bold text-foreground">{selectedRouteData.total_time} min</p>
                <p className="text-xs text-emerald-600">
                  -{selectedRouteData.baseline_comparison.time_saved} min saved
                </p>
              </div>

              <div className="p-3 rounded-lg bg-background/50 border border-border">
                <div className="flex items-center gap-2 mb-1">
                  <DollarSign className="h-4 w-4 text-emerald-500" />
                  <span className="text-xs text-muted-foreground">Cost</span>
                </div>
                <p className="text-xl font-bold text-foreground">${selectedRouteData.total_cost.toFixed(2)}</p>
                <p className="text-xs text-emerald-600">
                  -${selectedRouteData.baseline_comparison.cost_saved.toFixed(2)} saved
                </p>
              </div>

              <div className="p-3 rounded-lg bg-background/50 border border-border">
                <div className="flex items-center gap-2 mb-1">
                  <Leaf className="h-4 w-4 text-green-500" />
                  <span className="text-xs text-muted-foreground">CO₂</span>
                </div>
                <p className="text-xl font-bold text-foreground">{selectedRouteData.co2_emissions.toFixed(1)} kg</p>
                <p className="text-xs text-emerald-600">
                  -{selectedRouteData.baseline_comparison.co2_saved.toFixed(1)} kg saved
                </p>
              </div>
            </div>
          </Card>

          {/* Route Stops */}
          <Card className="p-6">
            <h4 className="text-lg font-semibold text-foreground mb-4">Route Timeline</h4>
            <div className="space-y-4">
              {/* Depot Start */}
              <div className="flex items-start gap-4">
                <div className="flex flex-col items-center">
                  <div className="p-2 rounded-full bg-gradient-to-br from-blue-500 to-blue-600">
                    <MapPin className="h-4 w-4 text-white" />
                  </div>
                  <div className="w-0.5 h-12 bg-gradient-to-b from-blue-500 to-purple-500"></div>
                </div>
                <div className="flex-1 pt-1">
                  <h5 className="font-semibold text-foreground">Downtown Boston Depot</h5>
                  <p className="text-sm text-muted-foreground">Departure: 07:30</p>
                  <Badge variant="secondary" className="mt-1 text-xs">
                    Start
                  </Badge>
                </div>
              </div>

              {/* Stops */}
              {selectedRouteData.stops.map((stop, index) => (
                <div key={index} className="flex items-start gap-4">
                  <div className="flex flex-col items-center">
                    <div className="p-2 rounded-full bg-gradient-to-br from-purple-500 to-pink-500">
                      <MapPin className="h-4 w-4 text-white" />
                    </div>
                    {index < selectedRouteData.stops.length - 1 && (
                      <div className="w-0.5 h-12 bg-gradient-to-b from-purple-500 to-pink-500"></div>
                    )}
                  </div>
                  <div className="flex-1 pt-1">
                    <h5 className="font-semibold text-foreground">{stop.location.name}</h5>
                    <div className="flex items-center gap-4 text-sm text-muted-foreground mt-1">
                      <span>Arrival: {stop.arrival_time}</span>
                      <span>•</span>
                      <span>Departure: {stop.departure_time}</span>
                      <span>•</span>
                      <span>Service: {stop.service_time.toFixed(1)} min</span>
                    </div>
                    <div className="flex items-center gap-2 mt-2">
                      <Badge variant="secondary" className="text-xs">
                        Demand: {stop.location.demand} units
                      </Badge>
                      <Badge variant="secondary" className="text-xs">
                        Load After: {stop.load_after} units
                      </Badge>
                      {stop.location.special_requirements.length > 0 && (
                        <Badge variant="secondary" className="text-xs bg-yellow-500/10 text-yellow-600">
                          {stop.location.special_requirements[0]}
                        </Badge>
                      )}
                    </div>
                  </div>
                </div>
              ))}

              {/* Depot Return */}
              <div className="flex items-start gap-4">
                <div className="flex flex-col items-center">
                  <div className="p-2 rounded-full bg-gradient-to-br from-emerald-500 to-green-600">
                    <MapPin className="h-4 w-4 text-white" />
                  </div>
                </div>
                <div className="flex-1 pt-1">
                  <h5 className="font-semibold text-foreground">Downtown Boston Depot</h5>
                  <p className="text-sm text-muted-foreground">
                    Return: {selectedRouteData.stops[selectedRouteData.stops.length - 1].departure_time}
                  </p>
                  <Badge variant="secondary" className="mt-1 text-xs bg-emerald-500/10 text-emerald-600">
                    Complete
                  </Badge>
                </div>
              </div>
            </div>
          </Card>
        </div>
      )}
    </div>
  )
}
