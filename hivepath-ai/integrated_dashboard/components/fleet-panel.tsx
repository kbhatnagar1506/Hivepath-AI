"use client"

import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Truck, Package, Zap, Clock, DollarSign } from "lucide-react"
import type { Vehicle } from "@/lib/api"

interface FleetPanelProps {
  vehicles: Vehicle[]
}

export function FleetPanel({ vehicles }: FleetPanelProps) {
  const totalCapacity = vehicles.reduce((sum, v) => sum + v.capacity, 0)
  const trucks = vehicles.filter((v) => v.type === "truck")
  const vans = vehicles.filter((v) => v.type === "van")
  const avgEfficiency = vehicles.reduce((sum, v) => sum + v.efficiency, 0) / vehicles.length
  const totalCostPerKm = vehicles.reduce((sum, v) => sum + v.cost_per_km, 0)
  const totalCostPerHour = vehicles.reduce((sum, v) => sum + v.cost_per_hour, 0)

  return (
    <div className="space-y-6 overflow-y-auto max-h-full">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="p-4 bg-gradient-to-br from-blue-500/10 to-blue-600/5 border-blue-500/20">
          <div className="flex items-center gap-3">
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-blue-500/20">
              <Truck className="h-6 w-6 text-blue-500" />
            </div>
            <div>
              <div className="text-2xl font-bold text-foreground">{vehicles.length}</div>
              <div className="text-xs text-muted-foreground">Total Vehicles</div>
            </div>
          </div>
        </Card>

        <Card className="p-4 bg-gradient-to-br from-emerald-500/10 to-emerald-600/5 border-emerald-500/20">
          <div className="flex items-center gap-3">
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-emerald-500/20">
              <Package className="h-6 w-6 text-emerald-500" />
            </div>
            <div>
              <div className="text-2xl font-bold text-foreground">{totalCapacity}</div>
              <div className="text-xs text-muted-foreground">Total Capacity</div>
            </div>
          </div>
        </Card>

        <Card className="p-4 bg-gradient-to-br from-purple-500/10 to-purple-600/5 border-purple-500/20">
          <div className="flex items-center gap-3">
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-purple-500/20">
              <Zap className="h-6 w-6 text-purple-500" />
            </div>
            <div>
              <div className="text-2xl font-bold text-foreground">
                {trucks.length}/{vans.length}
              </div>
              <div className="text-xs text-muted-foreground">Trucks / Vans</div>
            </div>
          </div>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {vehicles.map((vehicle, idx) => (
          <Card key={vehicle.id} className="p-4 hover:shadow-lg transition-shadow">
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-3">
                <div
                  className="flex h-10 w-10 items-center justify-center rounded-full"
                  style={{
                    backgroundColor: ["#3b82f6", "#06b6d4", "#8b5cf6", "#f59e0b"][idx % 4] + "20",
                  }}
                >
                  <Truck className="h-5 w-5" style={{ color: ["#3b82f6", "#06b6d4", "#8b5cf6", "#f59e0b"][idx % 4] }} />
                </div>
                <div>
                  <div className="font-semibold text-foreground">{vehicle.id}</div>
                  <div className="text-xs text-muted-foreground capitalize">{vehicle.type}</div>
                </div>
              </div>
              <Badge variant={vehicle.fuel_type === "electric" ? "default" : "secondary"} className="text-xs">
                {vehicle.fuel_type}
              </Badge>
            </div>

            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <div className="text-xs text-muted-foreground">Capacity</div>
                  <div className="flex items-baseline gap-1">
                    <span className="text-lg font-bold text-foreground">{vehicle.capacity}</span>
                    <span className="text-xs text-muted-foreground">units</span>
                  </div>
                </div>
                <div className="space-y-1">
                  <div className="text-xs text-muted-foreground">Dimensions</div>
                  <div className="text-sm font-medium text-foreground">
                    {vehicle.length}×{vehicle.width}×{vehicle.height}m
                  </div>
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">Fuel Efficiency</span>
                  <span className="font-medium text-foreground">{vehicle.fuel_efficiency} km/L</span>
                </div>
                <Progress value={vehicle.fuel_efficiency * 10} className="h-1.5" />
              </div>

              <div className="grid grid-cols-2 gap-2 pt-2 border-t border-border">
                <div className="flex items-center gap-2">
                  <DollarSign className="h-3.5 w-3.5 text-muted-foreground" />
                  <span className="text-xs text-muted-foreground">${vehicle.cost_per_km}/km</span>
                </div>
                <div className="flex items-center gap-2">
                  <Clock className="h-3.5 w-3.5 text-muted-foreground" />
                  <span className="text-xs text-muted-foreground">${vehicle.cost_per_hour}/hr</span>
                </div>
              </div>

              {vehicle.capabilities.length > 0 && (
                <div className="flex flex-wrap gap-1 pt-2">
                  {vehicle.capabilities.map((cap) => (
                    <Badge key={cap} variant="outline" className="text-[10px] px-1.5 py-0">
                      {cap}
                    </Badge>
                  ))}
                </div>
              )}
            </div>
          </Card>
        ))}
      </div>
    </div>
  )
}
