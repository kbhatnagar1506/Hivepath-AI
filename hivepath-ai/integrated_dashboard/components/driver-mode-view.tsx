"use client"

import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Navigation, Clock, MapPin, AlertTriangle, Fuel, Battery, TrendingUp } from "lucide-react"

export function DriverModeView() {
  return (
    <div className="flex h-full bg-secondary/20 overflow-y-auto">
      <div className="flex flex-1 flex-col lg:flex-row gap-4 p-6">
        {/* Left: ETA and Status */}
        <div className="w-full lg:w-80 space-y-4">
          <Card className="p-6">
            <div className="text-sm text-muted-foreground mb-2">Next Stop ETA</div>
            <div className="text-5xl font-bold text-foreground mb-4">18 min</div>
            <div className="flex items-center gap-2 mb-4">
              <MapPin className="h-4 w-4 text-primary" />
              <span className="text-sm text-foreground font-medium">Stop K - Boylston St</span>
            </div>
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm text-muted-foreground">Expected service: 4.0 min</span>
            </div>
          </Card>

          <Card className="p-4">
            <div className="text-sm font-medium text-foreground mb-3">Vehicle Status</div>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Clock className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm text-muted-foreground">HOS Remaining</span>
                </div>
                <Badge variant="secondary">6.2 hrs</Badge>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Fuel className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm text-muted-foreground">Fuel Range</span>
                </div>
                <Badge variant="secondary">285 mi</Badge>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Battery className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm text-muted-foreground">Capacity Used</span>
                </div>
                <Badge variant="secondary">78%</Badge>
              </div>
            </div>
          </Card>

          <Card className="p-4">
            <div className="text-sm font-medium text-foreground mb-3">Quick Actions</div>
            <div className="space-y-2">
              <Button variant="outline" className="w-full justify-start bg-transparent" size="sm">
                <AlertTriangle className="mr-2 h-4 w-4" />
                Report Blocked Curb
              </Button>
              <Button variant="outline" className="w-full justify-start bg-transparent" size="sm">
                <Navigation className="mr-2 h-4 w-4" />
                Replan Around Block
              </Button>
              <Button variant="outline" className="w-full justify-start bg-transparent" size="sm">
                <TrendingUp className="mr-2 h-4 w-4" />
                Suggest Better Zone
              </Button>
            </div>
          </Card>
        </div>

        {/* Center: Map with Last 300m */}
        <div className="flex-1 flex flex-col gap-4">
          <Card className="flex-1 p-4">
            <div className="text-sm font-medium text-foreground mb-3">Last 300m Approach - Live Map</div>
            <div className="relative h-full rounded-lg border border-border bg-secondary/30 overflow-hidden">
              <iframe
                src="https://www.openstreetmap.org/export/embed.html?bbox=-71.0820%2C42.3495%2C-71.0750%2C42.3525&layer=mapnik&marker=42.3510%2C-71.0785"
                className="absolute inset-0 w-full h-full"
                style={{ border: 0 }}
                title="Driver Navigation Map"
              />

              {/* Overlay with route information */}
              <div className="absolute top-4 left-4 right-4 pointer-events-none">
                <div className="flex items-center gap-2 bg-card/95 backdrop-blur-sm border border-border rounded-lg px-4 py-2 shadow-lg">
                  <Navigation className="h-4 w-4 text-primary animate-pulse" />
                  <span className="text-sm font-medium text-foreground">Following optimized route</span>
                  <Badge variant="secondary" className="ml-auto">
                    280m to destination
                  </Badge>
                </div>
              </div>

              <div className="absolute bottom-4 left-4 pointer-events-none">
                <Badge variant="outline" className="bg-card/95 backdrop-blur-sm">
                  Boylston St, Boston MA
                </Badge>
              </div>

              {/* Hazard overlay */}
              <div className="absolute bottom-4 right-4 pointer-events-none">
                <div className="flex items-center gap-2 bg-warning/90 backdrop-blur-sm border border-warning rounded-lg px-3 py-1.5 shadow-lg">
                  <AlertTriangle className="h-3 w-3 text-warning-foreground" />
                  <span className="text-xs font-medium text-warning-foreground">Bike lane ahead</span>
                </div>
              </div>
            </div>
          </Card>
        </div>

        {/* Right: Instructions */}
        <div className="w-full lg:w-80 space-y-4">
          <Card className="p-4">
            <div className="text-sm font-medium text-foreground mb-3">Turn-by-Turn</div>
            <div className="space-y-3">
              <div className="flex gap-3 p-3 rounded-md bg-primary/10 border border-primary/20">
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground font-bold text-sm">
                  1
                </div>
                <div>
                  <div className="text-sm font-medium text-foreground">Continue on Boylston St</div>
                  <div className="text-xs text-muted-foreground mt-1">Wide turn OK for truck</div>
                </div>
              </div>

              <div className="flex gap-3 p-3 rounded-md bg-secondary/50">
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-secondary text-secondary-foreground font-bold text-sm">
                  2
                </div>
                <div>
                  <div className="text-sm font-medium text-foreground">Right onto Boylston</div>
                  <div className="text-xs text-muted-foreground mt-1">Watch for bike lane on right</div>
                </div>
              </div>

              <div className="flex gap-3 p-3 rounded-md bg-secondary/50">
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-secondary text-secondary-foreground font-bold text-sm">
                  3
                </div>
                <div>
                  <div className="text-sm font-medium text-foreground">Loading zone ahead</div>
                  <div className="text-xs text-muted-foreground mt-1">60m on left side, legal 7am-7pm</div>
                </div>
              </div>
            </div>
          </Card>

          <Card className="p-4">
            <div className="text-sm font-medium text-foreground mb-3">Curb Details</div>
            <div className="space-y-3">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Access Score:</span>
                <Badge variant="secondary">0.78</Badge>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Curb Side:</span>
                <span className="text-foreground font-medium">Left</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Zone Type:</span>
                <span className="text-foreground font-medium">Loading</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Time Limit:</span>
                <span className="text-foreground font-medium">30 min</span>
              </div>
            </div>
          </Card>

          <Card className="p-4 bg-warning/10 border-warning/20">
            <div className="flex items-start gap-2">
              <AlertTriangle className="h-5 w-5 text-warning shrink-0 mt-0.5" />
              <div>
                <div className="text-sm font-medium text-foreground mb-1">Hazards Detected</div>
                <ul className="text-xs text-muted-foreground space-y-1">
                  <li>• Active bike lane on approach</li>
                  <li>• Narrow turn radius</li>
                  <li>• Pedestrian crossing nearby</li>
                </ul>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  )
}
