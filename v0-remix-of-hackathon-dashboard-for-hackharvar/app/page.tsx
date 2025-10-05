"use client"

import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { DashboardHeader } from "@/components/dashboard-header"
import { MetricsBar } from "@/components/metrics-bar"
import { MapView } from "@/components/map-view"
import { KnowledgeGraphView } from "@/components/knowledge-graph-view"
import { DriverModeView } from "@/components/driver-mode-view"
import { FleetPanel } from "@/components/fleet-panel"
import { AIInsightsPanel } from "@/components/ai-insights-panel"
import { SavingsImpactPanel } from "@/components/savings-impact-panel"
import { useEffect } from "react"
import { fetchVehicles, type Vehicle } from "@/lib/api"
import { RoutesPanel } from "@/components/routes-panel"

export default function DashboardPage() {
  const [activeRun, setActiveRun] = useState<"baseline" | "replan">("replan")
  const [riskShaper, setRiskShaper] = useState(true)
  const [warmStart, setWarmStart] = useState(true)
  const [serviceTimeGNN, setServiceTimeGNN] = useState(true)
  const [vehicles, setVehicles] = useState<Vehicle[]>([])

  useEffect(() => {
    async function loadVehicles() {
      try {
        const data = await fetchVehicles()
        setVehicles(data)
      } catch (error) {
        console.error("Failed to load vehicles:", error)
      }
    }
    loadVehicles()
  }, [])

  return (
    <div className="flex h-screen flex-col bg-background">
      <DashboardHeader
        activeRun={activeRun}
        onRunChange={setActiveRun}
        riskShaper={riskShaper}
        onRiskShaperChange={setRiskShaper}
        warmStart={warmStart}
        onWarmStartChange={setWarmStart}
        serviceTimeGNN={serviceTimeGNN}
        onServiceTimeGNNChange={setServiceTimeGNN}
      />

      <div className="flex flex-1 flex-col overflow-hidden">
        <Tabs defaultValue="map" className="flex flex-1 flex-col">
          <div className="border-b border-border bg-card px-6">
            <TabsList className="h-12 bg-transparent">
              <TabsTrigger value="map" className="data-[state=active]:bg-secondary">
                Map
              </TabsTrigger>
              <TabsTrigger value="graph" className="data-[state=active]:bg-secondary">
                3D Knowledge Graph
              </TabsTrigger>
              <TabsTrigger value="driver" className="data-[state=active]:bg-secondary">
                Driver Mode
              </TabsTrigger>
              <TabsTrigger value="routes" className="data-[state=active]:bg-secondary">
                Routes
              </TabsTrigger>
              <TabsTrigger value="fleet" className="data-[state=active]:bg-secondary">
                Fleet
              </TabsTrigger>
              <TabsTrigger value="ai" className="data-[state=active]:bg-secondary">
                AI Insights
              </TabsTrigger>
              <TabsTrigger value="savings" className="data-[state=active]:bg-secondary">
                Savings & Impact
              </TabsTrigger>
            </TabsList>
          </div>

          <div className="flex-1 overflow-hidden">
            <TabsContent value="map" className="h-full m-0">
              <MapView
                activeRun={activeRun}
                riskShaper={riskShaper}
                warmStart={warmStart}
                serviceTimeGNN={serviceTimeGNN}
              />
            </TabsContent>

            <TabsContent value="graph" className="h-full m-0">
              <KnowledgeGraphView />
            </TabsContent>

            <TabsContent value="driver" className="h-full m-0">
              <DriverModeView />
            </TabsContent>

            <TabsContent value="routes" className="h-full m-0 overflow-y-auto">
              <div className="p-6">
                <div className="mb-6">
                  <h2 className="text-2xl font-bold text-foreground mb-2">Optimized Routes</h2>
                  <p className="text-muted-foreground">
                    AI-optimized delivery routes with real-time risk assessment and efficiency metrics
                  </p>
                </div>
                <RoutesPanel />
              </div>
            </TabsContent>

            <TabsContent value="fleet" className="h-full m-0 overflow-y-auto">
              <div className="p-6">
                <div className="mb-6">
                  <h2 className="text-2xl font-bold text-foreground mb-2">Fleet Management</h2>
                  <p className="text-muted-foreground">
                    Comprehensive overview of all vehicles, their specifications, and capabilities
                  </p>
                </div>
                <FleetPanel vehicles={vehicles} />
              </div>
            </TabsContent>

            <TabsContent value="ai" className="h-full m-0 overflow-y-auto">
              <div className="p-6">
                <div className="mb-6">
                  <h2 className="text-2xl font-bold text-foreground mb-2">AI Insights</h2>
                  <p className="text-muted-foreground">
                    Real-time predictions and confidence scores from the GNN model
                  </p>
                </div>
                <AIInsightsPanel />
              </div>
            </TabsContent>

            <TabsContent value="savings" className="h-full m-0 overflow-y-auto">
              <div className="p-6">
                <div className="mb-6">
                  <h2 className="text-2xl font-bold text-foreground mb-2">Savings & Impact</h2>
                  <p className="text-muted-foreground">
                    Cost savings, carbon reduction, and environmental impact metrics
                  </p>
                </div>
                <SavingsImpactPanel />
              </div>
            </TabsContent>
          </div>
        </Tabs>

        <MetricsBar
          activeRun={activeRun}
          riskShaper={riskShaper}
          warmStart={warmStart}
          serviceTimeGNN={serviceTimeGNN}
        />
      </div>
    </div>
  )
}
