"use client"

import { useState, useEffect } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { DashboardHeader } from "@/components/dashboard-header"
import { MetricsBar } from "@/components/metrics-bar"
import { GoogleMapsEmbed } from "@/components/google-maps-embed"
import { KnowledgeGraphView } from "@/components/knowledge-graph-view"
import { DriverModeView } from "@/components/driver-mode-view"
import { FleetPanel } from "@/components/fleet-panel"
import { AIInsightsPanel } from "@/components/ai-insights-panel"
import { SavingsImpactPanel } from "@/components/savings-impact-panel"
import { fetchVehicles, fetchLocations, fetchPredictions, fetchAnalytics, fetchBulkData, fetchSystemHealth, type Vehicle, type Location, type Prediction, type Analytics } from "@/lib/api"
import { RoutesPanel } from "@/components/routes-panel"
import { DataDisplayPanel } from "@/components/data-display-panel"

export default function DashboardPage() {
  const [activeRun, setActiveRun] = useState<"baseline" | "replan">("replan")
  const [riskShaper, setRiskShaper] = useState(true)
  const [warmStart, setWarmStart] = useState(true)
  const [serviceTimeGNN, setServiceTimeGNN] = useState(true)
  const [vehicles, setVehicles] = useState<Vehicle[]>([])
  const [locations, setLocations] = useState<Location[]>([])
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [analytics, setAnalytics] = useState<Analytics | null>(null)
  const [systemHealth, setSystemHealth] = useState<any>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    async function loadAllData() {
      try {
        setIsLoading(true)
        console.log("[SwarmAura] Loading live data from API...")
        
        // Load all data in parallel
        const [vehiclesData, locationsData, predictionsData, analyticsData, healthData] = await Promise.all([
          fetchVehicles(),
          fetchLocations(),
          fetchPredictions(),
          fetchAnalytics(),
          fetchSystemHealth()
        ])
        
        setVehicles(vehiclesData)
        setLocations(locationsData)
        setPredictions(predictionsData)
        setAnalytics(analyticsData)
        setSystemHealth(healthData)
        
        console.log("[SwarmAura] Data loaded successfully:", {
          vehicles: vehiclesData.length,
          locations: locationsData.length,
          predictions: predictionsData.length,
          health: healthData.status
        })
      } catch (error) {
        console.error("[SwarmAura] Failed to load data:", error)
      } finally {
        setIsLoading(false)
      }
    }
    loadAllData()
  }, [])

  return (
    <div className="flex h-screen flex-col bg-background overflow-hidden">
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

      {isLoading && (
        <div className="flex flex-1 items-center justify-center flex-col gap-4 bg-gradient-to-br from-primary/5 via-background to-primary/5">
          <img 
            src="/logo.png" 
            alt="HivePath AI" 
            className="h-32 w-32 object-contain drop-shadow-lg animate-pulse"
              onError={(e) => {
                e.currentTarget.style.display = 'none';
                const nextElement = e.currentTarget.nextElementSibling as HTMLElement;
                if (nextElement) nextElement.style.display = 'block';
              }}
          />
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent hidden" />
          <div className="text-center">
            <h2 className="text-xl font-semibold text-foreground mb-2">HivePath AI</h2>
            <p className="text-sm text-muted-foreground">Reimagining infrastructure with AI-powered systems...</p>
          </div>
        </div>
      )}

      {!isLoading && (
      <div className="flex flex-1 flex-col min-h-0">
        <Tabs defaultValue="map" className="flex flex-1 flex-col min-h-0">
          <div className="border-b border-border bg-card px-6 flex-shrink-0">
            <TabsList className="h-12 bg-transparent">
              <TabsTrigger value="map" className="data-[state=active]:bg-secondary">
                🏗️ Smart Infrastructure
              </TabsTrigger>
              <TabsTrigger value="graph" className="data-[state=active]:bg-secondary">
                🧠 AI Knowledge Graph
              </TabsTrigger>
              <TabsTrigger value="driver" className="data-[state=active]:bg-secondary">
                🚛 Smart Logistics Hub
              </TabsTrigger>
              <TabsTrigger value="routes" className="data-[state=active]:bg-secondary">
                🛣️ Route Intelligence
              </TabsTrigger>
              <TabsTrigger value="fleet" className="data-[state=active]:bg-secondary">
                🚚 Fleet Command Center
              </TabsTrigger>
              <TabsTrigger value="ai" className="data-[state=active]:bg-secondary">
                🤖 AI Insights Engine
              </TabsTrigger>
              <TabsTrigger value="savings" className="data-[state=active]:bg-secondary">
                💰 Impact & ROI Analytics
              </TabsTrigger>
              <TabsTrigger value="data" className="data-[state=active]:bg-secondary">
                📊 System Data Hub
              </TabsTrigger>
            </TabsList>
          </div>

          <div className="flex-1 min-h-0 overflow-hidden">
            <TabsContent value="map" className="h-full m-0 overflow-hidden">
              <GoogleMapsEmbed
                activeRun={activeRun}
                riskShaper={riskShaper}
                warmStart={warmStart}
                serviceTimeGNN={serviceTimeGNN}
                locations={locations}
                isLoading={isLoading}
              />
            </TabsContent>

            <TabsContent value="graph" className="h-full m-0 overflow-y-auto">
              <KnowledgeGraphView />
            </TabsContent>

            <TabsContent value="driver" className="h-full m-0 overflow-y-auto">
              <DriverModeView />
            </TabsContent>

            <TabsContent value="routes" className="h-full m-0 overflow-y-auto">
              <div className="p-6 space-y-6">
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
              <div className="p-6 space-y-6">
                <div className="mb-6">
                  <h2 className="text-2xl font-bold text-foreground mb-2">🚚 Fleet Command Center</h2>
                  <p className="text-muted-foreground">
                    Advanced vehicle management with AI-powered logistics and real-time infrastructure intelligence
                  </p>
                </div>
                <FleetPanel vehicles={vehicles} />
              </div>
            </TabsContent>

            <TabsContent value="ai" className="h-full m-0 overflow-y-auto">
              <div className="p-6 space-y-6">
                <div className="mb-6">
                  <h2 className="text-2xl font-bold text-foreground mb-2">🤖 AI Insights Engine</h2>
                  <p className="text-muted-foreground">
                    Advanced AI predictions and machine learning insights powering tomorrow's smarter infrastructure systems
                  </p>
                </div>
                <AIInsightsPanel predictions={predictions} isLoading={isLoading} />
              </div>
            </TabsContent>

            <TabsContent value="savings" className="h-full m-0 overflow-y-auto">
              <div className="p-6 space-y-6">
                <div className="mb-6">
                  <h2 className="text-2xl font-bold text-foreground mb-2">Savings & Impact</h2>
                  <p className="text-muted-foreground">
                    Cost savings, carbon reduction, and environmental impact metrics
                  </p>
                </div>
                <SavingsImpactPanel />
              </div>
            </TabsContent>

            <TabsContent value="data" className="h-full m-0 overflow-y-auto">
              <div className="p-6 space-y-6">
                <div className="mb-6">
                  <h2 className="text-2xl font-bold text-foreground mb-2">📊 System Data Hub</h2>
                  <p className="text-muted-foreground">
                    Comprehensive data analytics and insights from our next-generation infrastructure platform
                  </p>
                </div>
                <DataDisplayPanel 
                  locations={locations}
                  vehicles={vehicles}
                  predictions={predictions}
                  analytics={analytics}
                  isLoading={isLoading}
                />
              </div>
            </TabsContent>
          </div>
        </Tabs>

      </div>
      )}
    </div>
  )
}
