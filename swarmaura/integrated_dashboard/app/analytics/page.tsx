"use client"

import { useState, useEffect } from "react"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { 
  TrendingUp, 
  TrendingDown, 
  BarChart3, 
  Activity, 
  Clock, 
  MapPin, 
  Truck, 
  Zap,
  ArrowLeft,
  RefreshCw,
  Download,
  Share2
} from "lucide-react"
import { fetchAnalytics, fetchSystemHealth, type Analytics } from "@/lib/api"
import Link from "next/link"

interface MetricCardProps {
  label: string
  value: number | string
  unit: string
  change?: number
  trend?: "up" | "down" | "neutral"
  description?: string
  icon?: React.ReactNode
}

function MetricCard({ label, value, unit, change, trend, description, icon }: MetricCardProps) {
  const getTrendColor = () => {
    if (trend === "up") return "text-green-600"
    if (trend === "down") return "text-red-600"
    return "text-slate-600"
  }

  const getTrendIcon = () => {
    if (trend === "up") return <TrendingUp className="h-4 w-4" />
    if (trend === "down") return <TrendingDown className="h-4 w-4" />
    return null
  }

  return (
    <Card className="p-6 hover:shadow-lg transition-all duration-200 border-l-4 border-l-blue-500">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-2">
            {icon}
            <h3 className="text-sm font-medium text-slate-600 dark:text-slate-400">{label}</h3>
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-bold text-slate-900 dark:text-slate-100">
              {value}
            </span>
            <span className="text-lg text-slate-500 dark:text-slate-500">{unit}</span>
          </div>
          {change !== undefined && (
            <div className={`flex items-center gap-1 mt-2 ${getTrendColor()}`}>
              {getTrendIcon()}
              <span className="text-sm font-medium">
                {change > 0 ? "+" : ""}{change}%
              </span>
            </div>
          )}
          {description && (
            <p className="text-xs text-slate-500 dark:text-slate-500 mt-2">{description}</p>
          )}
        </div>
      </div>
    </Card>
  )
}

export default function AnalyticsPage() {
  const [analytics, setAnalytics] = useState<Analytics | null>(null)
  const [systemHealth, setSystemHealth] = useState<any>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date())

  useEffect(() => {
    async function loadData() {
      try {
        setIsLoading(true)
        const [analyticsData, healthData] = await Promise.all([
          fetchAnalytics(),
          fetchSystemHealth()
        ])
        setAnalytics(analyticsData)
        setSystemHealth(healthData)
        setLastUpdated(new Date())
      } catch (error) {
        console.error("Failed to load analytics:", error)
      } finally {
        setIsLoading(false)
      }
    }
    loadData()
  }, [])

  const refreshData = async () => {
    setIsLoading(true)
    try {
      const [analyticsData, healthData] = await Promise.all([
        fetchAnalytics(),
        fetchSystemHealth()
      ])
      setAnalytics(analyticsData)
      setSystemHealth(healthData)
      setLastUpdated(new Date())
    } catch (error) {
      console.error("Failed to refresh data:", error)
    } finally {
      setIsLoading(false)
    }
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
        <div className="flex items-center justify-center min-h-screen">
          <div className="text-center">
            <div className="h-12 w-12 animate-spin rounded-full border-4 border-blue-500 border-t-transparent mx-auto mb-4"></div>
            <h2 className="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-2">Loading Analytics</h2>
            <p className="text-slate-600 dark:text-slate-400">Fetching real-time performance data...</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      {/* Header */}
      <div className="border-b border-slate-200 dark:border-slate-700 bg-white/80 dark:bg-slate-900/80 backdrop-blur-sm">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link href="/">
                <Button variant="ghost" size="sm" className="gap-2">
                  <ArrowLeft className="h-4 w-4" />
                  Back to Dashboard
                </Button>
              </Link>
              <div>
                <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">Performance Analytics</h1>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                  Detailed insights and metrics for HivePath AI infrastructure
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <Button variant="outline" size="sm" onClick={refreshData} disabled={isLoading} className="gap-2">
                <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
                Refresh
              </Button>
              <Button variant="outline" size="sm" className="gap-2">
                <Download className="h-4 w-4" />
                Export
              </Button>
              <Button variant="outline" size="sm" className="gap-2">
                <Share2 className="h-4 w-4" />
                Share
              </Button>
            </div>
          </div>
        </div>
      </div>

      <div className="p-6">
        <Tabs defaultValue="overview" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="performance">Performance</TabsTrigger>
            <TabsTrigger value="optimization">Optimization</TabsTrigger>
            <TabsTrigger value="system">System Health</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-6">
            {/* Key Metrics Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
              <MetricCard
                label="Total Distance"
                value={43.5}
                unit="km"
                change={4.0}
                trend="up"
                description="Total route distance covered"
                icon={<MapPin className="h-4 w-4 text-blue-500" />}
              />
              <MetricCard
                label="Drive Time"
                value={61}
                unit="min"
                change={9.0}
                trend="up"
                description="Average drive time per route"
                icon={<Clock className="h-4 w-4 text-orange-500" />}
              />
              <MetricCard
                label="Service Rate"
                value={100}
                unit="%"
                description="Percentage of locations successfully served"
                icon={<Activity className="h-4 w-4 text-green-500" />}
              />
              <MetricCard
                label="On-Time Rate"
                value={99}
                unit="%"
                change={6.7}
                trend="up"
                description="Percentage of deliveries completed on time"
                icon={<Zap className="h-4 w-4 text-emerald-500" />}
              />
              <MetricCard
                label="Vehicles Used"
                value="4/4"
                unit=""
                description="Active vehicles in the fleet"
                icon={<Truck className="h-4 w-4 text-purple-500" />}
              />
              <MetricCard
                label="CO₂ Emissions"
                value={6.3}
                unit="kg"
                change={7.4}
                trend="up"
                description="Total carbon emissions produced"
                icon={<BarChart3 className="h-4 w-4 text-red-500" />}
              />
              <MetricCard
                label="Risky Distance"
                value={5.2}
                unit="km"
                change={36.6}
                trend="up"
                description="Distance through high-risk areas"
                icon={<TrendingUp className="h-4 w-4 text-yellow-500" />}
              />
            </div>

            {/* Optimization Features */}
            <Card className="p-6">
              <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">Active Optimizations</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="flex items-center gap-3 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
                  <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse"></div>
                  <div>
                    <div className="font-medium text-green-900 dark:text-green-100">Risk Shaper</div>
                    <div className="text-sm text-green-700 dark:text-green-300">Risky km -18%</div>
                  </div>
                </div>
                <div className="flex items-center gap-3 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
                  <div className="h-2 w-2 rounded-full bg-blue-500"></div>
                  <div>
                    <div className="font-medium text-blue-900 dark:text-blue-100">Warm-Start</div>
                    <div className="text-sm text-blue-700 dark:text-blue-300">Solve time -43%</div>
                  </div>
                </div>
                <div className="flex items-center gap-3 p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-200 dark:border-purple-800">
                  <div className="h-2 w-2 rounded-full bg-purple-500"></div>
                  <div>
                    <div className="font-medium text-purple-900 dark:text-purple-100">Service-Time GNN</div>
                    <div className="text-sm text-purple-700 dark:text-purple-300">Late stops -31%</div>
                  </div>
                </div>
              </div>
            </Card>
          </TabsContent>

          <TabsContent value="performance" className="space-y-6">
            <Card className="p-6">
              <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">Performance Trends</h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 bg-slate-50 dark:bg-slate-800 rounded-lg">
                  <div>
                    <div className="font-medium">Distance Optimization</div>
                    <div className="text-sm text-slate-600 dark:text-slate-400">Route efficiency improvements</div>
                  </div>
                  <Badge variant="secondary" className="bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
                    +4.0% improvement
                  </Badge>
                </div>
                <div className="flex items-center justify-between p-4 bg-slate-50 dark:bg-slate-800 rounded-lg">
                  <div>
                    <div className="font-medium">Time Efficiency</div>
                    <div className="text-sm text-slate-600 dark:text-slate-400">Drive time optimization</div>
                  </div>
                  <Badge variant="secondary" className="bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200">
                    +9.0% increase
                  </Badge>
                </div>
                <div className="flex items-center justify-between p-4 bg-slate-50 dark:bg-slate-800 rounded-lg">
                  <div>
                    <div className="font-medium">On-Time Delivery</div>
                    <div className="text-sm text-slate-600 dark:text-slate-400">Delivery punctuality</div>
                  </div>
                  <Badge variant="secondary" className="bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200">
                    +6.7% improvement
                  </Badge>
                </div>
              </div>
            </Card>
          </TabsContent>

          <TabsContent value="optimization" className="space-y-6">
            <Card className="p-6">
              <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">AI Optimization Features</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div className="p-4 border border-green-200 dark:border-green-800 rounded-lg bg-green-50 dark:bg-green-900/20">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse"></div>
                      <h4 className="font-semibold text-green-900 dark:text-green-100">Risk Shaper</h4>
                    </div>
                    <p className="text-sm text-green-700 dark:text-green-300 mb-2">
                      Advanced risk assessment and route optimization to minimize dangerous areas.
                    </p>
                    <div className="text-lg font-bold text-green-900 dark:text-green-100">-18% risky distance</div>
                  </div>
                  
                  <div className="p-4 border border-blue-200 dark:border-blue-800 rounded-lg bg-blue-50 dark:bg-blue-900/20">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="h-2 w-2 rounded-full bg-blue-500"></div>
                      <h4 className="font-semibold text-blue-900 dark:text-blue-100">Warm-Start Algorithm</h4>
                    </div>
                    <p className="text-sm text-blue-700 dark:text-blue-300 mb-2">
                      Intelligent initialization using previous solutions for faster convergence.
                    </p>
                    <div className="text-lg font-bold text-blue-900 dark:text-blue-100">-43% solve time</div>
                  </div>
                </div>
                
                <div className="space-y-4">
                  <div className="p-4 border border-purple-200 dark:border-purple-800 rounded-lg bg-purple-50 dark:bg-purple-900/20">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="h-2 w-2 rounded-full bg-purple-500"></div>
                      <h4 className="font-semibold text-purple-900 dark:text-purple-100">Service-Time GNN</h4>
                    </div>
                    <p className="text-sm text-purple-700 dark:text-purple-300 mb-2">
                      Graph Neural Network for predicting and optimizing service times.
                    </p>
                    <div className="text-lg font-bold text-purple-900 dark:text-purple-100">-31% late stops</div>
                  </div>
                </div>
              </div>
            </Card>
          </TabsContent>

          <TabsContent value="system" className="space-y-6">
            <Card className="p-6">
              <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">System Health</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="flex items-center gap-3 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
                  <div className="h-3 w-3 rounded-full bg-green-500 animate-pulse"></div>
                  <div>
                    <div className="font-medium text-green-900 dark:text-green-100">System Online</div>
                    <div className="text-sm text-green-700 dark:text-green-300">All services operational</div>
                  </div>
                </div>
                <div className="flex items-center gap-3 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
                  <div className="h-3 w-3 rounded-full bg-blue-500"></div>
                  <div>
                    <div className="font-medium text-blue-900 dark:text-blue-100">AI Active</div>
                    <div className="text-sm text-blue-700 dark:text-blue-300">Neural networks running</div>
                  </div>
                </div>
                <div className="flex items-center gap-3 p-4 bg-emerald-50 dark:bg-emerald-900/20 rounded-lg border border-emerald-200 dark:border-emerald-800">
                  <div className="h-3 w-3 rounded-full bg-emerald-500"></div>
                  <div>
                    <div className="font-medium text-emerald-900 dark:text-emerald-100">Optimization Running</div>
                    <div className="text-sm text-emerald-700 dark:text-emerald-300">Real-time processing</div>
                  </div>
                </div>
              </div>
            </Card>

            <Card className="p-6">
              <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">Company Information</h3>
              <div className="flex items-center gap-4 p-4 bg-slate-50 dark:bg-slate-800 rounded-lg">
                <div className="h-12 w-12 rounded-lg bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center">
                  <span className="text-white font-bold text-lg">H</span>
                </div>
                <div>
                  <div className="font-semibold text-slate-900 dark:text-slate-100 text-lg">HivePath AI</div>
                  <div className="text-slate-600 dark:text-slate-400">Next-Gen Infrastructure Platform</div>
                  <div className="text-sm text-slate-500 dark:text-slate-500 mt-1">
                    © 2024 HivePath AI. All rights reserved. • v2.1.0 • Last updated: {lastUpdated.toLocaleTimeString()}
                  </div>
                </div>
              </div>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
