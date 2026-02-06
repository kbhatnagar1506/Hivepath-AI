"use client"

import { Badge } from "@/components/ui/badge"
import { TrendingUp, TrendingDown, Loader2 } from "lucide-react"
import { useEffect, useState } from "react"
import { fetchAnalytics, type Analytics } from "@/lib/api"

interface MetricsBarProps {
  activeRun: "baseline" | "replan"
  riskShaper: boolean
  warmStart: boolean
  serviceTimeGNN: boolean
}

export function MetricsBar({ activeRun, riskShaper, warmStart, serviceTimeGNN }: MetricsBarProps) {
  const [analytics, setAnalytics] = useState<Analytics | null>(null)
  const [animateMetrics, setAnimateMetrics] = useState(false)

  useEffect(() => {
    async function loadAnalytics() {
      try {
        const data = await fetchAnalytics()
        setAnalytics(data)
      } catch (error) {}
    }
    loadAnalytics()
  }, [])

  useEffect(() => {
    setAnimateMetrics(true)
    const timer = setTimeout(() => setAnimateMetrics(false), 500)
    return () => clearTimeout(timer)
  }, [activeRun, riskShaper, warmStart, serviceTimeGNN])

  const baselineMetrics = {
    distance: 45.3,
    driveTime: 67,
    served: 100,
    onTime: 92.8,
    vehicles: analytics?.total_vehicles || 8,
    co2: 6.8,
    riskyDistance: 8.2,
  }

  const calculateReplanMetrics = () => {
    let distance = 42.7
    let driveTime = 59
    let onTime = 95.2
    const co2 = 6.3
    let riskyDistance = 6.7

    if (riskShaper) {
      distance += 0.8
      driveTime += 2
      riskyDistance -= 1.5
      onTime += 1.5
    }

    if (warmStart) {
      onTime += 0.8
    }

    if (serviceTimeGNN) {
      onTime += 1.5
    }

    return {
      distance: Number(distance.toFixed(1)),
      driveTime: Math.round(driveTime),
      served: 100,
      onTime: Number(Math.min(onTime, 99.5).toFixed(1)),
      vehicles: analytics?.total_vehicles || 8,
      co2: Number(co2.toFixed(1)),
      riskyDistance: Number(Math.max(riskyDistance, 5.0).toFixed(1)),
    }
  }

  const metrics = activeRun === "baseline" ? baselineMetrics : calculateReplanMetrics()
  const showDelta = activeRun === "replan"

  const calculateDelta = (current: number, baseline: number, inverse = false) => {
    const delta = ((current - baseline) / baseline) * 100
    const isPositive = inverse ? delta < 0 : delta > 0
    return { delta: Math.abs(delta), isPositive }
  }

  const MetricCard = ({
    label,
    value,
    unit,
    baselineValue,
    inverse = false,
    highlight = false,
  }: {
    label: string
    value: number
    unit: string
    baselineValue?: number
    inverse?: boolean
    highlight?: boolean
  }) => {
    const delta = baselineValue ? calculateDelta(value, baselineValue, inverse) : null

    return (
      <div className={`${animateMetrics ? "animate-slide-up" : ""}`}>
        <div className="text-xs text-muted-foreground mb-1">{label}</div>
        <div className="flex items-center gap-2">
          <div className={`text-2xl font-bold ${highlight ? "text-primary" : "text-foreground"} transition-smooth`}>
            {value}
            {unit}
          </div>
          {showDelta && delta && (
            <Badge
              variant="outline"
              className={`gap-1 transition-smooth ${
                delta.isPositive
                  ? "bg-success/10 text-success border-success/20"
                  : "bg-destructive/10 text-destructive border-destructive/20"
              }`}
            >
              {delta.isPositive ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
              {delta.delta.toFixed(1)}%
            </Badge>
          )}
        </div>
      </div>
    )
  }

  if (!analytics) {
    return (
      <div className="border-t border-border bg-card">
        <div className="flex items-center justify-center px-6 py-4">
          <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
        </div>
      </div>
    )
  }

  return (
    <div className="border-t border-border bg-card">
      <div className="flex items-center justify-between px-6 py-4">
        <div className="flex items-center gap-8">
          <MetricCard
            label="Distance"
            value={metrics.distance}
            unit=" km"
            baselineValue={showDelta ? baselineMetrics.distance : undefined}
            inverse
          />

          <MetricCard
            label="Drive Time"
            value={metrics.driveTime}
            unit=" min"
            baselineValue={showDelta ? baselineMetrics.driveTime : undefined}
            inverse
          />

          <MetricCard label="Served" value={metrics.served} unit="%" />

          <MetricCard
            label="On-Time Rate"
            value={metrics.onTime}
            unit="%"
            baselineValue={showDelta ? baselineMetrics.onTime : undefined}
            highlight={showDelta}
          />

          <MetricCard label="Vehicles Used" value={metrics.vehicles} unit={`/${analytics.total_vehicles}`} />

          <MetricCard
            label="CO₂ Emissions"
            value={metrics.co2}
            unit=" kg"
            baselineValue={showDelta ? baselineMetrics.co2 : undefined}
            inverse
          />

          {(activeRun === "replan" || showDelta) && (
            <MetricCard
              label="Risky Distance"
              value={metrics.riskyDistance}
              unit=" km"
              baselineValue={showDelta ? baselineMetrics.riskyDistance : undefined}
              inverse
            />
          )}
        </div>

        {showDelta && (
          <div className="flex items-center gap-3 text-xs">
            {riskShaper && (
              <Badge variant="secondary" className="gap-1 animate-slide-up bg-primary/10 border-primary/20">
                <TrendingDown className="h-3 w-3" />
                Risk Shaper: Risky km -18%
              </Badge>
            )}
            {warmStart && (
              <Badge variant="secondary" className="gap-1 animate-slide-up bg-accent/10 border-accent/20">
                <TrendingDown className="h-3 w-3" />
                Warm-Start: Solve -43%
              </Badge>
            )}
            {serviceTimeGNN && (
              <Badge variant="secondary" className="gap-1 animate-slide-up bg-success/10 border-success/20">
                <TrendingDown className="h-3 w-3" />
                Service-Time: Late stops -31%
              </Badge>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
