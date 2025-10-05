"use client"

import { useEffect, useState } from "react"
import { Card } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { TrendingDown, DollarSign, Leaf, Clock, Zap, Award } from "lucide-react"

interface SavingsData {
  costSavings: {
    fuel: number
    labor: number
    maintenance: number
    total: number
  }
  carbonReduction: {
    co2Kg: number
    treesEquivalent: number
    percentReduction: number
  }
  efficiency: {
    timeReduction: number
    distanceReduction: number
    onTimeImprovement: number
  }
  environmental: {
    riskyDistanceReduction: number
    safetyScore: number
  }
}

export function SavingsImpactPanel() {
  const [savings, setSavings] = useState<SavingsData>({
    costSavings: {
      fuel: 1250,
      labor: 850,
      maintenance: 320,
      total: 2420,
    },
    carbonReduction: {
      co2Kg: 485,
      treesEquivalent: 22,
      percentReduction: 18.5,
    },
    efficiency: {
      timeReduction: 32,
      distanceReduction: 15.8,
      onTimeImprovement: 5.2,
    },
    environmental: {
      riskyDistanceReduction: 22.5,
      safetyScore: 94,
    },
  })

  const [animatedValues, setAnimatedValues] = useState({
    totalSavings: 0,
    co2Reduction: 0,
    timeReduction: 0,
  })

  useEffect(() => {
    const duration = 2000
    const steps = 60
    const interval = duration / steps

    let step = 0
    const timer = setInterval(() => {
      step++
      const progress = step / steps

      setAnimatedValues({
        totalSavings: Math.floor(savings.costSavings.total * progress),
        co2Reduction: Math.floor(savings.carbonReduction.co2Kg * progress),
        timeReduction: Math.floor(savings.efficiency.timeReduction * progress),
      })

      if (step >= steps) clearInterval(timer)
    }, interval)

    return () => clearInterval(timer)
  }, [savings])

  return (
    <div className="space-y-6">
      {/* Hero Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="relative overflow-hidden bg-gradient-to-br from-emerald-500/10 via-emerald-500/5 to-transparent border-emerald-500/20">
          <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/5 to-transparent" />
          <div className="relative p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20">
                <DollarSign className="h-6 w-6 text-emerald-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Total Cost Savings</p>
                <p className="text-xs text-emerald-500">Per Week</p>
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex items-baseline gap-2">
                <span className="text-4xl font-bold text-emerald-500">
                  ${animatedValues.totalSavings.toLocaleString()}
                </span>
                <TrendingDown className="h-5 w-5 text-emerald-500" />
              </div>
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <span className="px-2 py-1 rounded-full bg-emerald-500/10 text-emerald-500 font-medium">
                  -24% costs
                </span>
                <span>vs baseline</span>
              </div>
            </div>
          </div>
        </Card>

        <Card className="relative overflow-hidden bg-gradient-to-br from-green-500/10 via-green-500/5 to-transparent border-green-500/20">
          <div className="absolute inset-0 bg-gradient-to-br from-green-500/5 to-transparent" />
          <div className="relative p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-3 rounded-xl bg-green-500/10 border border-green-500/20">
                <Leaf className="h-6 w-6 text-green-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">CO₂ Reduction</p>
                <p className="text-xs text-green-500">Per Week</p>
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex items-baseline gap-2">
                <span className="text-4xl font-bold text-green-500">{animatedValues.co2Reduction}</span>
                <span className="text-xl text-green-500/70">kg</span>
              </div>
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <span className="px-2 py-1 rounded-full bg-green-500/10 text-green-500 font-medium">
                  {savings.carbonReduction.treesEquivalent} trees
                </span>
                <span>equivalent</span>
              </div>
            </div>
          </div>
        </Card>

        <Card className="relative overflow-hidden bg-gradient-to-br from-blue-500/10 via-blue-500/5 to-transparent border-blue-500/20">
          <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-transparent" />
          <div className="relative p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-3 rounded-xl bg-blue-500/10 border border-blue-500/20">
                <Clock className="h-6 w-6 text-blue-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Time Saved</p>
                <p className="text-xs text-blue-500">Per Week</p>
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex items-baseline gap-2">
                <span className="text-4xl font-bold text-blue-500">{animatedValues.timeReduction}</span>
                <span className="text-xl text-blue-500/70">hrs</span>
              </div>
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <span className="px-2 py-1 rounded-full bg-blue-500/10 text-blue-500 font-medium">
                  +{savings.efficiency.onTimeImprovement}pp
                </span>
                <span>on-time rate</span>
              </div>
            </div>
          </div>
        </Card>
      </div>

      {/* Detailed Breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Cost Breakdown */}
        <Card className="p-6 bg-card/50 backdrop-blur-sm border-border/50">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 rounded-lg bg-emerald-500/10">
              <DollarSign className="h-5 w-5 text-emerald-500" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-foreground">Cost Savings Breakdown</h3>
              <p className="text-sm text-muted-foreground">Weekly operational savings</p>
            </div>
          </div>

          <div className="space-y-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">Fuel Costs</span>
                <span className="font-semibold text-emerald-500">${savings.costSavings.fuel}</span>
              </div>
              <Progress value={65} className="h-2 bg-emerald-500/10" />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">Labor Costs</span>
                <span className="font-semibold text-emerald-500">${savings.costSavings.labor}</span>
              </div>
              <Progress value={45} className="h-2 bg-emerald-500/10" />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">Maintenance</span>
                <span className="font-semibold text-emerald-500">${savings.costSavings.maintenance}</span>
              </div>
              <Progress value={25} className="h-2 bg-emerald-500/10" />
            </div>

            <div className="pt-4 border-t border-border">
              <div className="flex items-center justify-between">
                <span className="font-semibold text-foreground">Total Weekly Savings</span>
                <span className="text-2xl font-bold text-emerald-500">${savings.costSavings.total}</span>
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                Projected annual savings: ${(savings.costSavings.total * 52).toLocaleString()}
              </p>
            </div>
          </div>
        </Card>

        {/* Environmental Impact */}
        <Card className="p-6 bg-card/50 backdrop-blur-sm border-border/50">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 rounded-lg bg-green-500/10">
              <Leaf className="h-5 w-5 text-green-500" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-foreground">Environmental Impact</h3>
              <p className="text-sm text-muted-foreground">Carbon footprint reduction</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="p-4 rounded-xl bg-green-500/5 border border-green-500/10">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-muted-foreground">CO₂ Emissions Reduced</span>
                <span className="text-lg font-bold text-green-500">{savings.carbonReduction.percentReduction}%</span>
              </div>
              <Progress value={savings.carbonReduction.percentReduction} className="h-2 bg-green-500/10" />
              <p className="text-xs text-muted-foreground mt-2">
                Equivalent to {savings.carbonReduction.treesEquivalent} trees planted
              </p>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 rounded-xl bg-blue-500/5 border border-blue-500/10">
                <div className="flex items-center gap-2 mb-2">
                  <Zap className="h-4 w-4 text-blue-500" />
                  <span className="text-xs text-muted-foreground">Distance</span>
                </div>
                <p className="text-2xl font-bold text-blue-500">-{savings.efficiency.distanceReduction}%</p>
                <p className="text-xs text-muted-foreground mt-1">Total km reduced</p>
              </div>

              <div className="p-4 rounded-xl bg-amber-500/5 border border-amber-500/10">
                <div className="flex items-center gap-2 mb-2">
                  <Award className="h-4 w-4 text-amber-500" />
                  <span className="text-xs text-muted-foreground">Safety</span>
                </div>
                <p className="text-2xl font-bold text-amber-500">{savings.environmental.safetyScore}</p>
                <p className="text-xs text-muted-foreground mt-1">Safety score</p>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-red-500/5 border border-red-500/10">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Risky Distance Avoided</span>
                <span className="text-lg font-bold text-red-500">-{savings.environmental.riskyDistanceReduction}%</span>
              </div>
              <Progress value={savings.environmental.riskyDistanceReduction} className="h-2 bg-red-500/10 mt-2" />
            </div>
          </div>
        </Card>
      </div>

      {/* Comparison Chart */}
      <Card className="p-6 bg-card/50 backdrop-blur-sm border-border/50">
        <h3 className="text-lg font-semibold text-foreground mb-6">Baseline vs HivePath AI Optimization</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="p-4 rounded-xl bg-gradient-to-br from-red-500/10 to-red-500/5 border border-red-500/20">
            <p className="text-xs text-muted-foreground mb-2">Baseline Cost</p>
            <p className="text-2xl font-bold text-red-500">$10,083</p>
            <p className="text-xs text-red-500/70 mt-1">per week</p>
          </div>

          <div className="p-4 rounded-xl bg-gradient-to-br from-emerald-500/10 to-emerald-500/5 border border-emerald-500/20">
            <p className="text-xs text-muted-foreground mb-2">Optimized Cost</p>
            <p className="text-2xl font-bold text-emerald-500">$7,663</p>
            <p className="text-xs text-emerald-500/70 mt-1">per week</p>
          </div>

          <div className="p-4 rounded-xl bg-gradient-to-br from-red-500/10 to-red-500/5 border border-red-500/20">
            <p className="text-xs text-muted-foreground mb-2">Baseline CO₂</p>
            <p className="text-2xl font-bold text-red-500">2,621 kg</p>
            <p className="text-xs text-red-500/70 mt-1">per week</p>
          </div>

          <div className="p-4 rounded-xl bg-gradient-to-br from-green-500/10 to-green-500/5 border border-green-500/20">
            <p className="text-xs text-muted-foreground mb-2">Optimized CO₂</p>
            <p className="text-2xl font-bold text-green-500">2,136 kg</p>
            <p className="text-xs text-green-500/70 mt-1">per week</p>
          </div>
        </div>
      </Card>
    </div>
  )
}
