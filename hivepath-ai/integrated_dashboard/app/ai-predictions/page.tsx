"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { fetchPredictions, type Prediction } from "@/lib/api"
import { Brain, TrendingUp, Clock, AlertCircle } from "lucide-react"

export default function AIPredictionsPage() {
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function loadPredictions() {
      try {
        const data = await fetchPredictions()
        setPredictions(data)
      } catch (error) {
        console.error("[v0] Failed to load predictions:", error)
      } finally {
        setLoading(false)
      }
    }
    loadPredictions()
  }, [])

  if (loading) {
    return (
      <div className="container mx-auto p-6 space-y-6">
        <div className="space-y-2">
          <Skeleton className="h-8 w-64" />
          <Skeleton className="h-4 w-96" />
        </div>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {[1, 2, 3, 4, 5].map((i) => (
            <Skeleton key={i} className="h-64" />
          ))}
        </div>
      </div>
    )
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.7) return "bg-green-500/10 text-green-700 dark:text-green-400"
    if (confidence >= 0.4) return "bg-yellow-500/10 text-yellow-700 dark:text-yellow-400"
    return "bg-red-500/10 text-red-700 dark:text-red-400"
  }

  const getConfidenceLabel = (confidence: number) => {
    if (confidence >= 0.7) return "High"
    if (confidence >= 0.4) return "Medium"
    return "Low"
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold tracking-tight flex items-center gap-2">
          <Brain className="h-8 w-8 text-primary" />
          AI Service Time Predictions
        </h1>
        <p className="text-muted-foreground">
          Machine learning predictions for delivery service times using Graph Neural Networks
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {predictions.map((prediction) => (
          <Card key={prediction.location_id} className="hover:shadow-lg transition-shadow">
            <CardHeader>
              <CardTitle className="text-lg flex items-center justify-between">
                <span className="truncate">{prediction.location_name}</span>
                <Badge variant="outline" className="ml-2 shrink-0">
                  {prediction.model_type.toUpperCase()}
                </Badge>
              </CardTitle>
              <CardDescription className="font-mono text-xs">{prediction.location_id}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <TrendingUp className="h-4 w-4" />
                    <span>Predicted Time</span>
                  </div>
                  <span className="text-2xl font-bold text-primary">{prediction.predicted_time.toFixed(1)}m</span>
                </div>

                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <Clock className="h-4 w-4" />
                    <span>Historical Avg</span>
                  </div>
                  <span className="text-lg font-semibold">{prediction.historical_avg.toFixed(1)}m</span>
                </div>

                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <AlertCircle className="h-4 w-4" />
                    <span>Confidence</span>
                  </div>
                  <Badge className={getConfidenceColor(prediction.confidence)}>
                    {getConfidenceLabel(prediction.confidence)} ({(prediction.confidence * 100).toFixed(0)}%)
                  </Badge>
                </div>
              </div>

              <div className="pt-3 border-t space-y-2">
                <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">Key Factors</p>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Demand:</span>
                    <span className="font-medium">{prediction.factors.demand}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Access:</span>
                    <span className="font-medium">{(prediction.factors.access_score * 100).toFixed(0)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Weather:</span>
                    <span className="font-medium">{(prediction.factors.weather_risk * 100).toFixed(0)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Traffic:</span>
                    <span className="font-medium">{(prediction.factors.traffic_risk * 100).toFixed(0)}%</span>
                  </div>
                  <div className="flex justify-between col-span-2">
                    <span className="text-muted-foreground">Peak Hour:</span>
                    <span className="font-medium">{prediction.factors.peak_hour.toFixed(1)}x</span>
                  </div>
                </div>
              </div>

              {prediction.predicted_time > prediction.historical_avg * 1.2 && (
                <div className="pt-3 border-t">
                  <div className="flex items-start gap-2 text-xs text-amber-600 dark:text-amber-400">
                    <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
                    <span>
                      Predicted time is {((prediction.predicted_time / prediction.historical_avg - 1) * 100).toFixed(0)}
                      % higher than historical average
                    </span>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        ))}
      </div>

      {predictions.length === 0 && (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <Brain className="h-12 w-12 text-muted-foreground mb-4" />
            <p className="text-lg font-semibold text-muted-foreground">No predictions available</p>
            <p className="text-sm text-muted-foreground mt-2">
              Check back later for AI-generated service time predictions
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
