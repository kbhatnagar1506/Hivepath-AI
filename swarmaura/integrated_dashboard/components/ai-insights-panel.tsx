"use client"

import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Brain, AlertCircle, CheckCircle2, Sparkles } from "lucide-react"
import { useEffect, useState } from "react"
import { fetchPredictions, type Prediction } from "@/lib/api"

interface AIInsightsPanelProps {
  predictions?: Prediction[]
  isLoading?: boolean
}

export function AIInsightsPanel({ predictions: propPredictions, isLoading: propIsLoading }: AIInsightsPanelProps) {
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function loadPredictions() {
      try {
        // Use passed predictions if available, otherwise fetch
        if (propPredictions && propPredictions.length > 0) {
          setPredictions(propPredictions)
          setLoading(false)
        } else {
          const data = await fetchPredictions()
          setPredictions(data)
          setLoading(false)
        }
      } catch (error) {
        console.error("Failed to load predictions:", error)
        setLoading(false)
      }
    }
    loadPredictions()
  }, [propPredictions])

  if (loading) {
    return (
      <Card className="p-6">
        <div className="flex items-center justify-center h-40">
          <div className="animate-spin h-8 w-8 border-4 border-primary border-t-transparent rounded-full" />
        </div>
      </Card>
    )
  }

  const avgConfidence = predictions.reduce((sum, p) => sum + p.confidence, 0) / predictions.length

  return (
    <div className="space-y-6 overflow-y-auto max-h-full">
      <Card className="p-6 bg-gradient-to-br from-purple-500/10 via-blue-500/5 to-cyan-500/10 border-purple-500/20">
        <div className="flex items-center gap-3 mb-4">
          <div className="flex h-12 w-12 items-center justify-center rounded-full bg-gradient-to-br from-purple-500 to-blue-500">
            <Brain className="h-6 w-6 text-white" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-foreground">AI Predictions</h3>
            <p className="text-sm text-muted-foreground">Service time estimates powered by GNN</p>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-4 mb-6">
          <div className="text-center">
            <div className="text-2xl font-bold text-foreground">{predictions.length}</div>
            <div className="text-xs text-muted-foreground">Predictions</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-foreground">{(avgConfidence * 100).toFixed(0)}%</div>
            <div className="text-xs text-muted-foreground">Avg Confidence</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-foreground">
              {predictions.filter((p) => p.confidence > 0.85).length}
            </div>
            <div className="text-xs text-muted-foreground">High Confidence</div>
          </div>
        </div>

        <div className="space-y-3">
          {predictions.map((prediction) => (
            <Card key={prediction.location_id} className="p-4 bg-card/50 backdrop-blur-sm">
              <div className="flex items-start justify-between mb-3">
                <div>
                  <div className="font-medium text-foreground">{prediction.location_name}</div>
                  <div className="text-xs text-muted-foreground mt-0.5">
                    Predicted: {prediction.predicted_time.toFixed(1)} min
                  </div>
                </div>
                <Badge
                  variant={prediction.confidence > 0.85 ? "default" : "secondary"}
                  className="flex items-center gap-1"
                >
                  {prediction.confidence > 0.85 ? (
                    <CheckCircle2 className="h-3 w-3" />
                  ) : (
                    <AlertCircle className="h-3 w-3" />
                  )}
                  {(prediction.confidence * 100).toFixed(0)}%
                </Badge>
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">Confidence Level</span>
                  <span className="font-medium text-foreground">{(prediction.confidence * 100).toFixed(1)}%</span>
                </div>
                <Progress value={prediction.confidence * 100} className="h-2" />
              </div>

              <div className="grid grid-cols-3 gap-2 mt-3 pt-3 border-t border-border">
                <div className="text-center">
                  <div className="text-xs text-muted-foreground">Demand</div>
                  <div className="text-sm font-medium text-foreground">{prediction.factors.demand}</div>
                </div>
                <div className="text-center">
                  <div className="text-xs text-muted-foreground">Access</div>
                  <div className="text-sm font-medium text-foreground">
                    {prediction.factors.access_score.toFixed(2)}
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-xs text-muted-foreground">Traffic</div>
                  <div className="text-sm font-medium text-foreground">
                    {(prediction.factors.traffic_risk * 100).toFixed(0)}%
                  </div>
                </div>
              </div>
            </Card>
          ))}
        </div>
      </Card>

      <Card className="p-4 bg-gradient-to-r from-emerald-500/10 to-teal-500/10 border-emerald-500/20">
        <div className="flex items-start gap-3">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-emerald-500/20">
            <Sparkles className="h-5 w-5 text-emerald-500" />
          </div>
          <div>
            <div className="font-medium text-foreground mb-1">AI Model Performance</div>
            <div className="text-sm text-muted-foreground">
              The GNN model is achieving {(avgConfidence * 100).toFixed(0)}% average confidence across all predictions,
              with {predictions.filter((p) => p.confidence > 0.85).length} high-confidence estimates.
            </div>
          </div>
        </div>
      </Card>
    </div>
  )
}
