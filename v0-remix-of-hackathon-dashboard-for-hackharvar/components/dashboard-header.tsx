"use client"

import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { AlertTriangle, Zap, Clock, Activity } from "lucide-react"

interface DashboardHeaderProps {
  activeRun: "baseline" | "replan"
  onRunChange: (run: "baseline" | "replan") => void
  riskShaper: boolean
  onRiskShaperChange: (value: boolean) => void
  warmStart: boolean
  onWarmStartChange: (value: boolean) => void
  serviceTimeGNN: boolean
  onServiceTimeGNNChange: (value: boolean) => void
}

export function DashboardHeader({
  activeRun,
  onRunChange,
  riskShaper,
  onRiskShaperChange,
  warmStart,
  onWarmStartChange,
  serviceTimeGNN,
  onServiceTimeGNNChange,
}: DashboardHeaderProps) {
  return (
    <header className="border-b border-border bg-card">
      <div className="flex items-center justify-between px-6 py-4">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-3">
            <div className="relative flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-primary via-primary to-accent shadow-lg">
              <Zap className="h-5 w-5 text-primary-foreground" />
              <div className="absolute inset-0 rounded-lg bg-primary/20 animate-pulse" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-foreground tracking-tight">HivePath AI</h1>
              <p className="text-xs text-muted-foreground">Self-Healing Logistics Platform</p>
            </div>
          </div>

          <div className="h-8 w-px bg-border" />

          <div className="flex items-center gap-3">
            <Label htmlFor="run-select" className="text-sm text-muted-foreground">
              Run:
            </Label>
            <Select value={activeRun} onValueChange={(v) => onRunChange(v as "baseline" | "replan")}>
              <SelectTrigger id="run-select" className="w-32 transition-smooth hover:border-primary/50">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="baseline">Baseline</SelectItem>
                <SelectItem value="replan">Replan</SelectItem>
              </SelectContent>
            </Select>
            <Badge variant="outline" className="font-mono text-xs animate-fade-in">
              <Clock className="mr-1 h-3 w-3" />
              {activeRun === "replan" ? "1.1s" : "1.9s"}
            </Badge>
            {activeRun === "replan" && warmStart && (
              <Badge
                variant="outline"
                className="font-mono text-xs bg-success/10 text-success border-success/20 animate-slide-up"
              >
                <Activity className="mr-1 h-3 w-3" />
                -43% solve
              </Badge>
            )}
          </div>
        </div>

        <div className="flex items-center gap-6">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Switch
                id="risk-shaper"
                checked={riskShaper}
                onCheckedChange={onRiskShaperChange}
                className="transition-smooth"
              />
              <Label htmlFor="risk-shaper" className="text-sm cursor-pointer transition-smooth hover:text-foreground">
                Risk Shaper
              </Label>
              {riskShaper && activeRun === "replan" && (
                <Badge variant="secondary" className="text-xs animate-slide-up">
                  Active
                </Badge>
              )}
            </div>

            <div className="flex items-center gap-2">
              <Switch
                id="warm-start"
                checked={warmStart}
                onCheckedChange={onWarmStartChange}
                className="transition-smooth"
              />
              <Label htmlFor="warm-start" className="text-sm cursor-pointer transition-smooth hover:text-foreground">
                Warm-Start
              </Label>
              {warmStart && activeRun === "replan" && (
                <Badge variant="secondary" className="text-xs animate-slide-up">
                  Active
                </Badge>
              )}
            </div>

            <div className="flex items-center gap-2">
              <Switch
                id="service-time"
                checked={serviceTimeGNN}
                onCheckedChange={onServiceTimeGNNChange}
                className="transition-smooth"
              />
              <Label htmlFor="service-time" className="text-sm cursor-pointer transition-smooth hover:text-foreground">
                Service-Time GNN
              </Label>
              {serviceTimeGNN && activeRun === "replan" && (
                <Badge variant="secondary" className="text-xs animate-slide-up">
                  Active
                </Badge>
              )}
            </div>
          </div>

          <Button
            variant="destructive"
            size="sm"
            className="transition-smooth hover:shadow-lg hover:shadow-destructive/20"
          >
            <AlertTriangle className="mr-2 h-4 w-4" />
            Inject Incident
          </Button>
        </div>
      </div>

      <div className="flex items-center gap-2 border-t border-border px-6 py-2">
        <span className="text-xs text-muted-foreground mr-2">Strategy:</span>
        <Button
          variant="secondary"
          size="sm"
          className="h-7 text-xs transition-smooth hover:bg-primary hover:text-primary-foreground"
        >
          Fast
        </Button>
        <Button variant="default" size="sm" className="h-7 text-xs">
          Balanced
        </Button>
        <Button
          variant="secondary"
          size="sm"
          className="h-7 text-xs transition-smooth hover:bg-primary hover:text-primary-foreground"
        >
          Quality
        </Button>
      </div>
    </header>
  )
}
