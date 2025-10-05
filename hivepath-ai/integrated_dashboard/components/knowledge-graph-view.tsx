"use client"

import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Network, Layers, Maximize2, Zap, MapPin, Truck, AlertTriangle, Cloud, Clock } from "lucide-react"
import { useState, useEffect } from "react"

type NodeType = "depot" | "stop" | "vehicle" | "route" | "risk" | "weather" | "traffic" | "time"

interface GraphNode {
  id: string
  type: NodeType
  label: string
  x: number
  y: number
  z: number
  connections: string[]
  data: any
}

export function KnowledgeGraphView() {
  const [selectedNode, setSelectedNode] = useState<string>("S3")
  const [graphLayer, setGraphLayer] = useState<"all" | "stops" | "risk" | "routes">("all")
  const [rotation, setRotation] = useState(0)
  const [hoveredNode, setHoveredNode] = useState<string | null>(null)

  useEffect(() => {
    console.log("[v0] KnowledgeGraphView component mounted")
    return () => console.log("[v0] KnowledgeGraphView component unmounted")
  }, [])

  useEffect(() => {
    const interval = setInterval(() => {
      setRotation((prev) => (prev + 0.3) % 360)
    }, 50)
    return () => clearInterval(interval)
  }, [])

  const generateNodes = (): GraphNode[] => {
    const nodes: GraphNode[] = []

    // Central depot
    nodes.push({
      id: "depot",
      type: "depot",
      label: "Central Depot",
      x: 0,
      y: 0,
      z: 0,
      connections: [],
      data: { capacity: 50, utilization: 0.82 },
    })

    // 12 delivery stops in a sphere
    for (let i = 0; i < 12; i++) {
      const phi = Math.acos(-1 + (2 * i) / 12)
      const theta = Math.sqrt(12 * Math.PI) * phi
      const radius = 120
      nodes.push({
        id: `S${i + 1}`,
        type: "stop",
        label: `Stop ${i + 1}`,
        x: radius * Math.sin(phi) * Math.cos(theta),
        y: radius * Math.sin(phi) * Math.sin(theta),
        z: radius * Math.cos(phi),
        connections: ["depot"],
        data: {
          demand: Math.random() * 100,
          risk: Math.random(),
          serviceTime: 3 + Math.random() * 5,
        },
      })
    }

    // 6 vehicles
    for (let i = 0; i < 6; i++) {
      const angle = (i * 60 * Math.PI) / 180
      const radius = 80
      nodes.push({
        id: `V${i + 1}`,
        type: "vehicle",
        label: `Vehicle ${i + 1}`,
        x: radius * Math.cos(angle),
        y: radius * Math.sin(angle),
        z: 50,
        connections: ["depot"],
        data: {
          capacity: 1000 + Math.random() * 500,
          type: i % 2 === 0 ? "Truck" : "Van",
        },
      })
    }

    // 4 route nodes
    for (let i = 0; i < 4; i++) {
      const angle = (i * 90 * Math.PI) / 180
      const radius = 150
      nodes.push({
        id: `R${i + 1}`,
        type: "route",
        label: `Route ${i + 1}`,
        x: radius * Math.cos(angle),
        y: radius * Math.sin(angle),
        z: -30,
        connections: [`V${i + 1}`, `S${i * 3 + 1}`, `S${i * 3 + 2}`, `S${i * 3 + 3}`],
        data: {
          distance: 15 + Math.random() * 20,
          stops: 3 + Math.floor(Math.random() * 3),
        },
      })
    }

    // 8 risk factor nodes
    for (let i = 0; i < 8; i++) {
      const angle = (i * 45 * Math.PI) / 180
      const radius = 170
      nodes.push({
        id: `RISK${i + 1}`,
        type: "risk",
        label: ["Traffic", "Weather", "Crime", "Construction", "Lighting", "Curb", "Slope", "Congestion"][i],
        x: radius * Math.cos(angle),
        y: radius * Math.sin(angle),
        z: -60,
        connections: [],
        data: {
          severity: Math.random(),
          impact: Math.random() * 0.3,
        },
      })
    }

    // 4 weather nodes
    for (let i = 0; i < 4; i++) {
      const angle = ((i * 90 + 45) * Math.PI) / 180
      const radius = 160
      nodes.push({
        id: `W${i + 1}`,
        type: "weather",
        label: ["Clear", "Rain", "Snow", "Fog"][i],
        x: radius * Math.cos(angle),
        y: radius * Math.sin(angle),
        z: 80,
        connections: [],
        data: { probability: Math.random() },
      })
    }

    // 6 time window nodes
    for (let i = 0; i < 6; i++) {
      const angle = (i * 60 * Math.PI) / 180
      const radius = 130
      nodes.push({
        id: `T${i + 1}`,
        type: "time",
        label: `${8 + i * 2}:00-${10 + i * 2}:00`,
        x: radius * Math.cos(angle),
        y: radius * Math.sin(angle),
        z: -90,
        connections: [],
        data: { priority: Math.random() },
      })
    }

    return nodes
  }

  const nodes = generateNodes()

  useEffect(() => {
    console.log("[v0] Generated nodes:", nodes.length)
  }, [])

  const visibleNodes = nodes.filter((node) => {
    if (graphLayer === "stops") return node.type === "stop" || node.type === "depot"
    if (graphLayer === "risk") return node.type === "risk" || node.type === "stop" || node.type === "depot"
    if (graphLayer === "routes") return node.type === "route" || node.type === "vehicle" || node.type === "depot"
    return true
  })

  useEffect(() => {
    console.log("[v0] Visible nodes:", visibleNodes.length, "Layer:", graphLayer)
  }, [graphLayer])

  const project3D = (x: number, y: number, z: number) => {
    const rad = (rotation * Math.PI) / 180
    const rotatedX = x * Math.cos(rad) - y * Math.sin(rad)
    const rotatedY = x * Math.sin(rad) + y * Math.cos(rad)
    const scale = 300 / (300 + z)
    return {
      x: rotatedX * scale + 770, // Moved 10px more to the right
      y: rotatedY * scale + 280, // Better centered
      scale: scale,
    }
  }

  const getNodeStyle = (node: GraphNode, isSelected: boolean, isHovered: boolean) => {
    const baseStyles = {
      depot: "bg-gradient-to-br from-blue-500 via-blue-600 to-indigo-600 ring-blue-400 shadow-blue-500/50",
      stop: "bg-gradient-to-br from-emerald-400 via-emerald-500 to-teal-600 ring-emerald-400 shadow-emerald-500/50",
      vehicle: "bg-gradient-to-br from-purple-400 via-purple-500 to-fuchsia-600 ring-purple-400 shadow-purple-500/50",
      route: "bg-gradient-to-br from-amber-400 via-orange-500 to-orange-600 ring-amber-400 shadow-amber-500/50",
      risk: "bg-gradient-to-br from-red-400 via-red-500 to-rose-600 ring-red-400 shadow-red-500/50",
      weather: "bg-gradient-to-br from-cyan-400 via-sky-500 to-blue-500 ring-cyan-400 shadow-cyan-500/50",
      traffic: "bg-gradient-to-br from-orange-400 via-orange-500 to-red-500 ring-orange-400 shadow-orange-500/50",
      time: "bg-gradient-to-br from-indigo-400 via-violet-500 to-purple-600 ring-indigo-400 shadow-indigo-500/50",
    }

    const sizeMap = {
      depot: isSelected ? "h-20 w-20" : "h-16 w-16",
      stop: isSelected ? "h-14 w-14" : "h-12 w-12",
      vehicle: isSelected ? "h-12 w-12" : "h-10 w-10",
      route: isSelected ? "h-11 w-11" : "h-9 w-9",
      risk: isSelected ? "h-10 w-10" : "h-8 w-8",
      weather: isSelected ? "h-10 w-10" : "h-8 w-8",
      traffic: isSelected ? "h-10 w-10" : "h-8 w-8",
      time: isSelected ? "h-9 w-9" : "h-7 w-7",
    }

    return {
      color: baseStyles[node.type],
      size: sizeMap[node.type],
      ring: isSelected ? "ring-4" : isHovered ? "ring-3" : "ring-2",
    }
  }

  const selectedNodeData = nodes.find((n) => n.id === selectedNode)

  return (
    <div className="flex h-full overflow-y-auto">
      <div className="relative flex-1 bg-gradient-to-br from-slate-950 via-blue-950/20 to-slate-950">
        <div className="absolute left-4 top-4 z-10 space-y-3">
          <Card className="p-4 animate-fade-in bg-slate-900/70 backdrop-blur-xl border-slate-700/50 shadow-2xl">
            <div className="flex items-center gap-2 mb-4">
              <div className="p-1.5 rounded-lg bg-blue-500/20">
                <Network className="h-4 w-4 text-blue-400" />
              </div>
              <span className="text-sm font-semibold text-slate-100">Graph Layers</span>
            </div>
            <div className="space-y-2">
              <Button
                variant={graphLayer === "all" ? "default" : "ghost"}
                size="sm"
                className="w-full justify-start text-xs hover:bg-slate-800 transition-all"
                onClick={() => setGraphLayer("all")}
              >
                <Layers className="mr-2 h-3 w-3" />
                All Nodes ({nodes.length})
              </Button>
              <Button
                variant={graphLayer === "stops" ? "default" : "ghost"}
                size="sm"
                className="w-full justify-start text-xs hover:bg-slate-800 transition-all"
                onClick={() => setGraphLayer("stops")}
              >
                <MapPin className="mr-2 h-3 w-3" />
                Stops Only
              </Button>
              <Button
                variant={graphLayer === "risk" ? "default" : "ghost"}
                size="sm"
                className="w-full justify-start text-xs hover:bg-slate-800 transition-all"
                onClick={() => setGraphLayer("risk")}
              >
                <AlertTriangle className="mr-2 h-3 w-3" />
                Risk Factors
              </Button>
              <Button
                variant={graphLayer === "routes" ? "default" : "ghost"}
                size="sm"
                className="w-full justify-start text-xs hover:bg-slate-800 transition-all"
                onClick={() => setGraphLayer("routes")}
              >
                <Truck className="mr-2 h-3 w-3" />
                Routes & Vehicles
              </Button>
            </div>
          </Card>

          <Card className="p-4 animate-fade-in bg-slate-900/70 backdrop-blur-xl border-slate-700/50 shadow-2xl">
            <div className="text-xs font-semibold text-slate-100 mb-3 flex items-center gap-2">
              <div className="h-1 w-1 rounded-full bg-blue-400 animate-pulse" />
              Node Types
            </div>
            <div className="space-y-2">
              <div className="flex items-center gap-2.5 group">
                <div className="h-3.5 w-3.5 rounded-full bg-gradient-to-br from-blue-500 via-blue-600 to-indigo-600 shadow-lg shadow-blue-500/30 group-hover:scale-110 transition-transform" />
                <span className="text-xs text-slate-300 font-medium">Depot (1)</span>
              </div>
              <div className="flex items-center gap-2.5 group">
                <div className="h-3.5 w-3.5 rounded-full bg-gradient-to-br from-emerald-400 via-emerald-500 to-teal-600 shadow-lg shadow-emerald-500/30 group-hover:scale-110 transition-transform" />
                <span className="text-xs text-slate-300 font-medium">Stops (12)</span>
              </div>
              <div className="flex items-center gap-2.5 group">
                <div className="h-3.5 w-3.5 rounded-full bg-gradient-to-br from-purple-400 via-purple-500 to-fuchsia-600 shadow-lg shadow-purple-500/30 group-hover:scale-110 transition-transform" />
                <span className="text-xs text-slate-300 font-medium">Vehicles (6)</span>
              </div>
              <div className="flex items-center gap-2.5 group">
                <div className="h-3.5 w-3.5 rounded-full bg-gradient-to-br from-amber-400 via-orange-500 to-orange-600 shadow-lg shadow-amber-500/30 group-hover:scale-110 transition-transform" />
                <span className="text-xs text-slate-300 font-medium">Routes (4)</span>
              </div>
              <div className="flex items-center gap-2.5 group">
                <div className="h-3.5 w-3.5 rounded-full bg-gradient-to-br from-red-400 via-red-500 to-rose-600 shadow-lg shadow-red-500/30 group-hover:scale-110 transition-transform" />
                <span className="text-xs text-slate-300 font-medium">Risk Factors (8)</span>
              </div>
              <div className="flex items-center gap-2.5 group">
                <div className="h-3.5 w-3.5 rounded-full bg-gradient-to-br from-cyan-400 via-sky-500 to-blue-500 shadow-lg shadow-cyan-500/30 group-hover:scale-110 transition-transform" />
                <span className="text-xs text-slate-300 font-medium">Weather (4)</span>
              </div>
              <div className="flex items-center gap-2.5 group">
                <div className="h-3.5 w-3.5 rounded-full bg-gradient-to-br from-indigo-400 via-violet-500 to-purple-600 shadow-lg shadow-indigo-500/30 group-hover:scale-110 transition-transform" />
                <span className="text-xs text-slate-300 font-medium">Time Windows (6)</span>
              </div>
            </div>
          </Card>
        </div>

        <div className="flex h-full items-center justify-center p-8">
          <div className="relative h-[600px] w-[1100px]">
            <svg className="absolute inset-0 h-full w-full pointer-events-none">
              <defs>
                <linearGradient id="lineGradient1" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#60a5fa" stopOpacity="0.8" />
                  <stop offset="50%" stopColor="#a78bfa" stopOpacity="1" />
                  <stop offset="100%" stopColor="#60a5fa" stopOpacity="0.8" />
                </linearGradient>
                <linearGradient id="lineGradient2" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#34d399" stopOpacity="0.8" />
                  <stop offset="50%" stopColor="#60a5fa" stopOpacity="1" />
                  <stop offset="100%" stopColor="#34d399" stopOpacity="0.8" />
                </linearGradient>
                <linearGradient id="lineGradient3" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#f59e0b" stopOpacity="0.8" />
                  <stop offset="50%" stopColor="#ec4899" stopOpacity="1" />
                  <stop offset="100%" stopColor="#f59e0b" stopOpacity="0.8" />
                </linearGradient>
                <radialGradient id="glowGradient">
                  <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.8" />
                  <stop offset="50%" stopColor="#8b5cf6" stopOpacity="0.4" />
                  <stop offset="100%" stopColor="#3b82f6" stopOpacity="0" />
                </radialGradient>
                <filter id="glow">
                  <feGaussianBlur stdDeviation="4" result="coloredBlur" />
                  <feMerge>
                    <feMergeNode in="coloredBlur" />
                    <feMergeNode in="coloredBlur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
                <filter id="strongGlow">
                  <feGaussianBlur stdDeviation="6" result="coloredBlur" />
                  <feMerge>
                    <feMergeNode in="coloredBlur" />
                    <feMergeNode in="coloredBlur" />
                    <feMergeNode in="coloredBlur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
              </defs>

              {visibleNodes.map((node, idx) => {
                const pos1 = project3D(node.x, node.y, node.z)
                return node.connections.map((connId, connIdx) => {
                  const connNode = visibleNodes.find((n) => n.id === connId)
                  if (!connNode) return null
                  const pos2 = project3D(connNode.x, connNode.y, connNode.z)

                  // Cycle through different gradient styles
                  const gradientId = `lineGradient${(idx % 3) + 1}`
                  const isSelected = selectedNode === node.id || selectedNode === connId

                  return (
                    <g key={`${node.id}-${connId}`}>
                      {/* Background glow line */}
                      <line
                        x1={pos1.x}
                        y1={pos1.y}
                        x2={pos2.x}
                        y2={pos2.y}
                        stroke={`url(#${gradientId})`}
                        strokeWidth={isSelected ? "8" : "6"}
                        opacity={isSelected ? 0.4 : 0.2}
                        filter="url(#strongGlow)"
                      />
                      {/* Main line */}
                      <line
                        x1={pos1.x}
                        y1={pos1.y}
                        x2={pos2.x}
                        y2={pos2.y}
                        stroke={`url(#${gradientId})`}
                        strokeWidth={isSelected ? "4" : "3"}
                        opacity={isSelected ? 0.95 : 0.75}
                        filter="url(#glow)"
                        strokeLinecap="round"
                      />
                      {/* Animated pulse for selected connections */}
                      {isSelected && (
                        <line
                          x1={pos1.x}
                          y1={pos1.y}
                          x2={pos2.x}
                          y2={pos2.y}
                          stroke="#ffffff"
                          strokeWidth="2"
                          opacity="0.6"
                          strokeLinecap="round"
                          className="animate-pulse"
                        />
                      )}
                    </g>
                  )
                })
              })}

              {/* Enhanced glow effects for selected node */}
              {selectedNodeData &&
                (() => {
                  const pos = project3D(selectedNodeData.x, selectedNodeData.y, selectedNodeData.z)
                  return (
                    <>
                      <circle cx={pos.x} cy={pos.y} r={80} fill="url(#glowGradient)" className="animate-pulse" />
                      <circle
                        cx={pos.x}
                        cy={pos.y}
                        r={50}
                        fill="none"
                        stroke="#60a5fa"
                        strokeWidth="3"
                        opacity="0.5"
                        className="animate-ping"
                      />
                      <circle cx={pos.x} cy={pos.y} r={35} fill="none" stroke="#a78bfa" strokeWidth="2" opacity="0.7" />
                    </>
                  )
                })()}
            </svg>

            {visibleNodes.map((node, idx) => {
              const pos = project3D(node.x, node.y, node.z)
              const isSelected = selectedNode === node.id
              const isHovered = hoveredNode === node.id
              const style = getNodeStyle(node, isSelected, isHovered)

              return (
                <button
                  key={node.id}
                  onClick={() => setSelectedNode(node.id)}
                  onMouseEnter={() => setHoveredNode(node.id)}
                  onMouseLeave={() => setHoveredNode(null)}
                  className={`absolute flex items-center justify-center rounded-full shadow-2xl transition-all duration-300 cursor-pointer hover:scale-110 ${style.color} ${style.size} ${style.ring}`}
                  style={{
                    left: `${pos.x}px`,
                    top: `${pos.y}px`,
                    transform: `translate(-50%, -50%) scale(${pos.scale})`,
                    zIndex: Math.floor(pos.scale * 100),
                    opacity: pos.scale * 0.7 + 0.3,
                    animationDelay: `${idx * 0.02}s`,
                  }}
                >
                  <span className="text-xs font-bold text-white drop-shadow-lg">
                    {node.type === "depot" ? "D" : node.id.substring(0, 2)}
                  </span>
                </button>
              )
            })}

            <div className="absolute bottom-4 left-4">
              <Badge
                variant="outline"
                className="bg-slate-900/80 backdrop-blur-xl border-slate-700/50 text-slate-300 shadow-lg px-3 py-1.5"
              >
                <Zap className="mr-1.5 h-3 w-3 text-blue-400 animate-pulse" />
                Auto-rotating • {visibleNodes.length} nodes visible
              </Badge>
            </div>

            <div className="absolute bottom-4 right-4">
              <Button
                variant="outline"
                size="sm"
                className="bg-slate-900/80 backdrop-blur-xl border-slate-700/50 hover:bg-slate-800 shadow-lg"
              >
                <Maximize2 className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      </div>

      <div className="w-80 border-l border-slate-700/50 bg-gradient-to-b from-slate-900 to-slate-950 p-4 overflow-y-auto">
        {selectedNodeData && (
          <>
            <div className="flex items-center gap-3 mb-2">
              <div className="p-2 rounded-lg bg-slate-800/50 backdrop-blur-sm">
                {selectedNodeData.type === "depot" && <MapPin className="h-5 w-5 text-blue-400" />}
                {selectedNodeData.type === "stop" && <MapPin className="h-5 w-5 text-emerald-400" />}
                {selectedNodeData.type === "vehicle" && <Truck className="h-5 w-5 text-purple-400" />}
                {selectedNodeData.type === "route" && <Network className="h-5 w-5 text-amber-400" />}
                {selectedNodeData.type === "risk" && <AlertTriangle className="h-5 w-5 text-red-400" />}
                {selectedNodeData.type === "weather" && <Cloud className="h-5 w-5 text-cyan-400" />}
                {selectedNodeData.type === "time" && <Clock className="h-5 w-5 text-indigo-400" />}
              </div>
              <div>
                <h3 className="text-lg font-bold text-slate-100">{selectedNodeData.label}</h3>
                <p className="text-xs text-slate-500">
                  {selectedNodeData.id} • {selectedNodeData.type}
                </p>
              </div>
            </div>

            <div className="space-y-4 mt-6">
              <div>
                <div className="text-xs font-semibold text-slate-400 mb-2 uppercase tracking-wider">Node Type</div>
                <Badge variant="secondary" className="capitalize font-medium">
                  {selectedNodeData.type}
                </Badge>
              </div>

              {selectedNodeData.type === "stop" && (
                <>
                  <div className="p-4 rounded-lg bg-slate-800/50 backdrop-blur-sm border border-slate-700/50">
                    <div className="text-xs font-semibold text-slate-400 mb-3 uppercase tracking-wider">
                      GNN Features
                    </div>
                    <div className="space-y-3">
                      <div>
                        <div className="flex justify-between text-xs mb-1.5">
                          <span className="text-slate-400">Demand score</span>
                          <span className="text-slate-100 font-bold font-mono">
                            {(selectedNodeData.data.demand / 100).toFixed(2)}
                          </span>
                        </div>
                        <div className="w-full h-2 bg-slate-700/50 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 shadow-lg shadow-emerald-500/50 transition-all duration-500"
                            style={{ width: `${selectedNodeData.data.demand}%` }}
                          />
                        </div>
                      </div>
                      <div>
                        <div className="flex justify-between text-xs mb-1.5">
                          <span className="text-slate-400">Risk factor</span>
                          <span className="text-slate-100 font-bold font-mono">
                            {selectedNodeData.data.risk.toFixed(2)}
                          </span>
                        </div>
                        <div className="w-full h-2 bg-slate-700/50 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-red-500 to-rose-400 shadow-lg shadow-red-500/50 transition-all duration-500"
                            style={{ width: `${selectedNodeData.data.risk * 100}%` }}
                          />
                        </div>
                      </div>
                      <div className="flex justify-between text-sm pt-2 border-t border-slate-700/50">
                        <span className="text-slate-400">Service time</span>
                        <span className="text-slate-100 font-bold font-mono">
                          {selectedNodeData.data.serviceTime.toFixed(1)} min
                        </span>
                      </div>
                    </div>
                  </div>
                </>
              )}

              {selectedNodeData.type === "vehicle" && (
                <div className="p-4 rounded-lg bg-slate-800/50 backdrop-blur-sm border border-slate-700/50">
                  <div className="text-xs font-semibold text-slate-400 mb-3 uppercase tracking-wider">
                    Vehicle Details
                  </div>
                  <div className="space-y-3">
                    <div className="flex justify-between text-xs mb-1.5">
                      <span className="text-slate-400">Type</span>
                      <span className="text-slate-100 font-bold font-mono">{selectedNodeData.data.type}</span>
                    </div>
                    <div className="flex justify-between text-xs mb-1.5">
                      <span className="text-slate-400">Capacity</span>
                      <span className="text-slate-100 font-bold font-mono">
                        {selectedNodeData.data.capacity.toFixed(0)} kg
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {selectedNodeData.type === "route" && (
                <div className="p-4 rounded-lg bg-slate-800/50 backdrop-blur-sm border border-slate-700/50">
                  <div className="text-xs font-semibold text-slate-400 mb-3 uppercase tracking-wider">
                    Route Details
                  </div>
                  <div className="space-y-3">
                    <div className="flex justify-between text-xs mb-1.5">
                      <span className="text-slate-400">Distance</span>
                      <span className="text-slate-100 font-bold font-mono">
                        {selectedNodeData.data.distance.toFixed(1)} km
                      </span>
                    </div>
                    <div className="flex justify-between text-xs mb-1.5">
                      <span className="text-slate-400">Stops</span>
                      <span className="text-slate-100 font-bold font-mono">{selectedNodeData.data.stops}</span>
                    </div>
                  </div>
                </div>
              )}

              {selectedNodeData.type === "risk" && (
                <div className="p-4 rounded-lg bg-slate-800/50 backdrop-blur-sm border border-slate-700/50">
                  <div className="text-xs font-semibold text-slate-400 mb-3 uppercase tracking-wider">
                    Risk Analysis
                  </div>
                  <div className="space-y-3">
                    <div className="flex justify-between text-xs mb-1.5">
                      <span className="text-slate-400">Severity</span>
                      <div className="flex items-center gap-2">
                        <div className="w-16 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-red-500 to-rose-400 shadow-lg shadow-red-500/50 transition-all duration-500"
                            style={{ width: `${selectedNodeData.data.severity * 100}%` }}
                          />
                        </div>
                        <span className="text-slate-100 font-bold font-mono text-xs">
                          {selectedNodeData.data.severity.toFixed(2)}
                        </span>
                      </div>
                    </div>
                    <div className="flex justify-between text-xs mb-1.5">
                      <span className="text-slate-400">Impact</span>
                      <span className="text-slate-100 font-bold font-mono text-xs">
                        +{(selectedNodeData.data.impact * 100).toFixed(1)}% time
                      </span>
                    </div>
                  </div>
                </div>
              )}

              <div>
                <div className="text-xs font-semibold text-slate-400 mb-3 uppercase tracking-wider">
                  Connected Nodes
                </div>
                <div className="space-y-2">
                  {selectedNodeData.connections.length > 0 ? (
                    selectedNodeData.connections.map((connId) => {
                      const connNode = nodes.find((n) => n.id === connId)
                      if (!connNode) return null
                      return (
                        <button
                          key={connId}
                          onClick={() => setSelectedNode(connId)}
                          className="w-full flex items-center justify-between p-2 rounded-md bg-slate-800 hover:bg-slate-700 transition-colors"
                        >
                          <div className="flex items-center gap-2">
                            <div
                              className={`h-2 w-2 rounded-full ${getNodeStyle(connNode, false, false).color.split(" ")[0]}`}
                            />
                            <span className="text-sm text-slate-200">{connNode.label}</span>
                          </div>
                          <Badge variant="outline" className="text-xs capitalize border-slate-600">
                            {connNode.type}
                          </Badge>
                        </button>
                      )
                    })
                  ) : (
                    <p className="text-xs text-slate-500 italic">No direct connections</p>
                  )}
                </div>
              </div>

              <div>
                <div className="text-xs font-semibold text-slate-400 mb-3 uppercase tracking-wider">Graph Position</div>
                <div className="p-3 rounded-md bg-slate-800 border border-slate-700">
                  <div className="grid grid-cols-3 gap-2 text-xs">
                    <div>
                      <div className="text-slate-500">X</div>
                      <div className="text-slate-100 font-mono">{selectedNodeData.x.toFixed(0)}</div>
                    </div>
                    <div>
                      <div className="text-slate-500">Y</div>
                      <div className="text-slate-100 font-mono">{selectedNodeData.y.toFixed(0)}</div>
                    </div>
                    <div>
                      <div className="text-slate-500">Z</div>
                      <div className="text-slate-100 font-mono">{selectedNodeData.z.toFixed(0)}</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
