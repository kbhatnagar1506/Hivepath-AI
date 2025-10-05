"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { 
  X, 
  Home, 
  BarChart3, 
  Map, 
  Network, 
  Truck, 
  Route, 
  Brain, 
  DollarSign, 
  Database,
  Settings,
  HelpCircle,
  LogOut,
  ChevronRight,
  Zap,
  Activity,
  Users,
  Shield,
  Globe
} from "lucide-react"
import Link from "next/link"

interface SidebarMenuProps {
  isOpen: boolean
  onClose: () => void
}

export function SidebarMenu({ isOpen, onClose }: SidebarMenuProps) {
  const [activeSection, setActiveSection] = useState<string | null>(null)

  const menuSections = [
    {
      id: "dashboard",
      title: "Dashboard",
      icon: Home,
      href: "/",
      description: "Main control center"
    },
    {
      id: "analytics",
      title: "Analytics",
      icon: BarChart3,
      href: "/analytics",
      description: "Performance metrics & insights",
      badge: "Live"
    },
    {
      id: "infrastructure",
      title: "🏗️ Smart Infrastructure",
      icon: Map,
      href: "/infrastructure",
      description: "Interactive maps & routing"
    },
    {
      id: "knowledge",
      title: "🧠 AI Knowledge Graph",
      icon: Network,
      href: "/",
      description: "Neural network visualization"
    },
    {
      id: "logistics",
      title: "🚛 Smart Logistics Hub",
      icon: Truck,
      href: "/",
      description: "Fleet management center"
    },
    {
      id: "routes",
      title: "🛣️ Route Intelligence",
      icon: Route,
      href: "/",
      description: "Optimized delivery routes"
    },
    {
      id: "fleet",
      title: "🚚 Fleet Command Center",
      icon: Truck,
      href: "/",
      description: "Advanced fleet operations"
    },
    {
      id: "ai",
      title: "🤖 AI Insights Engine",
      icon: Brain,
      href: "/",
      description: "Machine learning predictions"
    },
    {
      id: "impact",
      title: "💰 Impact & ROI Analytics",
      icon: DollarSign,
      href: "/",
      description: "Cost savings & environmental impact"
    },
    {
      id: "data",
      title: "📊 System Data Hub",
      icon: Database,
      href: "/",
      description: "Comprehensive data analytics"
    }
  ]

  const systemInfo = [
    {
      icon: Activity,
      label: "System Status",
      value: "Online",
      status: "success"
    },
    {
      icon: Zap,
      label: "AI Processing",
      value: "Active",
      status: "success"
    },
    {
      icon: Users,
      label: "Active Users",
      value: "12",
      status: "info"
    },
    {
      icon: Shield,
      label: "Security",
      value: "Protected",
      status: "success"
    }
  ]

  const getStatusColor = (status: string) => {
    switch (status) {
      case "success": return "text-green-600 bg-green-100 dark:bg-green-900/20 dark:text-green-400"
      case "info": return "text-blue-600 bg-blue-100 dark:bg-blue-900/20 dark:text-blue-400"
      case "warning": return "text-yellow-600 bg-yellow-100 dark:bg-yellow-900/20 dark:text-yellow-400"
      default: return "text-slate-600 bg-slate-100 dark:bg-slate-800 dark:text-slate-400"
    }
  }

  return (
    <>
      {/* Backdrop */}
      <div 
        className={`fixed inset-0 bg-black/50 backdrop-blur-sm z-40 transition-all duration-500 ease-out ${
          isOpen ? 'opacity-100 visible' : 'opacity-0 invisible'
        }`}
        onClick={onClose}
      />

      {/* Sidebar */}
      <div className={`fixed left-0 top-0 h-full w-80 bg-white dark:bg-slate-900 border-r border-slate-200 dark:border-slate-700 shadow-2xl z-50 transform transition-all duration-500 ease-out ${
        isOpen ? 'translate-x-0 opacity-100' : '-translate-x-full opacity-0'
      }`}>
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-200 dark:border-slate-700">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-lg overflow-hidden bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center">
              <img 
                src="/logo.png" 
                alt="HivePath AI" 
                className="h-full w-full object-contain"
                onError={(e) => {
                  e.currentTarget.style.display = 'none';
                  e.currentTarget.nextElementSibling.style.display = 'flex';
                }}
              />
              <div className="h-full w-full bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center" style={{ display: 'none' }}>
                <span className="text-white font-bold text-lg">H</span>
              </div>
            </div>
            <div>
              <div className="font-bold text-slate-900 dark:text-slate-100">Infrastructure Platform</div>
              <div className="text-xs text-slate-600 dark:text-slate-400">Next-Gen AI Solutions</div>
            </div>
          </div>
          <Button variant="ghost" size="sm" onClick={onClose} className="h-8 w-8 p-0">
            <X className="h-4 w-4" />
          </Button>
        </div>

        {/* Navigation */}
        <div className="flex-1 overflow-y-auto">
          <div className="p-4">
            <div className="space-y-1">
              {menuSections.map((section, index) => {
                const Icon = section.icon
                return (
                  <Link key={section.id} href={section.href} onClick={onClose}>
                    <div 
                      className={`flex items-center gap-3 p-3 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 transition-all duration-300 ease-out group cursor-pointer transform ${
                        isOpen ? 'translate-x-0 opacity-100' : 'translate-x-4 opacity-0'
                      }`}
                      style={{
                        transitionDelay: isOpen ? `${index * 50}ms` : '0ms'
                      }}
                      onMouseEnter={() => setActiveSection(section.id)}
                      onMouseLeave={() => setActiveSection(null)}
                    >
                      <div className="flex-shrink-0">
                        <Icon className="h-5 w-5 text-slate-600 dark:text-slate-400 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-slate-900 dark:text-slate-100 text-sm">
                            {section.title}
                          </span>
                          {section.badge && (
                            <Badge variant="secondary" className="text-xs bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400">
                              {section.badge}
                            </Badge>
                          )}
                        </div>
                        <p className="text-xs text-slate-600 dark:text-slate-400 mt-0.5">
                          {section.description}
                        </p>
                      </div>
                      <ChevronRight className="h-4 w-4 text-slate-400 opacity-0 group-hover:opacity-100 transition-opacity" />
                    </div>
                  </Link>
                )
              })}
            </div>
          </div>

          {/* System Status */}
          <div className={`p-4 border-t border-slate-200 dark:border-slate-700 transition-all duration-500 ease-out transform ${
            isOpen ? 'translate-y-0 opacity-100' : 'translate-y-4 opacity-0'
          }`} style={{ transitionDelay: isOpen ? '400ms' : '0ms' }}>
            <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100 mb-3">System Status</h3>
            <div className="space-y-2">
              {systemInfo.map((info, index) => {
                const Icon = info.icon
                return (
                  <div 
                    key={index} 
                    className={`flex items-center gap-3 p-2 rounded-lg transition-all duration-300 ease-out transform hover:scale-105 hover:shadow-md ${
                      isOpen ? 'translate-x-0 opacity-100' : 'translate-x-4 opacity-0'
                    }`}
                    style={{ transitionDelay: isOpen ? `${500 + index * 100}ms` : '0ms' }}
                  >
                    <Icon className="h-4 w-4 text-slate-600 dark:text-slate-400 transition-colors duration-200" />
                    <div className="flex-1">
                      <div className="text-xs text-slate-600 dark:text-slate-400">{info.label}</div>
                    </div>
                    <Badge variant="secondary" className={`text-xs transition-all duration-200 ${getStatusColor(info.status)}`}>
                      {info.value}
                    </Badge>
                  </div>
                )
              })}
            </div>
          </div>

          {/* Quick Actions */}
          <div className={`p-4 border-t border-slate-200 dark:border-slate-700 transition-all duration-500 ease-out transform ${
            isOpen ? 'translate-y-0 opacity-100' : 'translate-y-4 opacity-0'
          }`} style={{ transitionDelay: isOpen ? '600ms' : '0ms' }}>
            <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100 mb-3">Quick Actions</h3>
            <div className="space-y-2">
              {[
                { icon: Settings, label: "Settings" },
                { icon: HelpCircle, label: "Help & Support" },
                { icon: Globe, label: "Documentation" }
              ].map((action, index) => {
                const Icon = action.icon
                return (
                  <Button 
                    key={index}
                    variant="outline" 
                    size="sm" 
                    className={`w-full justify-start gap-2 transition-all duration-300 ease-out transform hover:scale-105 hover:shadow-md ${
                      isOpen ? 'translate-x-0 opacity-100' : 'translate-x-4 opacity-0'
                    }`}
                    style={{ transitionDelay: isOpen ? `${700 + index * 100}ms` : '0ms' }}
                  >
                    <Icon className="h-4 w-4" />
                    {action.label}
                  </Button>
                )
              })}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className={`p-4 border-t border-slate-200 dark:border-slate-700 transition-all duration-500 ease-out transform ${
          isOpen ? 'translate-y-0 opacity-100' : 'translate-y-4 opacity-0'
        }`} style={{ transitionDelay: isOpen ? '800ms' : '0ms' }}>
          <div className={`flex items-center gap-3 p-3 rounded-lg bg-slate-50 dark:bg-slate-800 transition-all duration-300 ease-out transform hover:scale-105 hover:shadow-md ${
            isOpen ? 'translate-x-0 opacity-100' : 'translate-x-4 opacity-0'
          }`} style={{ transitionDelay: isOpen ? '900ms' : '0ms' }}>
            <div className="h-8 w-8 rounded-full overflow-hidden bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center transition-transform duration-200 hover:scale-110">
              <img 
                src="/logo.png" 
                alt="User Avatar" 
                className="h-full w-full object-cover"
                onError={(e) => {
                  e.currentTarget.style.display = 'none';
                  e.currentTarget.nextElementSibling.style.display = 'flex';
                }}
              />
              <div className="h-full w-full bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center" style={{ display: 'none' }}>
                <span className="text-white font-semibold text-sm">KB</span>
              </div>
            </div>
            <div className="flex-1">
              <div className="text-sm font-medium text-slate-900 dark:text-slate-100">Krishna Bhatnagar</div>
              <div className="text-xs text-slate-600 dark:text-slate-400">System Administrator</div>
            </div>
            <Button variant="ghost" size="sm" className="h-8 w-8 p-0 transition-all duration-200 hover:scale-110 hover:bg-red-100 dark:hover:bg-red-900/20">
              <LogOut className="h-4 w-4" />
            </Button>
          </div>
          <div className={`mt-3 text-center transition-all duration-300 ease-out transform ${
            isOpen ? 'translate-y-0 opacity-100' : 'translate-y-2 opacity-0'
          }`} style={{ transitionDelay: isOpen ? '1000ms' : '0ms' }}>
            <div className="text-xs text-slate-500 dark:text-slate-500">
              © 2024 Infrastructure Platform • v2.1.0
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
