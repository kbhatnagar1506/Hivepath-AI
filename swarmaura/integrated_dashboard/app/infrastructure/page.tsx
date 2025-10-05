"use client"

import { useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ArrowLeft, MapPin, Truck, Route, Play, Square, Navigation } from "lucide-react"
import Link from "next/link"

export default function InfrastructurePage() {
  useEffect(() => {
    // Load Google Maps API
    const script = document.createElement('script')
    script.src = `https://maps.googleapis.com/maps/api/js?key=${process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY}&callback=initMap`
    script.async = true
    script.defer = true
    document.head.appendChild(script)

    // Define the map initialization function globally
    ;(window as any).initMap = initMap

    return () => {
      document.head.removeChild(script)
    }
  }, [])

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
                <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">🏗️ Smart Infrastructure</h1>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                  Real-time truck tracking, route optimization, and fleet management
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <Badge variant="secondary" className="bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400">
                <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse mr-2"></div>
                Live Tracking
              </Badge>
            </div>
          </div>
        </div>
      </div>

      {/* Control Panel */}
      <div className="absolute top-20 left-4 z-10 bg-white dark:bg-slate-900 rounded-xl shadow-2xl border border-slate-200 dark:border-slate-700 p-4 w-80">
        <div className="flex items-center gap-2 mb-4">
          <MapPin className="h-5 w-5 text-blue-600" />
          <h3 className="font-semibold text-slate-900 dark:text-slate-100">Truck History & Routing</h3>
          <Badge variant="secondary" className="text-xs">Demo</Badge>
        </div>

        <div className="space-y-3">
          {/* Truck Selection */}
          <div className="flex items-center gap-3">
            <label htmlFor="truckSel" className="text-sm font-medium text-slate-700 dark:text-slate-300 w-16">
              Truck
            </label>
            <select 
              id="truckSel" 
              className="flex-1 px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 text-sm"
            >
              {/* Options will be populated by JavaScript */}
            </select>
          </div>

          {/* Control Buttons */}
          <div className="flex gap-2">
            <Button 
              id="btnFit" 
              variant="outline" 
              size="sm" 
              className="flex-1 gap-2"
              title="Zoom to history"
            >
              <MapPin className="h-4 w-4" />
              Fit History
            </Button>
            <Button 
              id="btnPlay" 
              size="sm" 
              className="flex-1 gap-2 bg-blue-600 hover:bg-blue-700"
            >
              <Play className="h-4 w-4" />
              Play
            </Button>
            <Button 
              id="btnStop" 
              variant="outline" 
              size="sm" 
              className="flex-1 gap-2"
              disabled
            >
              <Square className="h-4 w-4" />
              Stop
            </Button>
          </div>

          {/* Origin Selection */}
          <div className="flex items-center gap-3">
            <label htmlFor="originSel" className="text-sm font-medium text-slate-700 dark:text-slate-300 w-16">
              Origin
            </label>
            <select 
              id="originSel" 
              className="flex-1 px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 text-sm"
            >
              <option value="depot">Depot</option>
              <option value="me">My Location</option>
            </select>
          </div>

          {/* Route Buttons */}
          <div className="flex gap-2">
            <Button 
              id="btnRoute" 
              size="sm" 
              className="flex-1 gap-2 bg-green-600 hover:bg-green-700"
            >
              <Truck className="h-4 w-4" />
              Route to Truck
            </Button>
            <Button 
              id="btnClear" 
              variant="outline" 
              size="sm" 
              className="flex-1 gap-2"
            >
              <Route className="h-4 w-4" />
              Clear Routes
            </Button>
          </div>

          {/* Status */}
          <div id="status" className="text-xs text-slate-600 dark:text-slate-400 bg-slate-50 dark:bg-slate-800 p-2 rounded-lg"></div>
        </div>
      </div>

      {/* Map Container */}
      <div id="map" className="h-screen w-full"></div>

      {/* Load the map script */}
      <script
        dangerouslySetInnerHTML={{
          __html: `
            // ======== DEMO DATA (Boston-ish) ========
            const DEPOT = { name: "Depot", lat: 42.3601, lng: -71.0589 };

            // Each history point has {lat, lng, t} where t is ISO timestamp (for tooltips).
            const TRUCKS = [
              {
                id: "T1",
                color: "#EF4444",
                history: [
                  { lat: 42.3601, lng: -71.0589, t: "2025-10-05T10:00:00Z" },
                  { lat: 42.3655, lng: -71.0540, t: "2025-10-05T10:05:00Z" },
                  { lat: 42.3700, lng: -71.0500, t: "2025-10-05T10:10:00Z" },
                  { lat: 42.3731, lng: -71.0470, t: "2025-10-05T10:15:00Z" },
                  { lat: 42.3762, lng: -71.0445, t: "2025-10-05T10:20:00Z" },
                  { lat: 42.3790, lng: -71.0420, t: "2025-10-05T10:25:00Z" }
                ]
              },
              {
                id: "T2",
                color: "#3B82F6",
                history: [
                  { lat: 42.3601, lng: -71.0589, t: "2025-10-05T10:00:00Z" },
                  { lat: 42.3550, lng: -71.0650, t: "2025-10-05T10:06:00Z" },
                  { lat: 42.3490, lng: -71.0710, t: "2025-10-05T10:12:00Z" },
                  { lat: 42.3440, lng: -71.0800, t: "2025-10-05T10:17:00Z" },
                  { lat: 42.3400, lng: -71.0900, t: "2025-10-05T10:24:00Z" }
                ]
              }
            ];

            // ======== MAP / UI LOGIC ========
            let map, info, directionsService, directionsRenderers = [];
            const polylines = new Map();      // truckId -> google.maps.Polyline
            const markers = new Map();        // truckId -> google.maps.Marker (current position)
            const pointMarkers = new Map();   // truckId -> [markers for points]
            const animState = { timer: null, idx: 0, truckId: null };

            function $(id) { return document.getElementById(id); }
            function setStatus(msg) { $("status").textContent = msg || ""; }

            async function initMap() {
              // Create map
              map = new google.maps.Map(document.getElementById("map"), {
                center: { lat: DEPOT.lat, lng: DEPOT.lng },
                zoom: 12,
                mapId: "DEMO_MAP_ID" // optional custom map style if you have one
              });
              info = new google.maps.InfoWindow();
              directionsService = new google.maps.DirectionsService();

              // Populate truck selector
              const sel = $("truckSel");
              TRUCKS.forEach(t => {
                const opt = document.createElement("option");
                opt.value = t.id; opt.textContent = t.id;
                sel.appendChild(opt);
              });

              // Draw everything for the first truck by default
              drawAllTrucks();
              sel.addEventListener("change", () => focusTruck(sel.value));

              $("btnFit").onclick = () => fitHistory($("truckSel").value);
              $("btnPlay").onclick = playAnimation;
              $("btnStop").onclick = stopAnimation;
              $("btnRoute").onclick = () => routeToTruck($("truckSel").value, $("originSel").value);
              $("btnClear").onclick = clearRoutes;

              // Fit initial view
              fitAll();
            }

            function drawAllTrucks() {
              TRUCKS.forEach(t => drawTruck(t));
              $("truckSel").value = TRUCKS[0].id;
              focusTruck(TRUCKS[0].id);
            }

            function drawTruck(t) {
              // Polyline for history
              const path = t.history.map(p => ({ lat: p.lat, lng: p.lng }));
              const poly = new google.maps.Polyline({
                path,
                strokeColor: t.color,
                strokeOpacity: 0.9,
                strokeWeight: 4,
                icons: [{
                  icon: { path: google.maps.SymbolPath.FORWARD_CLOSED_ARROW, scale: 3 },
                  offset: "50%",
                  repeat: "120px"
                }]
              });
              poly.setMap(map);
              polylines.set(t.id, poly);

              // Point markers with timestamp tooltips
              const pts = [];
              t.history.forEach((p, idx) => {
                const m = new google.maps.Marker({
                  position: { lat: p.lat, lng: p.lng },
                  map,
                  title: \`\${t.id} @ \${new Date(p.t).toLocaleTimeString()}\`,
                  icon: {
                    path: google.maps.SymbolPath.CIRCLE,
                    scale: 4,
                    fillColor: "#111827",
                    fillOpacity: 1,
                    strokeColor: t.color,
                    strokeWeight: 2
                  }
                });
                m.addListener("click", () => {
                  info.setContent(\`<b>\${t.id}</b><br/>#\${idx+1}: \${new Date(p.t).toLocaleString()}\`);
                  info.open(map, m);
                });
                pts.push(m);
              });
              pointMarkers.set(t.id, pts);

              // Current position marker (last point)
              const last = t.history[t.history.length - 1];
              const truckIcon = {
                url: "data:image/svg+xml;utf-8," + encodeURIComponent(\`
                  <svg xmlns='http://www.w3.org/2000/svg' width='32' height='32' viewBox='0 0 24 24'>
                    <path fill='\${t.color}' d='M3 7h10v7h1.5a2.5 2.5 0 1 0 0 2H8.5a2.5 2.5 0 1 0 0 2H3V7m12 0h3l3 3v4h-2a2 2 0 0 0-2 2h-2V7Z'/>
                  </svg>\`),
                scaledSize: new google.maps.Size(32, 32),
                anchor: new google.maps.Point(16, 16)
              };
              const cur = new google.maps.Marker({
                position: { lat: last.lat, lng: last.lng },
                map,
                title: \`\${t.id} (current)\`,
                icon: truckIcon
              });
              markers.set(t.id, cur);
            }

            function fitAll() {
              const b = new google.maps.LatLngBounds();
              TRUCKS.forEach(t => t.history.forEach(p => b.extend({ lat: p.lat, lng: p.lng })));
              b.extend({ lat: DEPOT.lat, lng: DEPOT.lng });
              map.fitBounds(b);
            }

            function focusTruck(truckId) {
              const t = TRUCKS.find(x => x.id === truckId);
              if (!t) return;
              const b = new google.maps.LatLngBounds();
              t.history.forEach(p => b.extend({ lat: p.lat, lng: p.lng }));
              map.fitBounds(b);
              setStatus(\`Focused on \${truckId}\`);
            }

            function fitHistory(truckId) {
              focusTruck(truckId);
            }

            // ----- Animation along history -----
            function playAnimation() {
              const truckId = $("truckSel").value;
              const t = TRUCKS.find(x => x.id === truckId);
              if (!t) return;

              stopAnimation(); // reset
              animState.truckId = truckId;
              animState.idx = 0;
              $("btnPlay").disabled = true; $("btnStop").disabled = false;

              const cursor = new google.maps.Marker({
                position: t.history[0],
                map,
                icon: { path: google.maps.SymbolPath.BACKWARD_CLOSED_ARROW, scale: 5, strokeColor: "#111", strokeWeight: 2 },
                title: "Playback"
              });

              animState.timer = setInterval(() => {
                animState.idx++;
                if (animState.idx >= t.history.length) {
                  stopAnimation();
                  cursor.setMap(null);
                  setStatus("Playback finished.");
                  return;
                }
                const p = t.history[animState.idx];
                cursor.setPosition({ lat: p.lat, lng: p.lng });
                setStatus(\`Playing \${truckId} – point \${animState.idx+1}/\${t.history.length}\`);
              }, 700);
            }

            function stopAnimation() {
              if (animState.timer) clearInterval(animState.timer);
              animState.timer = null; animState.idx = 0; animState.truckId = null;
              $("btnPlay").disabled = false; $("btnStop").disabled = true;
            }

            // ----- Directions to a truck -----
            function clearRoutes() {
              directionsRenderers.forEach(r => r.setMap(null));
              directionsRenderers = [];
              setStatus("Routes cleared.");
            }

            function routeToTruck(truckId, originMode) {
              const t = TRUCKS.find(x => x.id === truckId);
              if (!t) return;
              const dest = t.history[t.history.length - 1];

              const requestRoute = (originLatLng) => {
                clearRoutes();
                directionsService.route({
                  origin: originLatLng,
                  destination: dest,
                  travelMode: google.maps.TravelMode.DRIVING,
                  provideRouteAlternatives: true
                }, (res, status) => {
                  if (status !== "OK" || !res) {
                    setStatus("Directions failed: " + status);
                    return;
                  }
                  res.routes.forEach((route, i) => {
                    const dr = new google.maps.DirectionsRenderer({
                      map,
                      directions: res,
                      routeIndex: i,
                      preserveViewport: true,
                      suppressMarkers: false,
                      polylineOptions: {
                        strokeColor: i === 0 ? "#10B981" : "#6B7280",
                        strokeOpacity: 0.9,
                        strokeWeight: 5
                      }
                    });
                    directionsRenderers.push(dr);
                  });
                  const km = (res.routes[0].legs[0].distance.value / 1000).toFixed(1);
                  const min = Math.round(res.routes[0].legs[0].duration.value / 60);
                  setStatus(\`Routed to \${truckId}: \${km} km, ~\${min} min (showing \${res.routes.length} option\${res.routes.length>1?'s':''}).\`);
                });
              };

              if (originMode === "me" && navigator.geolocation) {
                setStatus("Getting your location…");
                navigator.geolocation.getCurrentPosition(
                  pos => requestRoute({ lat: pos.coords.latitude, lng: pos.coords.longitude }),
                  _ => { setStatus("Location denied, using depot."); requestRoute(DEPOT); },
                  { enableHighAccuracy: true, timeout: 8000 }
                );
              } else {
                requestRoute(DEPOT);
              }
            }
          `
        }}
      />
    </div>
  )
}
