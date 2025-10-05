#!/usr/bin/env python3
"""
PRINT ALL DATA SYSTEM
Comprehensive data printing with detailed information for all components
"""

import os
import sys
import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# Add backend to path
sys.path.append("/Users/krishnabhatnagar/hackharvard/swarmaura/backend")

class AllDataPrinter:
    def __init__(self):
        self.data_dir = Path("unified_data")
        self.results_dir = Path("detailed_extraction_results")
        
        # Load unified data
        from unified_data_system import UnifiedDataSystem
        self.uds = UnifiedDataSystem()
        
        # Initialize all models
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize all available models"""
        print("🧠 INITIALIZING MODELS FOR DATA PRINTING")
        print("=" * 60)
        
        # Service Time GNN
        try:
            from backend.services.service_time_model import predictor_singleton
            self.service_predictor = predictor_singleton
            print(f"✅ Service Time GNN: {self.service_predictor.mode}")
        except Exception as e:
            print(f"❌ Service Time GNN: {e}")
            self.service_predictor = None
        
        # Risk Shaper GNN
        try:
            from backend.services.risk_shaper import risk_shaper_singleton
            self.risk_shaper = risk_shaper_singleton
            print(f"✅ Risk Shaper GNN: {'Loaded' if self.risk_shaper.model else 'Fallback'}")
        except Exception as e:
            print(f"❌ Risk Shaper GNN: {e}")
            self.risk_shaper = None
        
        # Warm-start Clusterer
        try:
            from backend.services.warmstart import warmstart_singleton
            self.warmstart = warmstart_singleton
            print(f"✅ Warm-start Clusterer: {'Loaded' if self.warmstart.model else 'Fallback'}")
        except Exception as e:
            print(f"❌ Warm-start Clusterer: {e}")
            self.warmstart = None
        
        # OR-Tools Solver
        try:
            from backend.services.ortools_solver import solve_vrp
            self.solver = solve_vrp
            print(f"✅ OR-Tools Solver: Available")
        except Exception as e:
            print(f"❌ OR-Tools Solver: {e}")
            self.solver = None
        
        print()
    
    def print_master_dataset(self):
        """Print complete master dataset"""
        print("📊 MASTER DATASET - COMPLETE DATA")
        print("=" * 80)
        
        master_data = self.uds.master_data
        
        print(f"📅 Dataset Generated: {master_data.get('timestamp', 'Unknown')}")
        print(f"📍 Total Locations: {len(master_data['locations'])}")
        print(f"🚛 Total Vehicles: {len(master_data['vehicles'])}")
        print(f"🌤️ Environmental Data: {'Available' if 'environmental' in master_data else 'Not Available'}")
        print()
        
        # Print all locations
        print("📍 ALL LOCATIONS:")
        print("-" * 50)
        for i, loc in enumerate(master_data['locations'], 1):
            print(f"{i}. {loc['name']} ({loc['type']})")
            print(f"   ID: {loc['id']}")
            print(f"   Coordinates: ({loc['lat']:.6f}, {loc['lng']:.6f})")
            print(f"   Demand: {loc.get('demand', 0)} units")
            print(f"   Priority: {loc.get('priority', 1)}")
            print(f"   Service Time: {loc.get('service_min', 5)} min")
            print(f"   Access Score: {loc.get('access_score', 0.5):.2f}")
            print(f"   Features: {', '.join(loc.get('features', [])) if loc.get('features', []) else 'None'}")
            print(f"   Hazards: {', '.join(loc.get('hazards', [])) if loc.get('hazards', []) else 'None'}")
            print(f"   Weather Risk: {loc.get('weather_risk', 0.2):.2f}")
            print(f"   Traffic Risk: {loc.get('traffic_risk', 0.3):.2f}")
            print(f"   Crime Risk: {loc.get('crime_risk', 0.1):.2f}")
            print(f"   Congestion Score: {loc.get('congestion_score', 0.4):.2f}")
            print(f"   Sidewalk Width: {loc.get('sidewalk_width', 2.0)}m")
            print(f"   Curb Cuts: {loc.get('curb_cuts', 2)}")
            print(f"   Parking Spaces: {loc.get('parking_spaces', 10)}")
            print(f"   Loading Docks: {loc.get('loading_docks', 1)}")
            print(f"   Lighting Score: {loc.get('lighting_score', 0.7):.2f}")
            print(f"   Historical Avg: {loc.get('historical_avg', 7.0):.1f} min")
            print(f"   Visit Frequency: {loc.get('visit_frequency', 1.0):.1f}")
            print(f"   Last Visit: {loc.get('last_visit', '2024-01-01')}")
            print()
        
        # Print all vehicles
        print("🚛 ALL VEHICLES:")
        print("-" * 50)
        for i, veh in enumerate(master_data['vehicles'], 1):
            print(f"{i}. {veh['id']} ({veh['type']})")
            print(f"   Capacity: {veh['capacity']} units")
            print(f"   Capabilities: {', '.join(veh['capabilities']) if veh['capabilities'] else 'Standard'}")
            print(f"   Max Weight: {veh.get('max_weight', 1000)} kg")
            print(f"   Max Volume: {veh.get('max_volume', 10)} m³")
            print(f"   Fuel Type: {veh.get('fuel_type', 'gasoline')}")
            print(f"   Year: {veh.get('year', 2020)}")
            print(f"   Avg Speed: {veh.get('avg_speed_kmph', 40)} km/h")
            print(f"   Fuel Efficiency: {veh.get('fuel_efficiency', 8.5)} L/100km")
            print(f"   Maintenance Cost: ${veh.get('maintenance_cost_per_km', 0.15):.2f}/km")
            print(f"   Driver Required: {veh.get('driver_required', True)}")
            print(f"   Max Daily Hours: {veh.get('max_daily_hours', 10)}")
            print(f"   Max Daily Distance: {veh.get('max_daily_distance', 500)} km")
            print(f"   Rest Required: {veh.get('rest_required', True)}")
            print(f"   Special Licenses: {', '.join(veh.get('special_licenses', [])) if veh.get('special_licenses', []) else 'None'}")
            print(f"   Hourly Rate: ${veh.get('hourly_rate', 25.0):.2f}")
            print(f"   Fuel Cost: ${veh.get('fuel_cost_per_liter', 1.5):.2f}/L")
            print(f"   Insurance Cost: ${veh.get('insurance_cost', 200.0):.2f}")
            print(f"   Depreciation Rate: {veh.get('depreciation_rate', 0.15):.1%}")
            print()
        
        print("✅ Master dataset printed completely\n")
    
    def print_environmental_data(self):
        """Print complete environmental data"""
        print("🌤️ ENVIRONMENTAL DATA - COMPLETE INFORMATION")
        print("=" * 80)
        
        env_data = self.uds.get_environmental_data()
        
        # Weather data
        weather = env_data["weather"]
        print("🌤️ WEATHER CONDITIONS:")
        print("-" * 30)
        print(f"Temperature: {weather['temperature']}°C")
        print(f"Condition: {weather['condition']}")
        print(f"Humidity: {weather['humidity']}%")
        print(f"Wind Speed: {weather['wind_speed']} km/h")
        print(f"Wind Direction: {weather.get('wind_direction', 'N/A')}")
        print(f"Visibility: {weather['visibility']} km")
        print(f"Precipitation: {weather['precipitation']} mm")
        print(f"Pressure: {weather.get('pressure', 1013.25)} hPa")
        print(f"UV Index: {weather.get('uv_index', 0)}")
        print()
        
        # Weather forecast
        print("📅 WEATHER FORECAST:")
        print("-" * 30)
        if 'next_3_hours' in weather:
            print("Next 3 Hours:")
            for hour, data in weather['next_3_hours'].items():
                print(f"  {hour}: {data}")
        if 'next_24_hours' in weather:
            print("Next 24 Hours:")
            for hour, data in weather['next_24_hours'].items():
                print(f"  {hour}: {data}")
        if 'weekly_outlook' in weather:
            print("Weekly Outlook:")
            for day, data in weather['weekly_outlook'].items():
                print(f"  {day}: {data}")
        print()
        
        # Weather alerts
        print("⚠️ WEATHER ALERTS:")
        print("-" * 30)
        print(f"Warnings: {', '.join(weather.get('weather_warnings', [])) if weather.get('weather_warnings', []) else 'None'}")
        print(f"Advisory Level: {weather.get('advisory_level', 'normal')}")
        print()
        
        # Traffic data
        traffic = env_data["traffic"]
        print("🚦 TRAFFIC CONDITIONS:")
        print("-" * 30)
        print(f"Overall Congestion: {traffic['overall_congestion']*100:.0f}%")
        print(f"Active Incidents: {traffic['incidents']}")
        print(f"Construction Zones: {traffic['construction_zones']}")
        print(f"Rush Hour Multiplier: {traffic['rush_hour_multiplier']:.1f}x")
        print(f"Average Speed: {traffic.get('average_speed', 30)} km/h")
        print()
        
        # Route-specific traffic
        print("🛣️ ROUTE-SPECIFIC TRAFFIC:")
        print("-" * 30)
        if 'main_arteries' in traffic:
            print("Main Arteries:")
            for artery, data in traffic['main_arteries'].items():
                print(f"  {artery}: {data}")
        if 'highway_conditions' in traffic:
            print("Highway Conditions:")
            for highway, data in traffic['highway_conditions'].items():
                print(f"  {highway}: {data}")
        if 'local_streets' in traffic:
            print("Local Streets:")
            for street, data in traffic['local_streets'].items():
                print(f"  {street}: {data}")
        print()
        
        # Traffic predictions
        print("🔮 TRAFFIC PREDICTIONS:")
        print("-" * 30)
        print(f"Peak Hours: {traffic.get('peak_hours', [7, 8, 17, 18])}")
        if 'expected_delays' in traffic:
            print("Expected Delays:")
            for route, delay in traffic['expected_delays'].items():
                print(f"  {route}: {delay}")
        if 'alternative_routes' in traffic:
            print("Alternative Routes:")
            for i, route in enumerate(traffic['alternative_routes'], 1):
                print(f"  {i}. {route}")
        print()
        
        # Time context
        time_context = env_data["time"]
        print("⏰ TIME CONTEXT:")
        print("-" * 30)
        print(f"Timestamp: {time_context.get('timestamp', datetime.now().isoformat())}")
        print(f"Current Hour: {time_context['current_hour']:02d}:00")
        print(f"Current Weekday: {time_context['current_weekday']} ({'Weekday' if time_context['current_weekday'] < 5 else 'Weekend'})")
        print(f"Is Holiday: {'Yes' if time_context['is_holiday'] else 'No'}")
        print(f"Season: {time_context['season']}")
        print()
        
        # Business hours
        print("🏢 BUSINESS HOURS:")
        print("-" * 30)
        print(f"Peak Delivery Hours: {time_context.get('peak_delivery_hours', [9, 10, 11, 14, 15, 16])}")
        print(f"Off Hours: {time_context.get('off_hours', [22, 23, 0, 1, 2, 3, 4, 5, 6])}")
        if 'weekend_patterns' in time_context:
            print("Weekend Patterns:")
            for pattern, data in time_context['weekend_patterns'].items():
                print(f"  {pattern}: {data}")
        print()
        
        print("✅ Environmental data printed completely\n")
    
    def print_ml_predictions(self):
        """Print complete ML predictions"""
        print("🧠 ML PREDICTIONS - COMPLETE ANALYSIS")
        print("=" * 80)
        
        # Service Time Predictions
        if self.service_predictor:
            print("⏱️ SERVICE TIME PREDICTIONS:")
            print("-" * 40)
            
            service_data = self.uds.get_service_time_data()
            predictions = self.service_predictor.predict_minutes(service_data)
            
            for i, (service, pred) in enumerate(zip(service_data, predictions), 1):
                loc = next(loc for loc in self.uds.master_data["locations"] if loc["id"] == service["id"])
                
                # Calculate detailed factors
                factors = {
                    "demand_impact": service["demand"] * 0.06,
                    "access_impact": 5.0 * (1.0 - service["access_score"]),
                    "weather_impact": service["weather_risk"] * 2.0,
                    "traffic_impact": service["traffic_risk"] * 1.5,
                    "peak_impact": (service["peak_hour"] - 1.0) * 2.0,
                    "historical_baseline": service["historical_avg"]
                }
                
                # Calculate confidence
                confidence = 1.0 - (abs(pred - service["historical_avg"]) / service["historical_avg"])
                
                print(f"{i}. {loc['name']} ({service['id']})")
                print(f"   Predicted Time: {pred:.1f} min")
                print(f"   Historical Average: {service['historical_avg']:.1f} min")
                print(f"   Confidence: {confidence:.2f}")
                print(f"   Model Type: {self.service_predictor.mode}")
                print(f"   Input Features:")
                print(f"     - Demand: {service['demand']} units")
                print(f"     - Access Score: {service['access_score']:.2f}")
                print(f"     - Weather Risk: {service['weather_risk']:.2f}")
                print(f"     - Traffic Risk: {service['traffic_risk']:.2f}")
                print(f"     - Peak Hour: {service['peak_hour']:.1f}")
                print(f"   Factor Analysis:")
                print(f"     - Demand Impact: {factors['demand_impact']:.1f} min")
                print(f"     - Access Impact: {factors['access_impact']:.1f} min")
                print(f"     - Weather Impact: {factors['weather_impact']:.1f} min")
                print(f"     - Traffic Impact: {factors['traffic_impact']:.1f} min")
                print(f"     - Peak Impact: {factors['peak_impact']:.1f} min")
                print(f"     - Historical Baseline: {factors['historical_baseline']:.1f} min")
                print()
            
            # Statistical summary
            avg_prediction = np.mean(predictions)
            avg_confidence = np.mean([1.0 - (abs(pred - service["historical_avg"]) / service["historical_avg"]) for pred, service in zip(predictions, service_data)])
            
            print("📊 SERVICE TIME STATISTICS:")
            print(f"   Average Prediction: {avg_prediction:.1f} min")
            print(f"   Average Confidence: {avg_confidence:.2f}")
            print(f"   Min Prediction: {min(predictions):.1f} min")
            print(f"   Max Prediction: {max(predictions):.1f} min")
            print(f"   Standard Deviation: {np.std(predictions):.1f} min")
            print()
        
        # Risk Assessment
        if self.risk_shaper:
            print("⚠️ RISK ASSESSMENT:")
            print("-" * 40)
            
            locations = self.uds.master_data["locations"]
            stops_order = [loc["id"] for loc in locations]
            
            # Create OSRM matrix
            n = len(locations)
            osrm_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j:
                        osrm_matrix[i][j] = np.random.uniform(5, 25)
            
            # Features
            features = {}
            for loc in locations:
                features[loc["id"]] = {
                    "risk": loc["crime_risk"],
                    "light": loc["lighting_score"],
                    "cong": loc["congestion_score"]
                }
            
            # Get risk multipliers
            multipliers = self.risk_shaper.shape(stops_order, osrm_matrix.tolist(), 14, 2, features)
            
            print("🚨 ALL ROUTE RISK ANALYSIS:")
            risk_data = []
            for i, src in enumerate(locations):
                for j, dst in enumerate(locations):
                    if i != j:
                        risk_mult = multipliers[i][j]
                        base_time = osrm_matrix[i][j]
                        adjusted_time = base_time * (1 + risk_mult)
                        
                        risk_info = {
                            "src": src["name"],
                            "dst": dst["name"],
                            "src_id": src["id"],
                            "dst_id": dst["id"],
                            "base_time": base_time,
                            "risk_multiplier": risk_mult,
                            "adjusted_time": adjusted_time,
                            "time_increase": adjusted_time - base_time,
                            "src_risk": features[src["id"]]["risk"],
                            "dst_risk": features[dst["id"]]["risk"],
                            "src_light": features[src["id"]]["light"],
                            "dst_light": features[dst["id"]]["light"]
                        }
                        risk_data.append(risk_info)
            
            # Sort by risk
            risk_data.sort(key=lambda x: x["risk_multiplier"], reverse=True)
            
            for i, risk in enumerate(risk_data, 1):
                print(f"{i:2d}. {risk['src']} → {risk['dst']}")
                print(f"     Risk Multiplier: {risk['risk_multiplier']:.3f}")
                print(f"     Base Time: {risk['base_time']:.1f} min")
                print(f"     Adjusted Time: {risk['adjusted_time']:.1f} min")
                print(f"     Time Increase: {risk['time_increase']:.1f} min")
                print(f"     Source Risk: {risk['src_risk']:.2f}")
                print(f"     Destination Risk: {risk['dst_risk']:.2f}")
                print(f"     Source Lighting: {risk['src_light']:.2f}")
                print(f"     Destination Lighting: {risk['dst_light']:.2f}")
                print()
            
            # Risk statistics
            avg_risk = np.mean([r["risk_multiplier"] for r in risk_data])
            max_risk = max([r["risk_multiplier"] for r in risk_data])
            high_risk_count = len([r for r in risk_data if r["risk_multiplier"] > 0.3])
            
            print("📊 RISK STATISTICS:")
            print(f"   Total Route Pairs: {len(risk_data)}")
            print(f"   Average Risk: {avg_risk:.3f}")
            print(f"   Maximum Risk: {max_risk:.3f}")
            print(f"   High Risk Routes (>0.3): {high_risk_count}")
            print(f"   Risk Distribution:")
            print(f"     - Low Risk (0.0-0.2): {len([r for r in risk_data if 0.0 <= r['risk_multiplier'] < 0.2])}")
            print(f"     - Medium Risk (0.2-0.3): {len([r for r in risk_data if 0.2 <= r['risk_multiplier'] < 0.3])}")
            print(f"     - High Risk (0.3-0.4): {len([r for r in risk_data if 0.3 <= r['risk_multiplier'] < 0.4])}")
            print(f"     - Very High Risk (0.4+): {len([r for r in risk_data if r['risk_multiplier'] >= 0.4])}")
            print()
        
        print("✅ ML predictions printed completely\n")
    
    def print_routing_data(self):
        """Print complete routing data"""
        print("🚛 ROUTING DATA - COMPLETE ANALYSIS")
        print("=" * 80)
        
        if not self.solver:
            print("❌ Solver not available")
            return
        
        # Get routing data
        routing_data = self.uds.get_routing_data()
        
        # Add service times
        service_data = self.uds.get_service_time_data()
        for stop in routing_data["stops"]:
            for service in service_data:
                if service["id"] == stop["id"]:
                    stop["service_min"] = service["historical_avg"]
                    break
        
        # Solve routing
        start_time = time.time()
        result = self.solver(
            depot=routing_data["depot"],
            stops=routing_data["stops"],
            vehicles=routing_data["vehicles"],
            time_limit_sec=10,
            drop_penalty_per_priority=2000,
            use_access_scores=True
        )
        solve_time = time.time() - start_time
        
        # Print routing information
        routes = result.get("routes", [])
        summary = result.get("summary", {})
        
        print("🚛 ROUTING SOLUTION:")
        print("-" * 40)
        print(f"Solve Time: {solve_time:.2f} seconds")
        print(f"Status: {result.get('status', 'Unknown')}")
        print(f"Total Distance: {summary.get('total_distance_km', 'N/A')} km")
        print(f"Total Time: {summary.get('total_time_min', 'N/A')} min")
        print(f"Served Stops: {summary.get('served_stops', 'N/A')}")
        print(f"Served Rate: {summary.get('served_rate', 'N/A')}")
        print()
        
        print("🚛 DETAILED ROUTES:")
        print("-" * 40)
        for i, route in enumerate(routes, 1):
            if route.get("stops", []):
                vehicle = routing_data["vehicles"][i-1] if i-1 < len(routing_data["vehicles"]) else None
                stops = route["stops"]
                
                print(f"Route {i} ({vehicle['id'] if vehicle else 'Unknown'} - {vehicle['type'] if vehicle else 'Unknown'}):")
                print(f"   Distance: {route.get('distance_km', 0):.2f} km")
                print(f"   Time: {route.get('time_min', 0):.1f} min")
                print(f"   Load: {route.get('load', 0)}/{vehicle['capacity'] if vehicle else 0} units")
                print(f"   Capacity Utilization: {(route.get('load', 0) / vehicle['capacity'] * 100) if vehicle and vehicle['capacity'] > 0 else 0:.1f}%")
                print(f"   Stops: {len(stops)}")
                print(f"   Vehicle Capabilities: {', '.join(vehicle['capabilities']) if vehicle and vehicle['capabilities'] else 'Standard'}")
                print()
                
                print(f"   📍 STOP SEQUENCE:")
                for j, stop in enumerate(stops):
                    stop_name = stop.get('name', stop.get('id', 'Unknown'))
                    if j == 0:
                        print(f"      {j+1}. {stop_name} (DEPOT)")
                    else:
                        service_time = stop.get('service_min', 'N/A')
                        access_score = stop.get('access_score', 'N/A')
                        demand = stop.get('demand', 'N/A')
                        priority = stop.get('priority', 'N/A')
                        print(f"      {j+1}. {stop_name}")
                        print(f"         - Service Time: {service_time} min")
                        print(f"         - Access Score: {access_score}")
                        print(f"         - Demand: {demand} units")
                        print(f"         - Priority: {priority}")
                        print(f"         - Coordinates: ({stop.get('lat', 0):.6f}, {stop.get('lng', 0):.6f})")
                print()
        
        # Overall performance
        total_distance = sum(r.get("distance_km", 0) for r in routes)
        total_time = sum(r.get("time_min", 0) for r in routes)
        total_load = sum(r.get("load", 0) for r in routes)
        total_capacity = sum(v["capacity"] for v in routing_data["vehicles"])
        
        print("📊 ROUTING PERFORMANCE SUMMARY:")
        print("-" * 40)
        print(f"Total Routes: {len(routes)}")
        print(f"Total Distance: {total_distance:.2f} km")
        print(f"Total Time: {total_time:.1f} min")
        print(f"Total Load: {total_load} units")
        print(f"Total Capacity: {total_capacity} units")
        print(f"Capacity Utilization: {(total_load / total_capacity * 100) if total_capacity > 0 else 0:.1f}%")
        print(f"Vehicle Efficiency: {len([r for r in routes if r.get('stops', [])]) / len(routing_data['vehicles']):.1%}")
        print(f"Average Route Length: {np.mean([len(r.get('stops', [])) for r in routes]):.1f} stops")
        print(f"Average Distance per Route: {total_distance / len(routes):.2f} km")
        print(f"Average Load per Route: {total_load / len(routes):.1f} units")
        print()
        
        print("✅ Routing data printed completely\n")
    
    def print_accessibility_data(self):
        """Print complete accessibility data"""
        print("♿ ACCESSIBILITY DATA - COMPLETE ANALYSIS")
        print("=" * 80)
        
        access_data = self.uds.get_accessibility_data()
        
        print("♿ ACCESSIBILITY ANALYSIS:")
        print("-" * 40)
        
        for i, loc in enumerate(access_data, 1):
            features = loc["features"]
            hazards = loc["hazards"]
            
            # Calculate detailed accessibility score
            base_score = loc["access_score"] * 100
            
            # Feature bonuses
            feature_bonus = 0
            if "elevator" in features:
                feature_bonus += 15
            if "ramp" in features:
                feature_bonus += 10
            if "wide_doors" in features:
                feature_bonus += 5
            
            # Sidewalk width bonus
            sidewalk_bonus = min(10, loc["sidewalk_width"] * 2)
            
            # Curb cuts bonus
            curb_bonus = min(10, loc["curb_cuts"] * 2)
            
            # Hazard penalties
            hazard_penalty = len(hazards) * 10
            
            # Final score
            final_score = base_score + feature_bonus + sidewalk_bonus + curb_bonus - hazard_penalty
            final_score = max(0, min(100, final_score))
            
            print(f"{i}. {loc['id']}")
            print(f"   Base Score: {base_score:.0f}/100")
            print(f"   Features: {', '.join(features) if features else 'None'}")
            print(f"   Feature Bonus: +{feature_bonus} points")
            print(f"   Sidewalk Width: {loc['sidewalk_width']}m (+{sidewalk_bonus:.1f} points)")
            print(f"   Curb Cuts: {loc['curb_cuts']} (+{curb_bonus} points)")
            print(f"   Hazards: {', '.join(hazards) if hazards else 'None'}")
            print(f"   Hazard Penalty: -{hazard_penalty} points")
            print(f"   Final Score: {final_score:.0f}/100")
            print(f"   Parking Spaces: {loc['parking_spaces']}")
            print(f"   Loading Docks: {loc['loading_docks']}")
            print(f"   Lighting Score: {loc['lighting_score']:.2f}")
            print()
        
        # Statistical analysis
        scores = []
        for loc in access_data:
            features = loc["features"]
            hazards = loc["hazards"]
            base_score = loc["access_score"] * 100
            feature_bonus = 0
            if "elevator" in features:
                feature_bonus += 15
            if "ramp" in features:
                feature_bonus += 10
            if "wide_doors" in features:
                feature_bonus += 5
            sidewalk_bonus = min(10, loc["sidewalk_width"] * 2)
            curb_bonus = min(10, loc["curb_cuts"] * 2)
            hazard_penalty = len(hazards) * 10
            final_score = base_score + feature_bonus + sidewalk_bonus + curb_bonus - hazard_penalty
            final_score = max(0, min(100, final_score))
            scores.append(final_score)
        
        avg_score = np.mean(scores)
        excellent_locations = len([s for s in scores if s >= 80])
        good_locations = len([s for s in scores if 60 <= s < 80])
        poor_locations = len([s for s in scores if s < 60])
        
        print("📊 ACCESSIBILITY STATISTICS:")
        print(f"   Average Score: {avg_score:.1f}/100")
        print(f"   Excellent Locations (≥80): {excellent_locations}")
        print(f"   Good Locations (60-79): {good_locations}")
        print(f"   Poor Locations (<60): {poor_locations}")
        print(f"   Total Features: {sum(len(loc['features']) for loc in access_data)}")
        print(f"   Total Hazards: {sum(len(loc['hazards']) for loc in access_data)}")
        print(f"   Average Sidewalk Width: {np.mean([loc['sidewalk_width'] for loc in access_data]):.2f}m")
        print(f"   Average Curb Cuts: {np.mean([loc['curb_cuts'] for loc in access_data]):.1f}")
        print(f"   Average Parking Spaces: {np.mean([loc['parking_spaces'] for loc in access_data]):.1f}")
        print(f"   Average Loading Docks: {np.mean([loc['loading_docks'] for loc in access_data]):.1f}")
        print(f"   Average Lighting Score: {np.mean([loc['lighting_score'] for loc in access_data]):.2f}")
        print()
        
        print("✅ Accessibility data printed completely\n")
    
    def print_all_data(self):
        """Print all data comprehensively"""
        print("🚀 PRINT ALL DATA SYSTEM")
        print("=" * 100)
        print("Comprehensive data printing with detailed information for all components")
        print()
        
        # Print all data sections
        self.print_master_dataset()
        self.print_environmental_data()
        self.print_ml_predictions()
        self.print_routing_data()
        self.print_accessibility_data()
        
        print("🎯 ALL DATA PRINTED COMPLETELY!")
        print("=" * 50)
        print("✅ Master dataset printed")
        print("✅ Environmental data printed")
        print("✅ ML predictions printed")
        print("✅ Routing data printed")
        print("✅ Accessibility data printed")
        print()
        print("🚀 All data extraction and printing complete!")

def main():
    """Main function"""
    print("🚀 PRINT ALL DATA SYSTEM INITIALIZATION")
    print("=" * 70)
    
    # Change to swarmaura directory
    os.chdir("/Users/krishnabhatnagar/hackharvard/swarmaura")
    
    # Initialize and print all data
    printer = AllDataPrinter()
    printer.print_all_data()
    
    print(f"🎉 All data printed successfully!")

if __name__ == "__main__":
    main()
