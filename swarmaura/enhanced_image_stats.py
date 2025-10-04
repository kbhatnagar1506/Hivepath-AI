#!/usr/bin/env python3
"""
Enhanced Image Processing Statistics
"""

import cv2
import numpy as np
import requests
import time
import json
import hashlib
import os
from pathlib import Path
from collections import defaultdict
from backend.services.ortools_solver import solve_vrp

class EnhancedImageStats:
    """Enhanced statistics tracking for image processing"""
    
    def __init__(self):
        self.stats = {
            "processing": {
                "total_images": 0,
                "total_processing_time": 0.0,
                "avg_processing_time": 0.0,
                "min_processing_time": float('inf'),
                "max_processing_time": 0.0,
                "images_per_second": 0.0
            },
            "features": {
                "total_features_detected": 0,
                "feature_types": defaultdict(int),
                "avg_features_per_image": 0.0,
                "feature_detection_rate": 0.0
            },
            "accessibility": {
                "total_scores": 0,
                "avg_accessibility_score": 0.0,
                "min_accessibility_score": 100.0,
                "max_accessibility_score": 0.0,
                "score_distribution": {
                    "excellent": 0,  # 80-100
                    "good": 0,       # 60-79
                    "fair": 0,       # 40-59
                    "poor": 0        # 0-39
                }
            },
            "performance": {
                "opencv_operations": {
                    "edge_detection": {"count": 0, "total_time": 0.0, "avg_time": 0.0},
                    "line_detection": {"count": 0, "total_time": 0.0, "avg_time": 0.0},
                    "contour_analysis": {"count": 0, "total_time": 0.0, "avg_time": 0.0},
                    "color_analysis": {"count": 0, "total_time": 0.0, "avg_time": 0.0}
                },
                "cache_performance": {
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "hit_rate": 0.0,
                    "cache_savings_time": 0.0
                }
            },
            "quality": {
                "successful_analyses": 0,
                "failed_analyses": 0,
                "success_rate": 0.0,
                "image_quality_issues": 0
            },
            "geographic": {
                "locations_analyzed": 0,
                "unique_locations": set(),
                "headings_analyzed": defaultdict(int),
                "coverage_areas": []
            }
        }
        
        self.timing_data = []
        self.feature_history = []
        self.score_history = []
    
    def start_image_processing(self, location_id, heading):
        """Start timing an image processing operation"""
        return {
            "location_id": location_id,
            "heading": heading,
            "start_time": time.time(),
            "opencv_times": {},
            "features_detected": [],
            "accessibility_score": 0,
            "success": False
        }
    
    def end_image_processing(self, processing_data, result):
        """End timing and record statistics"""
        end_time = time.time()
        processing_time = end_time - processing_data["start_time"]
        
        # Update processing stats
        self.stats["processing"]["total_images"] += 1
        self.stats["processing"]["total_processing_time"] += processing_time
        self.stats["processing"]["min_processing_time"] = min(
            self.stats["processing"]["min_processing_time"], processing_time
        )
        self.stats["processing"]["max_processing_time"] = max(
            self.stats["processing"]["max_processing_time"], processing_time
        )
        
        # Update feature stats
        if result and "features" in result:
            features = result["features"]
            self.stats["features"]["total_features_detected"] += len(features)
            for feature in features:
                self.stats["features"]["feature_types"][feature] += 1
            processing_data["features_detected"] = features
        
        # Update accessibility stats
        if result and "accessibility_score" in result:
            score = result["accessibility_score"]
            self.stats["accessibility"]["total_scores"] += 1
            self.stats["accessibility"]["min_accessibility_score"] = min(
                self.stats["accessibility"]["min_accessibility_score"], score
            )
            self.stats["accessibility"]["max_accessibility_score"] = max(
                self.stats["accessibility"]["max_accessibility_score"], score
            )
            processing_data["accessibility_score"] = score
            
            # Update score distribution
            if score >= 80:
                self.stats["accessibility"]["score_distribution"]["excellent"] += 1
            elif score >= 60:
                self.stats["accessibility"]["score_distribution"]["good"] += 1
            elif score >= 40:
                self.stats["accessibility"]["score_distribution"]["fair"] += 1
            else:
                self.stats["accessibility"]["score_distribution"]["poor"] += 1
        
        # Update quality stats
        if result and "error" not in result:
            self.stats["quality"]["successful_analyses"] += 1
            processing_data["success"] = True
        else:
            self.stats["quality"]["failed_analyses"] += 1
        
        # Update geographic stats
        self.stats["geographic"]["unique_locations"].add(processing_data["location_id"])
        self.stats["geographic"]["headings_analyzed"][processing_data["heading"]] += 1
        
        # Store timing data
        self.timing_data.append({
            "location_id": processing_data["location_id"],
            "heading": processing_data["heading"],
            "processing_time": processing_time,
            "features_count": len(processing_data["features_detected"]),
            "accessibility_score": processing_data["accessibility_score"],
            "success": processing_data["success"]
        })
        
        # Store feature and score history
        self.feature_history.extend(processing_data["features_detected"])
        self.score_history.append(processing_data["accessibility_score"])
    
    def record_opencv_operation(self, operation_name, operation_time):
        """Record OpenCV operation timing"""
        if operation_name in self.stats["performance"]["opencv_operations"]:
            op_stats = self.stats["performance"]["opencv_operations"][operation_name]
            op_stats["count"] += 1
            op_stats["total_time"] += operation_time
            op_stats["avg_time"] = op_stats["total_time"] / op_stats["count"]
    
    def record_cache_performance(self, hit, time_saved=0.0):
        """Record cache performance"""
        if hit:
            self.stats["performance"]["cache_performance"]["cache_hits"] += 1
            self.stats["performance"]["cache_performance"]["cache_savings_time"] += time_saved
        else:
            self.stats["performance"]["cache_performance"]["cache_misses"] += 1
        
        total_requests = (self.stats["performance"]["cache_performance"]["cache_hits"] + 
                         self.stats["performance"]["cache_performance"]["cache_misses"])
        self.stats["performance"]["cache_performance"]["hit_rate"] = (
            self.stats["performance"]["cache_performance"]["cache_hits"] / total_requests
            if total_requests > 0 else 0.0
        )
    
    def calculate_derived_stats(self):
        """Calculate derived statistics"""
        # Processing stats
        if self.stats["processing"]["total_images"] > 0:
            self.stats["processing"]["avg_processing_time"] = (
                self.stats["processing"]["total_processing_time"] / 
                self.stats["processing"]["total_images"]
            )
            self.stats["processing"]["images_per_second"] = (
                self.stats["processing"]["total_images"] / 
                self.stats["processing"]["total_processing_time"]
            )
        
        # Feature stats
        if self.stats["processing"]["total_images"] > 0:
            self.stats["features"]["avg_features_per_image"] = (
                self.stats["features"]["total_features_detected"] / 
                self.stats["processing"]["total_images"]
            )
            self.stats["features"]["feature_detection_rate"] = (
                self.stats["features"]["total_features_detected"] / 
                (self.stats["processing"]["total_images"] * 7)  # Max possible features
            )
        
        # Accessibility stats
        if self.stats["accessibility"]["total_scores"] > 0:
            self.stats["accessibility"]["avg_accessibility_score"] = (
                sum(self.score_history) / len(self.score_history)
            )
        
        # Quality stats
        total_analyses = (self.stats["quality"]["successful_analyses"] + 
                         self.stats["quality"]["failed_analyses"])
        if total_analyses > 0:
            self.stats["quality"]["success_rate"] = (
                self.stats["quality"]["successful_analyses"] / total_analyses
            )
        
        # Geographic stats
        self.stats["geographic"]["locations_analyzed"] = len(self.stats["geographic"]["unique_locations"])
    
    def get_detailed_report(self):
        """Generate detailed statistics report"""
        self.calculate_derived_stats()
        
        report = {
            "summary": {
                "total_images_processed": self.stats["processing"]["total_images"],
                "total_processing_time": f"{self.stats['processing']['total_processing_time']:.3f}s",
                "avg_processing_time": f"{self.stats['processing']['avg_processing_time']:.3f}s",
                "images_per_second": f"{self.stats['processing']['images_per_second']:.2f}",
                "success_rate": f"{self.stats['quality']['success_rate']*100:.1f}%"
            },
            "performance_breakdown": {
                "processing_times": {
                    "min": f"{self.stats['processing']['min_processing_time']:.3f}s",
                    "max": f"{self.stats['processing']['max_processing_time']:.3f}s",
                    "avg": f"{self.stats['processing']['avg_processing_time']:.3f}s"
                },
                "opencv_operations": self.stats["performance"]["opencv_operations"],
                "cache_performance": {
                    "hit_rate": f"{self.stats['performance']['cache_performance']['hit_rate']*100:.1f}%",
                    "time_saved": f"{self.stats['performance']['cache_performance']['cache_savings_time']:.3f}s",
                    "hits": self.stats["performance"]["cache_performance"]["cache_hits"],
                    "misses": self.stats["performance"]["cache_performance"]["cache_misses"]
                }
            },
            "feature_analysis": {
                "total_features": self.stats["features"]["total_features_detected"],
                "avg_per_image": f"{self.stats['features']['avg_features_per_image']:.2f}",
                "detection_rate": f"{self.stats['features']['feature_detection_rate']*100:.1f}%",
                "feature_types": dict(self.stats["features"]["feature_types"])
            },
            "accessibility_analysis": {
                "score_stats": {
                    "min": f"{self.stats['accessibility']['min_accessibility_score']:.1f}",
                    "max": f"{self.stats['accessibility']['max_accessibility_score']:.1f}",
                    "avg": f"{self.stats['accessibility']['avg_accessibility_score']:.1f}"
                },
                "distribution": self.stats["accessibility"]["score_distribution"]
            },
            "geographic_coverage": {
                "unique_locations": self.stats["geographic"]["locations_analyzed"],
                "headings_analyzed": dict(self.stats["geographic"]["headings_analyzed"]),
                "coverage_per_location": f"{len(self.stats['geographic']['headings_analyzed'])/4*100:.1f}%"
            },
            "quality_metrics": {
                "successful_analyses": self.stats["quality"]["successful_analyses"],
                "failed_analyses": self.stats["quality"]["failed_analyses"],
                "success_rate": f"{self.stats['quality']['success_rate']*100:.1f}%"
            }
        }
        
        return report

def analyze_accessibility_with_enhanced_stats(image_data, stats, location_id, heading):
    """Enhanced accessibility analysis with detailed statistics"""
    
    processing_data = stats.start_image_processing(location_id, heading)
    
    try:
        # Convert bytes to OpenCV image
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            stats.end_image_processing(processing_data, {"error": "Could not decode image"})
            return {"error": "Could not decode image"}
        
        features = []
        confidence_scores = []
        analysis_details = {}
        
        # Convert to different color spaces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 1. Edge detection with timing
        edge_start = time.time()
        edges = cv2.Canny(gray, 50, 150)
        edge_time = time.time() - edge_start
        stats.record_opencv_operation("edge_detection", edge_time)
        
        # 2. Line detection with timing
        line_start = time.time()
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=5)
        line_time = time.time() - line_start
        stats.record_opencv_operation("line_detection", line_time)
        
        diagonal_lines = 0
        horizontal_lines = 0
        vertical_lines = 0
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                if 20 < abs(angle) < 70:
                    diagonal_lines += 1
                elif abs(angle) < 15 or abs(angle - 180) < 15:
                    horizontal_lines += 1
                elif 75 < abs(angle) < 105:
                    vertical_lines += 1
        
        analysis_details["diagonal_lines"] = diagonal_lines
        analysis_details["horizontal_lines"] = horizontal_lines
        analysis_details["vertical_lines"] = vertical_lines
        
        if diagonal_lines > 2:
            features.append("curb_cut")
            confidence_scores.append(min(0.9, diagonal_lines * 0.15))
        
        if horizontal_lines > 3:
            features.append("stairs")
            confidence_scores.append(min(0.8, horizontal_lines * 0.12))
        
        # 3. Contour analysis with timing
        contour_start = time.time()
        lower_white = np.array([200, 200, 200])
        upper_white = np.array([255, 255, 255])
        white_mask = cv2.inRange(image, lower_white, upper_white)
        
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        crosswalk_areas = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 300:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 2 < aspect_ratio < 15:
                    crosswalk_areas += 1
        
        contour_time = time.time() - contour_start
        stats.record_opencv_operation("contour_analysis", contour_time)
        
        analysis_details["crosswalk_areas"] = crosswalk_areas
        
        if crosswalk_areas > 1:
            features.append("crosswalk")
            confidence_scores.append(min(0.7, crosswalk_areas * 0.25))
        
        # 4. Color analysis with timing
        color_start = time.time()
        
        # Parking detection
        edges = cv2.Canny(gray, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        parking_rectangles = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 < area < 20000:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) == 4:
                    parking_rectangles += 1
        
        analysis_details["parking_rectangles"] = parking_rectangles
        
        if parking_rectangles > 0:
            features.append("parking")
            confidence_scores.append(min(0.6, parking_rectangles * 0.3))
        
        # Accessibility sign detection
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blue_rectangles = 0
        
        for contour in blue_contours:
            area = cv2.contourArea(contour)
            if area > 200:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) == 4:
                    blue_rectangles += 1
        
        analysis_details["blue_rectangles"] = blue_rectangles
        analysis_details["blue_ratio"] = cv2.countNonZero(blue_mask) / (image.shape[0] * image.shape[1])
        
        if blue_rectangles > 0 or analysis_details["blue_ratio"] > 0.03:
            features.append("accessibility_sign")
            confidence_scores.append(min(0.7, (blue_rectangles * 0.2 + analysis_details["blue_ratio"] * 5)))
        
        # Hazard detection
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        red_ratio = cv2.countNonZero(red_mask) / (image.shape[0] * image.shape[1])
        analysis_details["red_ratio"] = red_ratio
        
        if red_ratio > 0.02:
            features.append("hazard")
            confidence_scores.append(min(0.8, red_ratio * 20))
        
        if vertical_lines > 2:
            features.append("elevator")
            confidence_scores.append(min(0.6, vertical_lines * 0.2))
        
        color_time = time.time() - color_start
        stats.record_opencv_operation("color_analysis", color_time)
        
        # Generate description
        description_parts = []
        
        if "curb_cut" in features:
            description_parts.append("accessible ramp or curb cut")
        if "stairs" in features:
            description_parts.append("stairs present")
        if "crosswalk" in features:
            description_parts.append("marked crosswalk")
        if "parking" in features:
            description_parts.append("parking area")
        if "accessibility_sign" in features:
            description_parts.append("accessibility signage")
        if "elevator" in features:
            description_parts.append("elevator access")
        if "hazard" in features:
            description_parts.append("potential hazards")
        
        if not description_parts:
            description_parts.append("standard urban environment")
        
        description = f"Urban location with {', '.join(description_parts)}"
        
        # Calculate accessibility score
        positive_features = [f for f in features if f in ["curb_cut", "crosswalk", "parking", "accessibility_sign", "elevator"]]
        negative_features = [f for f in features if f in ["stairs", "hazard"]]
        
        base_score = 50
        positive_bonus = len(positive_features) * 15
        negative_penalty = len(negative_features) * 20
        
        accessibility_score = max(0, min(100, base_score + positive_bonus - negative_penalty))
        
        result = {
            "accessibility_score": accessibility_score,
            "features": features,
            "description": description,
            "confidence_scores": confidence_scores,
            "positive_features": positive_features,
            "negative_features": negative_features,
            "analysis_details": analysis_details
        }
        
        stats.end_image_processing(processing_data, result)
        return result
        
    except Exception as e:
        stats.end_image_processing(processing_data, {"error": str(e)})
        return {"error": str(e)}

def test_enhanced_image_stats():
    """Test enhanced image processing statistics"""
    
    print("📊 ENHANCED IMAGE PROCESSING STATISTICS TEST")
    print("=" * 60)
    print("Testing detailed performance and quality metrics...")
    print()
    
    # Initialize enhanced stats
    stats = EnhancedImageStats()
    
    # Test locations
    locations = [
        {
            "id": "back_bay",
            "lat": 42.3503,
            "lng": -71.0740,
            "name": "Back Bay Station"
        },
        {
            "id": "north_end",
            "lat": 42.3647,
            "lng": -71.0542,
            "name": "North End"
        },
        {
            "id": "cambridge",
            "lat": 42.3736,
            "lng": -71.1097,
            "name": "Harvard Square"
        }
    ]
    
    GOOGLE_MAPS_API_KEY = "AIzaSyAUeiyRuSuKcnPBFmezWCFuUStl8unv7_0"
    
    print(f"📊 ENHANCED STATS CONFIGURATION:")
    print(f"   • Detailed Timing: ✅")
    print(f"   • Feature Analysis: ✅")
    print(f"   • Performance Metrics: ✅")
    print(f"   • Quality Tracking: ✅")
    print(f"   • Geographic Coverage: ✅")
    print(f"   • Locations: {len(locations)}")
    print()
    
    # Process images with enhanced statistics
    print("🔍 Processing Images with Enhanced Statistics")
    print("-" * 50)
    
    for i, location in enumerate(locations, 1):
        print(f"   📍 Processing {i}/{len(locations)}: {location['name']}")
        
        headings = [0, 90, 180, 270]
        
        for heading in headings:
            try:
                # Get Street View image URL
                street_view_url = f"https://maps.googleapis.com/maps/api/streetview?location={location['lat']},{location['lng']}&heading={heading}&pitch=0&fov=90&size=640x640&key={GOOGLE_MAPS_API_KEY}"
                
                # Download image
                response = requests.get(street_view_url, timeout=5)
                if response.status_code != 200:
                    continue
                
                image_data = response.content
                
                # Analyze with enhanced statistics
                result = analyze_accessibility_with_enhanced_stats(
                    image_data, stats, location['id'], heading
                )
                
                if "error" not in result:
                    print(f"      📸 {heading}°: {len(result['features'])} features, score {result['accessibility_score']:.1f}")
                else:
                    print(f"      ❌ {heading}°: {result['error']}")
                
            except Exception as e:
                print(f"      ❌ {heading}°: {str(e)}")
    
    print()
    
    # Generate detailed report
    print("📊 ENHANCED STATISTICS REPORT")
    print("=" * 40)
    
    report = stats.get_detailed_report()
    
    # Summary
    print("📈 SUMMARY")
    print("-" * 15)
    summary = report["summary"]
    print(f"🔍 Total Images: {summary['total_images_processed']}")
    print(f"⏱️  Total Time: {summary['total_processing_time']}")
    print(f"⚡ Avg Time: {summary['avg_processing_time']}")
    print(f"🚀 Images/sec: {summary['images_per_second']}")
    print(f"✅ Success Rate: {summary['success_rate']}")
    print()
    
    # Performance breakdown
    print("⚡ PERFORMANCE BREAKDOWN")
    print("-" * 25)
    perf = report["performance_breakdown"]
    print(f"⏱️  Processing Times:")
    print(f"   Min: {perf['processing_times']['min']}")
    print(f"   Max: {perf['processing_times']['max']}")
    print(f"   Avg: {perf['processing_times']['avg']}")
    print()
    
    print("🔧 OpenCV Operations:")
    for op_name, op_stats in perf["opencv_operations"].items():
        print(f"   {op_name}: {op_stats['count']} ops, {op_stats['avg_time']:.3f}s avg")
    print()
    
    print("💾 Cache Performance:")
    cache_perf = perf["cache_performance"]
    print(f"   Hit Rate: {cache_perf['hit_rate']}")
    print(f"   Time Saved: {cache_perf['time_saved']}")
    print(f"   Hits: {cache_perf['hits']}, Misses: {cache_perf['misses']}")
    print()
    
    # Feature analysis
    print("🔍 FEATURE ANALYSIS")
    print("-" * 20)
    features = report["feature_analysis"]
    print(f"📊 Total Features: {features['total_features']}")
    print(f"📈 Avg per Image: {features['avg_per_image']}")
    print(f"🎯 Detection Rate: {features['detection_rate']}")
    print()
    
    print("🏷️  Feature Types:")
    for feature_type, count in features["feature_types"].items():
        print(f"   {feature_type}: {count}")
    print()
    
    # Accessibility analysis
    print("♿ ACCESSIBILITY ANALYSIS")
    print("-" * 25)
    accessibility = report["accessibility_analysis"]
    print(f"📊 Score Stats:")
    print(f"   Min: {accessibility['score_stats']['min']}")
    print(f"   Max: {accessibility['score_stats']['max']}")
    print(f"   Avg: {accessibility['score_stats']['avg']}")
    print()
    
    print("📈 Score Distribution:")
    dist = accessibility["distribution"]
    print(f"   Excellent (80-100): {dist['excellent']}")
    print(f"   Good (60-79): {dist['good']}")
    print(f"   Fair (40-59): {dist['fair']}")
    print(f"   Poor (0-39): {dist['poor']}")
    print()
    
    # Geographic coverage
    print("🌍 GEOGRAPHIC COVERAGE")
    print("-" * 22)
    geo = report["geographic_coverage"]
    print(f"📍 Unique Locations: {geo['unique_locations']}")
    print(f"🧭 Coverage per Location: {geo['coverage_per_location']}")
    print()
    
    print("🧭 Headings Analyzed:")
    for heading, count in geo["headings_analyzed"].items():
        print(f"   {heading}°: {count} images")
    print()
    
    # Quality metrics
    print("✅ QUALITY METRICS")
    print("-" * 18)
    quality = report["quality_metrics"]
    print(f"✅ Successful: {quality['successful_analyses']}")
    print(f"❌ Failed: {quality['failed_analyses']}")
    print(f"📊 Success Rate: {quality['success_rate']}")
    print()
    
    print("🎉 ENHANCED STATISTICS COMPLETE!")
    print("=" * 40)
    print("✅ Detailed performance tracking")
    print("✅ Comprehensive feature analysis")
    print("✅ Quality metrics monitoring")
    print("✅ Geographic coverage tracking")
    print("✅ Production-ready insights")

if __name__ == "__main__":
    test_enhanced_image_stats()
