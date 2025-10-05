#!/usr/bin/env python3
"""
Cloudflare R2 Storage Setup for HivePath AI
Upload ML models, knowledge graph data, and analytics to R2
"""

import boto3
import os
import json
from pathlib import Path

class CloudflareR2Manager:
    def __init__(self, account_id, access_key_id, secret_access_key):
        self.s3_client = boto3.client(
            's3',
            endpoint_url='https://your-account-id.r2.cloudflarestorage.com',
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name='auto'
        )
        
    def upload_ml_models(self):
        """Upload trained ML models to R2"""
        models_dir = Path("../models")
        bucket_name = "hivepath-ai-models"
        
        models_to_upload = [
            "service_time.joblib",
            "warmstart_edge_clf.joblib"
        ]
        
        for model_file in models_to_upload:
            model_path = models_dir / model_file
            if model_path.exists():
                print(f"Uploading {model_file} to R2...")
                self.s3_client.upload_file(
                    str(model_path),
                    bucket_name,
                    f"models/{model_file}"
                )
                print(f"✅ {model_file} uploaded successfully")
            else:
                print(f"❌ {model_file} not found")
    
    def upload_knowledge_graph_data(self):
        """Upload knowledge graph data to R2"""
        data_dir = Path("../data")
        bucket_name = "hivepath-ai-knowledge-graph"
        
        kg_files = [
            "kg_nodes.csv",
            "kg_edges.csv",
            "edges_obs.csv"
        ]
        
        for kg_file in kg_files:
            kg_path = data_dir / kg_file
            if kg_path.exists():
                print(f"Uploading {kg_file} to R2...")
                self.s3_client.upload_file(
                    str(kg_path),
                    bucket_name,
                    f"knowledge-graph/{kg_file}"
                )
                print(f"✅ {kg_file} uploaded successfully")
            else:
                print(f"❌ {kg_file} not found")
    
    def upload_analytics_data(self):
        """Upload analytics and performance data to R2"""
        bucket_name = "hivepath-ai-analytics"
        
        # Create sample analytics data
        analytics_data = {
            "performance_metrics": {
                "cost_reduction": 0.30,
                "efficiency_gain": 0.40,
                "time_savings": 0.25,
                "accuracy": 0.85
            },
            "system_health": {
                "uptime": 0.999,
                "response_time": 120,
                "throughput": 1000
            },
            "timestamp": "2024-01-15T10:30:00Z"
        }
        
        print("Uploading analytics data to R2...")
        self.s3_client.put_object(
            Bucket=bucket_name,
            Key="analytics/performance.json",
            Body=json.dumps(analytics_data, indent=2),
            ContentType="application/json"
        )
        print("✅ Analytics data uploaded successfully")
    
    def setup_cors_policy(self):
        """Set up CORS policy for R2 buckets"""
        buckets = [
            "hivepath-ai-models",
            "hivepath-ai-knowledge-graph", 
            "hivepath-ai-analytics"
        ]
        
        cors_configuration = {
            'CORSRules': [
                {
                    'AllowedHeaders': ['*'],
                    'AllowedMethods': ['GET', 'PUT', 'POST', 'DELETE'],
                    'AllowedOrigins': ['*'],
                    'ExposeHeaders': ['ETag'],
                    'MaxAgeSeconds': 3000
                }
            ]
        }
        
        for bucket in buckets:
            try:
                self.s3_client.put_bucket_cors(
                    Bucket=bucket,
                    CORSConfiguration=cors_configuration
                )
                print(f"✅ CORS policy set for {bucket}")
            except Exception as e:
                print(f"❌ Failed to set CORS for {bucket}: {e}")

def main():
    # Configuration - Replace with your actual credentials
    ACCOUNT_ID = "your-cloudflare-account-id"
    ACCESS_KEY_ID = "your-r2-access-key-id"
    SECRET_ACCESS_KEY = "your-r2-secret-access-key"
    
    print("🚀 Setting up Cloudflare R2 Storage for HivePath AI")
    print("=" * 50)
    
    r2_manager = CloudflareR2Manager(ACCOUNT_ID, ACCESS_KEY_ID, SECRET_ACCESS_KEY)
    
    # Upload all data
    print("\n📦 Uploading ML Models...")
    r2_manager.upload_ml_models()
    
    print("\n🧠 Uploading Knowledge Graph Data...")
    r2_manager.upload_knowledge_graph_data()
    
    print("\n📊 Uploading Analytics Data...")
    r2_manager.upload_analytics_data()
    
    print("\n🌐 Setting up CORS policies...")
    r2_manager.setup_cors_policy()
    
    print("\n✅ Cloudflare R2 setup complete!")
    print("\n🎯 Your HivePath AI data is now stored in Cloudflare R2:")
    print("   • ML Models: hivepath-ai-models bucket")
    print("   • Knowledge Graph: hivepath-ai-knowledge-graph bucket")
    print("   • Analytics: hivepath-ai-analytics bucket")

if __name__ == "__main__":
    main()
