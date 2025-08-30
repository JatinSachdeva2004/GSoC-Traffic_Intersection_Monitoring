"""
Direct InfluxDB to Grafana connection tester with more verbose error reporting
"""
import requests
import json
import subprocess
import os
import sys
import time
from datetime import datetime, timedelta
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# Configuration
INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "kNFfXEpPQoWrk5Tteowda21Dzv6xD3jY7QHSHHQHb5oYW6VH6mkAgX9ZMjQJkaHHa8FwzmyVFqDG7qqzxN09uQ=="
INFLUX_ORG = "smart-intersection-org"
INFLUX_BUCKET = "traffic_monitoring"
GRAFANA_URL = "http://localhost:3000"
GRAFANA_USER = "admin"
GRAFANA_PASSWORD = "admin"

def check_service_status(service_name):
    """Check if a service is running"""
    try:
        # For Windows
        output = subprocess.check_output(f"tasklist /FI \"IMAGENAME eq {service_name}*\"", shell=True).decode()
        if service_name.lower() in output.lower():
            print(f"✅ {service_name} appears to be running")
            return True
        else:
            print(f"❌ {service_name} doesn't appear to be running")
            return False
    except Exception as e:
        print(f"❌ Error checking {service_name} status: {e}")
        return False

def test_influxdb_connection():
    """Test direct connection to InfluxDB"""
    print("\n===== TESTING INFLUXDB CONNECTION =====")
    try:
        client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        health = client.health()
        print(f"✅ InfluxDB connection successful")
        print(f"   Status: {health.status}")
        print(f"   Version: {health.version}")
        print(f"   Message: {health.message}")
        
        # Check if bucket exists
        buckets_api = client.buckets_api()
        buckets = buckets_api.find_buckets().buckets
        bucket_exists = False
        for bucket in buckets:
            if bucket.name == INFLUX_BUCKET:
                bucket_exists = True
                print(f"✅ Bucket '{INFLUX_BUCKET}' exists")
                break
        
        if not bucket_exists:
            print(f"❌ Bucket '{INFLUX_BUCKET}' does not exist")
            return False
        
        return True
    except Exception as e:
        print(f"❌ InfluxDB connection failed: {e}")
        return False

def check_influx_data():
    """Query InfluxDB directly to check if data exists"""
    print("\n===== CHECKING INFLUXDB DATA =====")
    try:
        client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        query_api = client.query_api()
        
        # Query for performance data
        query = f'''
        from(bucket: "{INFLUX_BUCKET}")
            |> range(start: -1h)
            |> filter(fn: (r) => r._measurement == "performance")
            |> count()
        '''
        
        result = query_api.query(query=query, org=INFLUX_ORG)
        
        if result and len(result) > 0:
            count = 0
            for table in result:
                for record in table.records:
                    count = record.get_value()
            
            print(f"✅ Found {count} performance data points in InfluxDB")
            
            # If data exists, print a sample to verify structure
            if count > 0:
                sample_query = f'''
                from(bucket: "{INFLUX_BUCKET}")
                    |> range(start: -1h)
                    |> filter(fn: (r) => r._measurement == "performance")
                    |> limit(n: 1)
                '''
                
                sample_result = query_api.query(query=sample_query, org=INFLUX_ORG)
                if sample_result and len(sample_result) > 0:
                    print("Sample data structure:")
                    for table in sample_result:
                        for record in table.records:
                            print(f"  _measurement: {record.get_measurement()}")
                            print(f"  _field: {record.get_field()}")
                            print(f"  _value: {record.get_value()}")
                            print(f"  _time: {record.get_time()}")
                            for key, value in record.values.items():
                                if key not in ['_measurement', '_field', '_value', '_time']:
                                    print(f"  {key}: {value}")
        else:
            print("❌ No data found in InfluxDB")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Error querying InfluxDB: {e}")
        return False

def check_grafana_datasource():
    """Check Grafana InfluxDB datasource configuration"""
    print("\n===== CHECKING GRAFANA DATASOURCE =====")
    try:
        response = requests.get(
            f"{GRAFANA_URL}/api/datasources",
            auth=(GRAFANA_USER, GRAFANA_PASSWORD)
        )
        
        if response.status_code != 200:
            print(f"❌ Failed to get Grafana datasources: {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        datasources = response.json()
        influx_ds = None
        
        for ds in datasources:
            if "influx" in ds["type"].lower():
                influx_ds = ds
                break
        
        if not influx_ds:
            print("❌ No InfluxDB datasource found in Grafana")
            return False
        
        print("InfluxDB datasource details:")
        print(f"  Name: {influx_ds.get('name')}")
        print(f"  Type: {influx_ds.get('type')}")
        print(f"  URL: {influx_ds.get('url')}")
        print(f"  Access: {influx_ds.get('access')}")
        
        # Check JSON data
        json_data = influx_ds.get('jsonData', {})
        print("  JSON Data:")
        print(f"    Version: {json_data.get('version')}")
        print(f"    Default Bucket: {json_data.get('defaultBucket')}")
        print(f"    Organization: {json_data.get('organization')}")
        print(f"    HTTP Mode: {json_data.get('httpMode')}")
        
        # Validate critical settings
        issues = []
        if json_data.get('version') != 'Flux':
            issues.append("⚠️ Version is not set to 'Flux'")
        
        if json_data.get('defaultBucket') != INFLUX_BUCKET:
            issues.append(f"⚠️ Default bucket ({json_data.get('defaultBucket')}) doesn't match expected ({INFLUX_BUCKET})")
        
        if json_data.get('organization') != INFLUX_ORG:
            issues.append(f"⚠️ Organization ({json_data.get('organization')}) doesn't match expected ({INFLUX_ORG})")
        
        if influx_ds.get('url') != INFLUX_URL:
            issues.append(f"⚠️ URL ({influx_ds.get('url')}) doesn't match expected ({INFLUX_URL})")
        
        if issues:
            print("\nPotential configuration issues:")
            for issue in issues:
                print(f"  {issue}")
            
            # Try to fix the datasource
            print("\nAttempting to fix data source configuration...")
            fixed_ds = {
                "id": influx_ds.get('id'),
                "name": influx_ds.get('name'),
                "type": "influxdb",
                "url": INFLUX_URL,
                "access": "proxy",
                "basicAuth": False,
                "isDefault": True,
                "jsonData": {
                    "defaultBucket": INFLUX_BUCKET,
                    "organization": INFLUX_ORG,
                    "version": "Flux",
                    "timeInterval": "10s"
                },
                "secureJsonData": {
                    "token": INFLUX_TOKEN
                }
            }
            
            # Update datasource
            response = requests.put(
                f"{GRAFANA_URL}/api/datasources/{influx_ds.get('id')}",
                auth=(GRAFANA_USER, GRAFANA_PASSWORD),
                headers={"Content-Type": "application/json"},
                data=json.dumps(fixed_ds)
            )
            
            if response.status_code == 200:
                print("✅ Datasource updated successfully")
            else:
                print(f"❌ Failed to update datasource: {response.status_code}")
                print(f"Response: {response.text}")
        else:
            print("✅ Datasource configuration appears correct")
        
        # Test datasource
        print("\nTesting datasource connection from Grafana...")
        response = requests.post(
            f"{GRAFANA_URL}/api/datasources/{influx_ds.get('id')}/health",
            auth=(GRAFANA_USER, GRAFANA_PASSWORD)
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Datasource health check: {result.get('status')} - {result.get('message')}")
        else:
            print(f"❌ Datasource health check failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Error checking Grafana datasource: {e}")
        return False

def write_timestamp_correction_data():
    """Write data with corrected timestamps"""
    print("\n===== WRITING TIMESTAMP CORRECTED DATA =====")
    try:
        client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        write_api = client.write_api(write_options=SYNCHRONOUS)
        
        # Current time
        now = datetime.utcnow()
        
        # Write data for the last 15 minutes with correct timestamps
        for i in range(15):
            # Calculate timestamp (going backwards from now to 15 minutes ago)
            timestamp = now - timedelta(minutes=i)
            
            # Create points
            perf_point = Point("performance") \
                .tag("camera_id", "timestamp_fixed") \
                .field("fps", 30.0 - i*0.5) \
                .field("processing_time_ms", 40.0 + i) \
                .time(timestamp)
            
            # Write the point
            write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=perf_point)
            print(f"Written timestamp-corrected data point {i+1}/15 at {timestamp}")
        
        print("✅ Wrote 15 data points with corrected timestamps")
        return True
    except Exception as e:
        print(f"❌ Error writing timestamp-corrected data: {e}")
        return False

def force_recreate_datasource():
    """Delete and recreate the InfluxDB datasource"""
    print("\n===== FORCE RECREATING INFLUXDB DATASOURCE =====")
    try:
        # Get existing datasources
        response = requests.get(
            f"{GRAFANA_URL}/api/datasources",
            auth=(GRAFANA_USER, GRAFANA_PASSWORD)
        )
        
        if response.status_code != 200:
            print(f"❌ Failed to get Grafana datasources: {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        datasources = response.json()
        influx_ds = None
        
        for ds in datasources:
            if "influx" in ds["type"].lower():
                influx_ds = ds
                break
        
        # Delete existing datasource if found
        if influx_ds:
            print(f"Deleting existing datasource: {influx_ds.get('name')}")
            response = requests.delete(
                f"{GRAFANA_URL}/api/datasources/{influx_ds.get('id')}",
                auth=(GRAFANA_USER, GRAFANA_PASSWORD)
            )
            
            if response.status_code != 200:
                print(f"❌ Failed to delete datasource: {response.status_code}")
                print(f"Response: {response.text}")
        
        # Create new datasource
        new_ds = {
            "name": "InfluxDB_Fixed",
            "type": "influxdb",
            "url": INFLUX_URL,
            "access": "proxy",
            "basicAuth": False,
            "isDefault": True,
            "jsonData": {
                "defaultBucket": INFLUX_BUCKET,
                "organization": INFLUX_ORG,
                "version": "Flux",
                "timeInterval": "10s"
            },
            "secureJsonData": {
                "token": INFLUX_TOKEN
            }
        }
        
        response = requests.post(
            f"{GRAFANA_URL}/api/datasources",
            auth=(GRAFANA_USER, GRAFANA_PASSWORD),
            headers={"Content-Type": "application/json"},
            data=json.dumps(new_ds)
        )
        
        if response.status_code == 200:
            print("✅ New datasource created successfully")
            return True
        else:
            print(f"❌ Failed to create new datasource: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error recreating datasource: {e}")
        return False

def create_minimal_dashboard():
    """Create an extremely minimal dashboard with basic queries"""
    print("\n===== CREATING MINIMAL TEST DASHBOARD =====")
    try:
        # Get data source ID
        response = requests.get(
            f"{GRAFANA_URL}/api/datasources",
            auth=(GRAFANA_USER, GRAFANA_PASSWORD)
        )
        
        if response.status_code != 200:
            print(f"❌ Failed to get datasources: {response.status_code}")
            return False
        
        datasources = response.json()
        influx_ds = None
        
        for ds in datasources:
            if "influx" in ds["type"].lower():
                influx_ds = ds
                break
        
        if not influx_ds:
            print("❌ No InfluxDB datasource found")
            return False
        
        # Create minimal dashboard
        dashboard = {
            "dashboard": {
                "id": None,
                "uid": "minimal-test",
                "title": "MINIMAL TEST - Timestamp Fixed",
                "tags": ["test"],
                "timezone": "browser",
                "refresh": "5s",
                "time": {
                    "from": "now-15m",
                    "to": "now"
                },
                "panels": [
                    {
                        "id": 1,
                        "title": "Minimal FPS Test",
                        "type": "timeseries",
                        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 0},
                        "datasource": {
                            "type": influx_ds["type"],
                            "uid": influx_ds["uid"]
                        },
                        "fieldConfig": {
                            "defaults": {
                                "custom": {
                                    "drawStyle": "line",
                                    "lineInterpolation": "linear",
                                    "fillOpacity": 10
                                }
                            }
                        },
                        "options": {
                            "tooltip": {
                                "mode": "single",
                                "sort": "none"
                            }
                        },
                        "targets": [
                            {
                                "refId": "A",
                                "datasource": {
                                    "type": influx_ds["type"],
                                    "uid": influx_ds["uid"]
                                },
                                "query": f'from(bucket: "{INFLUX_BUCKET}")\n  |> range(start: -15m)\n  |> filter(fn: (r) => r._measurement == "performance")\n  |> filter(fn: (r) => r._field == "fps")\n  |> aggregateWindow(every: 10s, fn: mean, createEmpty: false)',
                                "hide": False
                            }
                        ]
                    }
                ]
            },
            "overwrite": True,
            "message": "Created minimal test dashboard"
        }
        
        response = requests.post(
            f"{GRAFANA_URL}/api/dashboards/db",
            auth=(GRAFANA_USER, GRAFANA_PASSWORD),
            headers={"Content-Type": "application/json"},
            data=json.dumps(dashboard)
        )
        
        if response.status_code == 200:
            result = response.json()
            dashboard_url = f"{GRAFANA_URL}/d/{result['uid']}"
            print(f"✅ Minimal dashboard created: {dashboard_url}")
            print(f"👉 Please open this URL in your browser: {dashboard_url}")
            return True
        else:
            print(f"❌ Failed to create dashboard: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error creating minimal dashboard: {e}")
        return False

if __name__ == "__main__":
    print("===== ADVANCED GRAFANA-INFLUXDB DIAGNOSTICS AND REPAIR =====")
    
    # Check service status
    print("\n===== CHECKING SERVICE STATUS =====")
    influx_running = check_service_status("influxd")
    grafana_running = check_service_status("grafana")
    
    if not influx_running or not grafana_running:
        print("⚠️ One or more required services not running. Cannot proceed with diagnostics.")
        sys.exit(1)
    
    # Test InfluxDB connection and check data
    influx_connected = test_influxdb_connection()
    data_exists = check_influx_data()
    
    # Check Grafana datasource
    ds_ok = check_grafana_datasource()
    
    # If datasource has issues, force recreate it
    if not ds_ok:
        print("⚠️ Datasource has issues. Attempting to recreate it...")
        force_recreate_datasource()
    
    # Write timestamp-corrected data
    write_timestamp_correction_data()
    
    # Create minimal dashboard
    create_minimal_dashboard()
    
    print("\n===== SUMMARY =====")
    print(f"InfluxDB Running: {'✅' if influx_running else '❌'}")
    print(f"Grafana Running: {'✅' if grafana_running else '❌'}")
    print(f"InfluxDB Connection: {'✅' if influx_connected else '❌'}")
    print(f"Data Exists in InfluxDB: {'✅' if data_exists else '❌'}")
    print(f"Datasource Configuration: {'✅' if ds_ok else '🔄 Attempted repair'}")
    
    print("\nTroubleshooting Instructions:")
    print("1. Check the minimal test dashboard URL provided above")
    print("2. Ensure browser cache is cleared or try in private/incognito mode")
    print("3. If still no data, restart both Grafana and InfluxDB services")
    print("4. Check if your firewall is blocking localhost connections")
    print("5. Try changing the time range in the Grafana dashboard")
    print("6. Check your system time and timezone settings - InfluxDB is time-sensitive")
    print("7. Look for errors in the Grafana and InfluxDB log files")
    print("8. If needed, restart your computer and try again")
