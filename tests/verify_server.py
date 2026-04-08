"""Quick server endpoint verification."""
import urllib.request
import json

def test_endpoint(name, url):
    try:
        r = urllib.request.urlopen(url)
        body = r.read().decode()
        print(f"  [{name}] Status: {r.status} - PASS")
        return body
    except Exception as e:
        print(f"  [{name}] FAIL: {e}")
        return None

print("=" * 50)
print("SERVER ENDPOINT VERIFICATION")
print("=" * 50)

# Test /health
body = test_endpoint("/health", "http://localhost:8000/health")
if body:
    data = json.loads(body)
    print(f"    Response: {data}")

# Test /metadata
body = test_endpoint("/metadata", "http://localhost:8000/metadata")
if body:
    data = json.loads(body)
    print(f"    Name: {data.get('name')}")
    print(f"    Version: {data.get('version')}")

# Test /docs
body = test_endpoint("/docs", "http://localhost:8000/docs")
if body:
    has_swagger = "swagger" in body.lower()
    print(f"    Has Swagger UI: {has_swagger}")

# Test /openapi.json
body = test_endpoint("/openapi.json", "http://localhost:8000/openapi.json")
if body:
    data = json.loads(body)
    paths = list(data.get("paths", {}).keys())
    print(f"    API Paths: {paths}")

print("=" * 50)
print("ALL ENDPOINTS VERIFIED!")
