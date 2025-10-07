#!/bin/bash

# FastAPI API Testing Script
echo "Testing FastAPI endpoints..."
echo

# Test 1: Health check endpoint
echo "=== Testing GET / (Health Check) ==="
curl -X GET http://localhost:8000/
echo -e "\n"

# Test 2: File upload endpoint
echo "=== Testing POST /uploadfile/ ==="
# Replace 'obama.txt' with your actual file name
curl -X POST 'http://localhost:8000/uploadfile/' -F 'file=@obama.txt'
echo -e "\n"

# Test 3: Ask question endpoint
# echo "=== Testing POST /ask/ ==="
# curl -X POST 'http://localhost:8000/ask/' \
#   -H 'Content-Type: application/json' \
#   -d '{"question": "What is FastAPI?"}'
# echo -e "\n"

echo "All tests completed!"