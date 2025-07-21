#!/bin/bash

# Deep Tree Echo WebVM-RWKV Integration Deployment Script

set -e

echo "🚀 Starting Deep Tree Echo WebVM-RWKV Integration deployment..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is required but not installed."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
cd src
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p logs cache temp sessions

# Set environment variables
export FLASK_APP=app.py
export FLASK_ENV=production
export PYTHONPATH=$(pwd)

# Run database migrations if needed
echo "🗄️ Setting up database..."
# Add database setup commands here when implemented

# Start the application
echo "🎯 Starting Deep Tree Echo WebVM-RWKV Integration..."
echo "🌐 Application will be available at: http://localhost:8000"
echo "📊 API documentation available at: http://localhost:8000/api/status"
echo ""
echo "Press Ctrl+C to stop the application"

python app.py

