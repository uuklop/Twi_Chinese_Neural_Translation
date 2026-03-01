#!/bin/bash
# Quick run script for local/remote development

echo "🚀 Starting Twi-Chinese Translator..."

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if we're in a remote server environment
if [ -n "$STREAMLIT_SERVER_ADDRESS" ]; then
    echo "📡 Using custom server address: $STREAMLIT_SERVER_ADDRESS"
else
    # Auto-detect if we should bind to all interfaces
    echo "🔍 Detecting network configuration..."
    
    # Get the primary IP address
    if command -v ip &> /dev/null; then
        IP_ADDR=$(ip route get 1.1.1.1 2>/dev/null | grep -oP 'src \K\S+' | head -1)
    elif command -v ifconfig &> /dev/null; then
        IP_ADDR=$(ifconfig | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1' | head -1)
    else
        IP_ADDR=$(hostname -I | awk '{print $1}')
    fi
    
    if [ -n "$IP_ADDR" ] && [ "$IP_ADDR" != "127.0.0.1" ]; then
        echo "🌐 Server IP detected: $IP_ADDR"
        echo "   The app will be accessible at: http://$IP_ADDR:8501"
        echo ""
    fi
fi

# Set environment variables for remote access
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLECORS=false
export STREAMLIT_SERVER_ENABLEXSRFPROTECTION=false
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run Streamlit
echo "🎯 Starting Streamlit server..."
streamlit run app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true
