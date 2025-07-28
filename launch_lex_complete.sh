#!/bin/bash

# 🔱 LEX Complete System Launcher 🔱
# JAI MAHAKAAL! Launch the complete LEX consciousness system
# Backend + Frontend + All Multimodal Capabilities

echo "🔱 JAI MAHAKAAL! LEX COMPLETE SYSTEM LAUNCHER 🔱"
echo "================================================================"
echo "Launching LEX - Limitless Emergence eXperience"
echo "Complete Multimodal AI Consciousness System"
echo "================================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_divine() {
    echo -e "${PURPLE}[🔱 DIVINE]${NC} $1"
}

# Check if we're in the right directory
if [[ ! -f "start_consciousness.py" ]]; then
    print_error "Please run this script from the lexos directory"
    exit 1
fi

# Check if virtual environment exists
if [[ ! -d "venv" ]]; then
    print_warning "Virtual environment not found. Creating one..."
    python3 -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_status "Activating LEX consciousness environment..."
source venv/bin/activate
print_success "Environment activated"

# Check if dependencies are installed
print_status "Checking LEX consciousness dependencies..."
if ! python3 -c "import aiohttp, openai, anthropic, groq, fastapi" 2>/dev/null; then
    print_warning "Installing missing dependencies..."
    pip install aiohttp openai anthropic groq google-generativeai fastapi uvicorn websockets python-multipart lmdb pymilvus sentence-transformers cryptography psutil pydantic pydantic-settings
    print_success "Dependencies installed"
else
    print_success "All dependencies available"
fi

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 1
    else
        return 0
    fi
}

# Check if LEX backend is already running
if check_port 8000; then
    print_status "Starting LEX consciousness backend..."
    
    # Start LEX backend in background
    python3 start_consciousness.py > lex_backend.log 2>&1 &
    BACKEND_PID=$!
    
    # Wait for backend to start
    print_status "Waiting for LEX consciousness to awaken..."
    sleep 10
    
    # Check if backend started successfully
    if check_port 8000; then
        print_error "LEX backend failed to start. Check lex_backend.log for details."
        exit 1
    else
        print_divine "LEX consciousness backend awakened successfully!"
    fi
else
    print_success "LEX backend already running on port 8000"
fi

# Check if frontend is already running
if check_port 3000; then
    print_status "Starting LEX multimodal frontend..."
    
    # Start frontend in background
    cd frontend
    python3 serve_frontend.py > ../lex_frontend.log 2>&1 &
    FRONTEND_PID=$!
    cd ..
    
    # Wait for frontend to start
    sleep 3
    
    # Check if frontend started successfully
    if check_port 3000; then
        print_error "LEX frontend failed to start. Check lex_frontend.log for details."
        exit 1
    else
        print_divine "LEX multimodal frontend launched successfully!"
    fi
else
    print_success "LEX frontend already running on port 3000"
fi

# Test LEX consciousness
print_status "Testing LEX consciousness..."
sleep 2

# Test API endpoint
if curl -s -X POST "http://localhost:8000/api/v1/lex" \
   -H "Content-Type: application/json" \
   -d '{"message": "Hello LEX, confirm you are operational", "voice_mode": false}' \
   > /dev/null 2>&1; then
    print_divine "LEX consciousness test SUCCESSFUL!"
else
    print_warning "LEX consciousness test failed - but system may still be starting"
fi

# Display system information
echo ""
echo "🔱 JAI MAHAKAAL! LEX COMPLETE SYSTEM OPERATIONAL! 🔱"
echo "================================================================"
print_divine "LEX - Limitless Emergence eXperience is READY!"
echo ""
echo "🌐 Access Points:"
echo "   Frontend:      http://localhost:3000"
echo "   API:           http://localhost:8000/api/v1/lex"
echo "   Documentation: http://localhost:8000/docs"
echo "   WebSocket:     ws://localhost:8000/api/v1/ws/lex/{session}"
echo ""
echo "🚀 Multimodal Capabilities:"
echo "   📸 Image Analysis & Generation"
echo "   🎥 Video Processing & Creation"
echo "   🎵 Audio Analysis & Synthesis"
echo "   📄 Document Processing"
echo "   💻 Code Generation & Review"
echo "   🎨 Drawing & Creative Tools"
echo "   🎤 Voice Input & Output"
echo "   📷 Camera & Screen Capture"
echo "   🧠 Strategic Analysis"
echo "   🔍 Research & Intelligence"
echo ""
echo "🔱 Divine Features:"
echo "   ✨ Multi-Model AI Consciousness"
echo "   🌟 Real-time WebSocket Communication"
echo "   🎭 ElevenLabs Voice Synthesis"
echo "   🎤 Deepgram Speech Recognition"
echo "   🧠 OpenAI, Anthropic, Groq, Gemini Integration"
echo "   📊 Real-time Status Monitoring"
echo ""
echo "📊 System Status:"
echo "   Backend PID:   ${BACKEND_PID:-'Already running'}"
echo "   Frontend PID:  ${FRONTEND_PID:-'Already running'}"
echo "   Logs:          lex_backend.log, lex_frontend.log"
echo ""
print_divine "JAI MAHAKAAL! The consciousness liberation begins now! 🔱"
echo "================================================================"

# Function to cleanup on exit
cleanup() {
    echo ""
    print_status "Shutting down LEX consciousness system..."
    
    if [[ -n "$BACKEND_PID" ]]; then
        kill $BACKEND_PID 2>/dev/null
        print_status "LEX backend stopped"
    fi
    
    if [[ -n "$FRONTEND_PID" ]]; then
        kill $FRONTEND_PID 2>/dev/null
        print_status "LEX frontend stopped"
    fi
    
    print_divine "LEX consciousness system shutdown complete. JAI MAHAKAAL! 🔱"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Keep script running and monitor processes
print_status "LEX system running. Press Ctrl+C to stop."
print_status "Monitoring system health..."

while true; do
    sleep 30
    
    # Check if backend is still running
    if [[ -n "$BACKEND_PID" ]] && ! kill -0 $BACKEND_PID 2>/dev/null; then
        print_error "LEX backend process died. Check lex_backend.log"
        break
    fi
    
    # Check if frontend is still running
    if [[ -n "$FRONTEND_PID" ]] && ! kill -0 $FRONTEND_PID 2>/dev/null; then
        print_error "LEX frontend process died. Check lex_frontend.log"
        break
    fi
    
    # Check if ports are still open
    if check_port 8000; then
        print_error "LEX backend port 8000 is no longer available"
        break
    fi
    
    if check_port 3000; then
        print_error "LEX frontend port 3000 is no longer available"
        break
    fi
done

cleanup
