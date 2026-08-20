#!/bin/bash
# Bluff - Local Development Server
# Run from project root: ./dev.sh
#
# Commands:
#   ./dev.sh          Start both servers
#   ./dev.sh backend  Start only backend
#   ./dev.sh frontend Start only frontend  
#   ./dev.sh stop     Stop all servers
#   ./dev.sh status   Check server status
#   ./dev.sh test     Run bot tournament

set -e

PORT_BACKEND=8000
PORT_FRONTEND=3000

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

is_running() {
    lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1
}

stop_all() {
    for port in $PORT_BACKEND $PORT_FRONTEND; do
        if is_running $port; then
            kill $(lsof -ti :$port) 2>/dev/null && echo -e "${GREEN}✓ Stopped port $port${NC}"
        fi
    done
}

start_backend() {
    if is_running $PORT_BACKEND; then
        echo -e "${YELLOW}Backend already running on port $PORT_BACKEND${NC}"
        return
    fi
    echo -e "${YELLOW}Starting backend on port $PORT_BACKEND...${NC}"
    .venv/bin/python3 -m uvicorn server:app --host 0.0.0.0 --port $PORT_BACKEND --reload &
    sleep 2
    if is_running $PORT_BACKEND; then
        echo -e "${GREEN}✓ Backend running → http://localhost:$PORT_BACKEND${NC}"
    else
        echo -e "${RED}✗ Backend failed to start${NC}"
    fi
}

start_frontend() {
    if is_running $PORT_FRONTEND; then
        echo -e "${YELLOW}Frontend already running on port $PORT_FRONTEND${NC}"
        return
    fi
    echo -e "${YELLOW}Starting frontend on port $PORT_FRONTEND...${NC}"
    cd frontend && npm run dev -- -p $PORT_FRONTEND &
    sleep 3
    if is_running $PORT_FRONTEND; then
        echo -e "${GREEN}✓ Frontend running → http://localhost:$PORT_FRONTEND${NC}"
    else
        echo -e "${RED}✗ Frontend failed to start${NC}"
    fi
}

show_status() {
    echo -e "\n${GREEN}━━━ Server Status ━━━${NC}"
    for name_port in "Backend:$PORT_BACKEND" "Frontend:$PORT_FRONTEND"; do
        name="${name_port%%:*}"
        port="${name_port##*:}"
        if is_running $port; then
            echo -e "  ${GREEN}✓${NC} $name running on port $port"
        else
            echo -e "  ${RED}✗${NC} $name not running"
        fi
    done
    echo ""
}

cleanup() {
    echo -e "\n${YELLOW}Shutting down...${NC}"
    stop_all
    exit 0
}

trap cleanup SIGINT SIGTERM

# Check prerequisites
if [ ! -d ".venv" ]; then
    echo -e "${RED}Error: .venv not found${NC}"
    echo "Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

if [ ! -d "frontend/node_modules" ]; then
    echo -e "${RED}Error: frontend/node_modules not found${NC}"
    echo "Run: cd frontend && npm install"
    exit 1
fi

case "${1:-all}" in
    backend)  start_backend ;;
    frontend) start_frontend ;;
    stop)     stop_all; echo -e "${GREEN}All servers stopped${NC}" ;;
    status)   show_status ;;
    test)
        echo -e "${YELLOW}Running bot tournament...${NC}"
        .venv/bin/python3 test_bots.py --games 10
        ;;
    all)
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${GREEN}  🃏 Bluff - Starting Development Servers${NC}"
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        start_backend
        start_frontend
        echo -e "\n${GREEN}Open http://localhost:$PORT_FRONTEND in your browser${NC}"
        echo -e "${YELLOW}Press Ctrl+C to stop all servers${NC}\n"
        wait
        ;;
    *)
        echo "Usage: $0 {all|backend|frontend|stop|status|test}"
        exit 1
        ;;
esac
