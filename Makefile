# Samanvaya Makefile: Single-Command Automation

.PHONY: install run dev api test docker clean info help

PYTHON := python3
PIP := pip

help:
	@echo "Samanvaya: Lunar Optical Image Registration Framework"
	@echo "Usage:"
	@echo "  make dev      - ONE COMMAND: Start all 3 services (ML + Node.js + React)"
	@echo "  make install  - Install dependencies and register 'samanvaya' CLI"
	@echo "  make run      - Launch interactive Streamlit portal (port 8501)"
	@echo "  make api      - Launch FastAPI REST backend (port 8000)"
	@echo "  make test     - Run full automated verification suite"
	@echo "  make docker   - Build and start Docker container stack"
	@echo "  make clean    - Remove build artifacts and caches"
	@echo "  make info     - Display system environment and mission telemetry"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo "✅ Python core installed! Run 'make install-all' to also install Node.js deps."

install-all:
	@echo "📦 Installing all dependencies (Python + Node.js)..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	cd backend && npm install
	cd frontend && npm install
	@echo "✅ All dependencies installed! Run 'make dev' to start."

dev:
	@echo "🌙 Starting Samanvaya Full-Stack (ML + Gateway + React)..."
	@npm start

run:
	@echo "🚀 Launching Samanvaya Web Portal on http://localhost:8501 ..."
	streamlit run lunar_core/ui/app.py --server.port 8501

api:
	@echo "🛰️ Launching Samanvaya REST API on http://localhost:8000 ..."
	uvicorn ch2_lunar_reg.interfaces.api:app --host 0.0.0.0 --port 8000 --reload

test:
	@echo "🧪 Running full automated verification test suite..."
	pytest tests/ ch2_lunar_reg/tests/ -v

docker:
	@echo "🐳 Building and starting Docker container stack..."
	docker compose up --build

info:
	@$(PYTHON) -m lunar_core.cli info

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[cod]" -delete
