#!/usr/bin/env python3
"""FaceGuard-CNN Entrypoint Runner.

Run with:
    python run.py
or:
    python run.py --port 8080 --host 127.0.0.1
"""

import sys
import argparse
import uvicorn
from app.config import HOST, PORT, DEBUG, APP_TITLE, VERSION

def main():
    parser = argparse.ArgumentParser(description="Run FaceGuard-CNN FastAPI Server")
    parser.add_argument("--host", default=HOST, help=f"Host interface (default: {HOST})")
    parser.add_argument("--port", type=int, default=PORT, help=f"Port number (default: {PORT})")
    parser.add_argument("--reload", action="store_true", default=DEBUG, help="Enable auto-reload on code change")
    args = parser.parse_args()

    banner = f"""
    ===================================================================
      🛡️  {APP_TITLE} (v{VERSION})
    ===================================================================
      🚀 Dashboard UI   : http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}
      📖 Interactive API: http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}/docs
      ❤️  Health Check   : http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}/health
    ===================================================================
    """
    print(banner)

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()
