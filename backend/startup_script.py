# start_server.py - Simple startup script to avoid asyncio issues
import uvicorn
import config

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 AI Trading System v2.0 Starting...")
    print(f"🌐 FastAPI server: http://{config.WEB_HOST}:{config.WEB_PORT}")
    print(f"📊 Streamlit dashboard: http://localhost:{config.STREAMLIT_PORT}")
    print("=" * 60)
    
    try:
        uvicorn.run(
            "complete_realtime_main:app",
            host=config.WEB_HOST,
            port=config.WEB_PORT,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        print("Please check your configuration and try again")