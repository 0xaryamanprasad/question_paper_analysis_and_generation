"""
FastAPI Application Entry Point.

Configures the FastAPI app, middleware, lifespan events,
and includes all API routers.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.database import init_db, close_db
from backend.utils.logger import setup_logging, get_logger
from backend.api.routes import papers, analysis, generation, syllabus

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events:
    - Startup: Initialize logging, connect to database
    - Shutdown: Close database connections
    """
    # --- Startup ---
    setup_logging()
    logger.info("application_starting", version=get_settings().app_version)

    # Initialize database tables gracefully
    try:
        await init_db()
        logger.info("database_initialized")
    except Exception as e:
        logger.warning(
            "database_init_warning",
            error=str(e),
            note="PostgreSQL is not reachable yet. Server will start; start database via 'docker compose up -d db'.",
        )

    yield

    # --- Shutdown ---
    try:
        await close_db()
    except Exception:
        pass
    logger.info("application_shutdown")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "AI-Powered Question Paper Analysis and Automated "
            "Question Paper Generation Using NLP and Generative AI"
        ),
        lifespan=lifespan,
    )

    # CORS middleware for standalone frontend, localhost, and file:// protocol
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check endpoint
    @app.get("/health", tags=["System"])
    async def health_check():
        """Basic health check endpoint."""
        return {
            "status": "healthy",
            "version": settings.app_version,
        }

    # Serve the interactive dashboard directly on root
    @app.get("/", tags=["Dashboard"])
    async def root_dashboard():
        """Serve the interactive React analytics & generation dashboard."""
        from fastapi.responses import HTMLResponse
        from pathlib import Path
        html_path = Path(__file__).resolve().parent.parent / "frontend" / "standalone_dashboard.html"
        if html_path.exists():
            return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>PaperMind: Question Paper Analysis & Generation</h1>")

    # Serve Brand Logo & Favicon
    from fastapi.responses import FileResponse
    from pathlib import Path

    logo_path = Path(__file__).resolve().parent / "static" / "papermind_logo.png"
    icon_path = Path(__file__).resolve().parent / "static" / "papermind_icon.png"

    @app.get("/logo.png", tags=["Branding"])
    @app.get("/papermind_logo.png", tags=["Branding"])
    async def get_brand_logo():
        """Serve the PaperMind logo."""
        if logo_path.exists():
            return FileResponse(logo_path, media_type="image/png")
        return {"error": "Logo not found"}

    @app.get("/icon.png", tags=["Branding"])
    @app.get("/favicon.ico", tags=["Branding"])
    @app.get("/papermind_icon.png", tags=["Branding"])
    async def get_brand_icon():
        """Serve the PaperMind icon / favicon."""
        if icon_path.exists():
            return FileResponse(icon_path, media_type="image/png")
        return {"error": "Icon not found"}

    # Mount static assets
    from fastapi.staticfiles import StaticFiles
    static_dir = Path(__file__).resolve().parent / "static"
    frontend_assets_dir = Path(__file__).resolve().parent.parent / "frontend" / "assets"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    if frontend_assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(frontend_assets_dir)), name="assets")

    # Include API routers
    app.include_router(papers.router, prefix="/api/v1")
    app.include_router(analysis.router, prefix="/api/v1")
    app.include_router(generation.router, prefix="/api/v1")
    app.include_router(syllabus.router, prefix="/api/v1")

    logger.info("application_created", debug=settings.debug)
    return app


# Application instance used by uvicorn
app = create_app()
