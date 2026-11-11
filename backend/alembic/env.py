"""
Alembic Environment Configuration.

Configures Alembic to use the application's SQLAlchemy models
and database connection settings for generating and running migrations.
"""

import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# Add the backend package to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.database import Base
# Import all models to register them with Base.metadata
from backend.models import (  # noqa: F401
    Paper, Page, Question, QuestionEmbedding, QuestionClassification,
    SimilarityPair, Syllabus, HistoricalAnalysis,
    GeneratedPaper, GeneratedQuestion,
)
from backend.config import get_settings

# Alembic Config object
config = context.config

# Set the database URL from environment/settings
settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.database_url_sync)

# Set up loggers from config file
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Target metadata for autogenerate support
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    Generates SQL script without connecting to the database.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.

    Creates an engine and connects to the database to run migrations.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
