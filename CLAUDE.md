# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a FastAPI-based Python application using Python 3.13.5. The project includes:
- **FastAPI** (0.116.2) for the web framework
- **Celery** (5.5.3) for asynchronous task queue
- **pytest** (8.4.2) for testing
- **Alembic** for database migrations
- **Black**, **flake8**, and **mypy** for code quality

## Environment Setup

Activate the virtual environment:
```bash
source .venv/bin/activate
```

## Development Commands

### Code Quality
**CRITICAL**: Always run Black formatting after making code changes but BEFORE committing:
```bash
black .
```

Run linting:
```bash
flake8 .
```

Run type checking:
```bash
mypy .
```

### Testing
Run all tests:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov
```

Run a single test file:
```bash
pytest path/to/test_file.py
```

Run a specific test:
```bash
pytest path/to/test_file.py::test_function_name
```

### Database Migrations
Create a new migration:
```bash
alembic revision --autogenerate -m "description"
```

Apply migrations:
```bash
alembic upgrade head
```

Rollback migration:
```bash
alembic downgrade -1
```

### Celery Tasks
Start Celery worker:
```bash
celery -A <app_name> worker --loglevel=info
```

## Container Workflow

When making changes that need to be reflected in the frontend:
- Always rebuild the container to see those changes
