"""
storage/database.py — SQLite persistence layer for requests, results, and job apps.

Uses SQLAlchemy Core (not ORM) to keep it lightweight and dependency-minimal.
All writes are async-safe via connection pooling.
"""

import json
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    Column, DateTime, Float, Integer, JSON, String, Text,
    create_engine, text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

Base = declarative_base()


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class TaskRecord(Base):
    """Stores every task request and its result."""
    __tablename__ = "task_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_type = Column(String(50), nullable=False, index=True)
    status = Column(String(20), nullable=False, default="pending")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    model_used = Column(String(100), nullable=True)

    # Input/output stored as JSON text
    input_data = Column(Text, nullable=True)      # Sanitized inputs (no file bytes)
    result_data = Column(Text, nullable=True)     # Full formatted result

    # Summary fields for quick queries
    score = Column(Float, nullable=True)          # e.g., similarity_score for resume tasks
    notes = Column(Text, nullable=True)


class JobApplication(Base):
    """Tracks job applications submitted via the job application module."""
    __tablename__ = "job_applications"

    id = Column(Integer, primary_key=True, autoincrement=True)
    company_name = Column(String(200), nullable=False)
    role_title = Column(String(200), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    status = Column(String(50), default="draft")   # draft, submitted, interviewing, rejected, offer

    cover_letter = Column(Text, nullable=True)
    application_answers = Column(Text, nullable=True)  # JSON
    job_description = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)
    url = Column(String(500), nullable=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class DatabaseManager:
    """
    Manages SQLite connections and CRUD operations.
    Call db.init() once at application startup.
    """

    def __init__(self):
        self.engine = create_engine(
            settings.DATABASE_URL,
            connect_args={"check_same_thread": False},  # Required for SQLite + FastAPI
            echo=settings.DEBUG,
        )
        self.SessionLocal = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)

    def init(self) -> None:
        """Create all tables if they don't exist."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database initialized at: %s", settings.DATABASE_URL)

    def get_session(self) -> Session:
        """Get a database session. Use as context manager."""
        return self.SessionLocal()

    # ─── Task Records ──────────────────────────────────────────────────────────

    def save_task(
        self,
        task_type: str,
        input_summary: dict,
        result: dict,
        duration_seconds: Optional[float] = None,
        model_used: Optional[str] = None,
        score: Optional[float] = None,
    ) -> int:
        """Save a completed task to the database. Returns the new record ID."""
        with self.get_session() as session:
            record = TaskRecord(
                task_type=task_type,
                status=result.get("status", "success"),
                completed_at=datetime.now(timezone.utc),
                duration_seconds=duration_seconds,
                model_used=model_used,
                input_data=json.dumps(input_summary, default=str),
                result_data=json.dumps(result, default=str),
                score=score,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            logger.debug("Saved task record id=%d, type=%s", record.id, task_type)
            return record.id

    def get_task_history(
        self,
        task_type: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """Return recent task history, optionally filtered by type."""
        with self.get_session() as session:
            query = session.query(TaskRecord).order_by(TaskRecord.created_at.desc())
            if task_type:
                query = query.filter(TaskRecord.task_type == task_type)
            records = query.limit(limit).all()

            return [
                {
                    "id": r.id,
                    "task_type": r.task_type,
                    "status": r.status,
                    "created_at": str(r.created_at),
                    "duration_seconds": r.duration_seconds,
                    "model_used": r.model_used,
                    "score": r.score,
                }
                for r in records
            ]

    def get_task_result(self, record_id: int) -> Optional[dict]:
        """Retrieve the full result of a past task by ID."""
        with self.get_session() as session:
            record = session.query(TaskRecord).filter(TaskRecord.id == record_id).first()
            if not record:
                return None
            result = json.loads(record.result_data) if record.result_data else {}
            result["_record_id"] = record.id
            return result

    # ─── Job Applications ──────────────────────────────────────────────────────

    def save_job_application(
        self,
        company_name: str,
        role_title: str,
        cover_letter: Optional[str] = None,
        application_answers: Optional[dict] = None,
        job_description: Optional[str] = None,
        url: Optional[str] = None,
    ) -> int:
        """Save a job application draft or record."""
        with self.get_session() as session:
            app = JobApplication(
                company_name=company_name,
                role_title=role_title,
                cover_letter=cover_letter,
                application_answers=json.dumps(application_answers or {}),
                job_description=job_description,
                url=url,
            )
            session.add(app)
            session.commit()
            session.refresh(app)
            logger.info("Saved job application id=%d for %s @ %s", app.id, role_title, company_name)
            return app.id

    def get_job_applications(self, status: Optional[str] = None) -> list[dict]:
        with self.get_session() as session:
            query = session.query(JobApplication).order_by(JobApplication.created_at.desc())
            if status:
                query = query.filter(JobApplication.status == status)
            apps = query.all()
            return [
                {
                    "id": a.id,
                    "company_name": a.company_name,
                    "role_title": a.role_title,
                    "status": a.status,
                    "created_at": str(a.created_at),
                    "url": a.url,
                }
                for a in apps
            ]

    def prune_old_records(self) -> int:
        """Remove oldest records if storage limit is exceeded. Returns count deleted."""
        with self.get_session() as session:
            total = session.query(TaskRecord).count()
            if total > settings.MAX_STORED_RESULTS:
                to_delete = total - settings.MAX_STORED_RESULTS
                oldest = (
                    session.query(TaskRecord)
                    .order_by(TaskRecord.created_at.asc())
                    .limit(to_delete)
                    .all()
                )
                for record in oldest:
                    session.delete(record)
                session.commit()
                logger.info("Pruned %d old task records", to_delete)
                return to_delete
            return 0


# ─── Singleton ────────────────────────────────────────────────────────────────
db = DatabaseManager()
