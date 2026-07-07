import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from llava_lens.core.config import Config
from llava_lens.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Metric:
    name: str
    value: float
    step: Optional[int] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Run:
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: Optional[str] = None
    status: str = "running"
    model_name: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None
    metrics: List[Metric] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    artifacts: List[str] = field(default_factory=list)

    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        self.metrics.append(Metric(name=name, value=value, step=step))

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        for name, value in metrics.items():
            self.log_metric(name, value, step)

    def set_status(self, status: str) -> None:
        self.status = status
        if status in ("completed", "failed"):
            self.end_time = datetime.now().isoformat()

    def add_tag(self, tag: str) -> None:
        if tag not in self.tags:
            self.tags.append(tag)

    def add_artifact(self, path: str) -> None:
        if path not in self.artifacts:
            self.artifacts.append(path)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["metrics"] = [asdict(m) for m in self.metrics]
        return d


class ExperimentTracker:
    def __init__(self, config: Config):
        self.config = config
        self.db_path = Path(config.tracking.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    name TEXT,
                    status TEXT,
                    model_name TEXT,
                    config TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    tags TEXT,
                    notes TEXT,
                    artifacts TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    name TEXT,
                    value REAL,
                    step INTEGER,
                    timestamp TEXT,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                )
            """)

    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(str(self.db_path))
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create_run(self, name: Optional[str] = None, **kwargs) -> Run:
        run = Run(name=name, **kwargs)
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO runs (run_id, name, status, model_name, config, start_time, tags, notes, artifacts)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run.run_id,
                    run.name,
                    run.status,
                    run.model_name,
                    json.dumps(run.config),
                    run.start_time,
                    json.dumps(run.tags),
                    run.notes,
                    json.dumps(run.artifacts),
                ),
            )
        logger.info(f"Created run {run.run_id}: {run.name}")
        return run

    def save_run(self, run: Run) -> None:
        with self._get_conn() as conn:
            conn.execute(
                """UPDATE runs SET name=?, status=?, model_name=?, config=?, end_time=?, tags=?, notes=?, artifacts=?
                   WHERE run_id=?""",
                (
                    run.name,
                    run.status,
                    run.model_name,
                    json.dumps(run.config),
                    run.end_time,
                    json.dumps(run.tags),
                    run.notes,
                    json.dumps(run.artifacts),
                    run.run_id,
                ),
            )
            for metric in run.metrics:
                conn.execute(
                    """INSERT INTO metrics (run_id, name, value, step, timestamp)
                       VALUES (?, ?, ?, ?, ?)""",
                    (run.run_id, metric.name, metric.value, metric.step, metric.timestamp),
                )

    def get_run(self, run_id: str) -> Optional[Run]:
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if not row:
                return None
            return self._row_to_run(row)

    def list_runs(self, limit: int = 50) -> List[Run]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY start_time DESC LIMIT ?", (limit,)
            ).fetchall()
            return [self._row_to_run(row) for row in rows]

    def get_run_metrics(self, run_id: str) -> List[Metric]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT name, value, step, timestamp FROM metrics WHERE run_id=? ORDER BY step",
                (run_id,),
            ).fetchall()
            return [Metric(name=r[0], value=r[1], step=r[2], timestamp=r[3]) for r in rows]

    def delete_run(self, run_id: str) -> bool:
        with self._get_conn() as conn:
            conn.execute("DELETE FROM metrics WHERE run_id=?", (run_id,))
            result = conn.execute("DELETE FROM runs WHERE run_id=?", (run_id,))
            return result.rowcount > 0

    def search_runs(self, query: str) -> List[Run]:
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT * FROM runs WHERE name LIKE ? OR notes LIKE ? OR tags LIKE ?
                   ORDER BY start_time DESC""",
                (f"%{query}%", f"%{query}%", f"%{query}%"),
            ).fetchall()
            return [self._row_to_run(row) for row in rows]

    def _row_to_run(self, row) -> Run:
        return Run(
            run_id=row[0],
            name=row[1],
            status=row[2],
            model_name=row[3],
            config=json.loads(row[4]) if row[4] else {},
            start_time=row[5],
            end_time=row[6],
            tags=json.loads(row[7]) if row[7] else [],
            notes=row[8] or "",
            artifacts=json.loads(row[9]) if row[9] else [],
        )
