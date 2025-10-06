
"""
Ultimate Expert Conversational Timesheet API - Enhanced with Mandatory Comments
Added features:
- Mandatory comments validation
- Enhanced date parsing (25 March, March 25, etc.)
- Comments display in timesheet view
- Better natural language processing for dates and comments
"""

import os
import json
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Union
from uuid import uuid4
import re
import calendar

import pyodbc
import dateparser
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from contextlib import asynccontextmanager
import uvicorn

# Professional logging with comprehensive tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s'
)
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_CONFIG = {
    "server": os.getenv("DB_SERVER", "localhost"),
    "database": os.getenv("DB_NAME", "TimesheetDB"),
    "username": os.getenv("DB_USERNAME", "sa"),
    "password": os.getenv("DB_PASSWORD", "YourPassword123"),
    "driver": "ODBC Driver 17 for SQL Server",
    "timeout": 30
}

# Pydantic Models (Enhanced with mandatory comments)
class ChatRequest(BaseModel):
    email: str = Field(..., min_length=5, pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    user_prompt: str = Field(..., min_length=1, max_length=2000)

class TimesheetEntry(BaseModel):
    system: str = Field(..., pattern=r'^(Oracle|Mars)$')
    date: str = Field(..., pattern=r'^\d{4}-\d{2}-\d{2}$')
    hours: float = Field(..., ge=0.25, le=24.0)
    project_code: str = Field(..., min_length=3, max_length=50)
    task_code: Optional[str] = Field(None, max_length=50)
    comments: str = Field(..., min_length=3, max_length=500, description="Comments are mandatory")

    @validator('comments')
    def comments_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Comments are mandatory and cannot be empty')
        if len(v.strip()) < 3:
            raise ValueError('Comments must be at least 3 characters long')
        return v.strip()

class ConversationState(BaseModel):
    user_email: str
    current_entries: List[Dict] = []
    conversation_phase: str = "gathering"
    missing_fields: List[str] = []
    systems_in_progress: List[str] = []
    last_interaction: datetime = Field(default_factory=datetime.utcnow)
    draft_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    tabular_data: Optional[str] = None
    conversation_phase: str
    missing_fields: List[str] = []
    current_data: Optional[Dict] = None
    suggestions: List[str] = []
    session_id: str

class ProjectCodeResponse(BaseModel):
    system: str
    projects: List[Dict]
    count: int
    formatted_display: str

class TimesheetSummaryResponse(BaseModel):
    user_email: str
    system: str
    entries: List[Dict]
    total_hours: float
    count: int
    formatted_display: str

# Enhanced Database Manager with Comments Support
class EnhancedDatabaseManager:
    def __init__(self):
        self.connection_string = self._build_connection_string()
        self._init_connection_pool()
        self._initialize_all_tables()
        logger.info("Enhanced Database Manager initialized successfully")

    def _build_connection_string(self) -> str:
        return (
            f"DRIVER={{{DATABASE_CONFIG['driver']}}};"
            f"SERVER={DATABASE_CONFIG['server']};"
            f"DATABASE={DATABASE_CONFIG['database']};"
            f"UID={DATABASE_CONFIG['username']};"
            f"PWD={DATABASE_CONFIG['password']};"
            f"TrustServerCertificate=yes;"
            f"Connection Timeout={DATABASE_CONFIG['timeout']};"
            f"CommandTimeout={DATABASE_CONFIG['timeout']};"
        )

    def _init_connection_pool(self):
        pyodbc.pooling = True
        pyodbc.timeout = DATABASE_CONFIG['timeout']

    def get_connection(self):
        try:
            conn = pyodbc.connect(self.connection_string)
            conn.timeout = DATABASE_CONFIG['timeout']
            return conn
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise HTTPException(status_code=500, detail="Database connection failed")

    def execute_query(self, query: str, params: tuple = None, fetch: bool = True):
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            if query.strip().upper().startswith(('INSERT', 'UPDATE', 'DELETE', 'MERGE')):
                conn.commit()
                if 'OUTPUT INSERTED.' in query.upper():
                    result = cursor.fetchone()
                    return result[0] if result else None
                return cursor.rowcount
            elif fetch:
                return cursor.fetchall()
            else:
                return None

        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Query execution failed: {e}")
            raise HTTPException(status_code=500, detail=f"Database operation failed: {str(e)}")
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def _initialize_all_tables(self):
        """Initialize all required tables with enhanced comments support"""
        tables = {
            "OracleTimesheet": """
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='OracleTimesheet' AND xtype='U')
            BEGIN
                CREATE TABLE OracleTimesheet (
                    ID BIGINT IDENTITY(1,1) PRIMARY KEY,
                    UserEmail NVARCHAR(255) NOT NULL,
                    EntryDate DATE NOT NULL,
                    ProjectCode NVARCHAR(50) NOT NULL,
                    TaskCode NVARCHAR(50),
                    Hours DECIMAL(5,2) NOT NULL CHECK (Hours > 0 AND Hours <= 24),
                    Comments NVARCHAR(500) NOT NULL CHECK (LEN(TRIM(Comments)) >= 3),
                    Status NVARCHAR(20) DEFAULT 'Draft' CHECK (Status IN ('Draft', 'Submitted', 'Approved')),
                    CreatedAt DATETIME2 DEFAULT GETDATE(),
                    UpdatedAt DATETIME2 DEFAULT GETDATE()
                );
                CREATE INDEX IX_OracleTimesheet_UserEmail_Date ON OracleTimesheet(UserEmail, EntryDate);
                CREATE INDEX IX_OracleTimesheet_Comments ON OracleTimesheet(UserEmail, EntryDate, Comments);
            END
            """,
            "MarsTimesheet": """
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='MarsTimesheet' AND xtype='U')
            BEGIN
                CREATE TABLE MarsTimesheet (
                    ID BIGINT IDENTITY(1,1) PRIMARY KEY,
                    UserEmail NVARCHAR(255) NOT NULL,
                    EntryDate DATE NOT NULL,
                    ProjectCode NVARCHAR(50) NOT NULL,
                    TaskCode NVARCHAR(50),
                    Hours DECIMAL(5,2) NOT NULL CHECK (Hours > 0 AND Hours <= 24),
                    Comments NVARCHAR(500) NOT NULL CHECK (LEN(TRIM(Comments)) >= 3),
                    Status NVARCHAR(20) DEFAULT 'Draft' CHECK (Status IN ('Draft', 'Submitted', 'Approved')),
                    CreatedAt DATETIME2 DEFAULT GETDATE(),
                    UpdatedAt DATETIME2 DEFAULT GETDATE()
                );
                CREATE INDEX IX_MarsTimesheet_UserEmail_Date ON MarsTimesheet(UserEmail, EntryDate);
                CREATE INDEX IX_MarsTimesheet_Comments ON MarsTimesheet(UserEmail, EntryDate, Comments);
            END
            """,
            "ProjectCodes": """
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='ProjectCodes' AND xtype='U')
            BEGIN
                CREATE TABLE ProjectCodes (
                    ID INT IDENTITY(1,1) PRIMARY KEY,
                    ProjectCode NVARCHAR(50) NOT NULL,
                    ProjectName NVARCHAR(255) NOT NULL,
                    System NVARCHAR(20) NOT NULL CHECK (System IN ('Oracle', 'Mars', 'Both')),
                    IsActive BIT DEFAULT 1,
                    CreatedAt DATETIME2 DEFAULT GETDATE()
                );
            END
            """,
            "ConversationSessions": """
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='ConversationSessions' AND xtype='U')
            BEGIN
                CREATE TABLE ConversationSessions (
                    SessionID NVARCHAR(50) PRIMARY KEY,
                    UserEmail NVARCHAR(255) NOT NULL,
                    SessionData NVARCHAR(MAX),
                    ConversationPhase NVARCHAR(50),
                    LastInteraction DATETIME2 DEFAULT GETDATE(),
                    CreatedAt DATETIME2 DEFAULT GETDATE()
                );
                CREATE INDEX IX_ConversationSessions_UserEmail ON ConversationSessions(UserEmail);
            END
            """,
            "TimesheetDrafts": """
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='TimesheetDrafts' AND xtype='U')
            BEGIN
                CREATE TABLE TimesheetDrafts (
                    DraftID NVARCHAR(50) PRIMARY KEY,
                    UserEmail NVARCHAR(255) NOT NULL,
                    DraftData NVARCHAR(MAX),
                    TotalHours DECIMAL(8,2),
                    SystemsUsed NVARCHAR(100),
                    CreatedAt DATETIME2 DEFAULT GETDATE(),
                    UpdatedAt DATETIME2 DEFAULT GETDATE()
                );
                CREATE INDEX IX_TimesheetDrafts_UserEmail ON TimesheetDrafts(UserEmail);
            END
            """
        }

        for table_name, create_sql in tables.items():
            try:
                self.execute_query(create_sql, fetch=False)
                logger.info(f"Initialized table: {table_name}")
            except Exception as e:
                logger.warning(f"Table {table_name} initialization: {e}")

        # Initialize sample project codes
        self._initialize_project_codes()

    def _initialize_project_codes(self):
        """Initialize sample project codes"""
        try:
            check_query = "SELECT COUNT(*) FROM ProjectCodes"
            result = self.execute_query(check_query)
            count = result[0][0] if result and result[0] else 0

            if count == 0:
                sample_projects = [
                    ('ORG-001', 'Oracle Core Development', 'Oracle'),
                    ('ORG-002', 'Oracle Database Maintenance', 'Oracle'),
                    ('ORG-003', 'Oracle Integration Services', 'Oracle'),
                    ('ORG-004', 'Oracle Security Implementation', 'Oracle'),
                    ('ORG-005', 'Oracle Performance Optimization', 'Oracle'),
                    ('MRS-001', 'Mars Data Processing', 'Mars'),
                    ('MRS-002', 'Mars Analytics Platform', 'Mars'),
                    ('MRS-003', 'Mars Reporting Services', 'Mars'),
                    ('MRS-004', 'Mars Machine Learning', 'Mars'),
                    ('MRS-005', 'Mars Data Visualization', 'Mars'),
                    ('CMN-001', 'Common Documentation', 'Both'),
                    ('CMN-002', 'Common Training', 'Both'),
                    ('CMN-003', 'Common Testing', 'Both'),
                    ('CMN-004', 'Common Architecture', 'Both'),
                    ('CMN-005', 'Common Security', 'Both')
                ]

                for code, name, system in sample_projects:
                    insert_query = """
                    INSERT INTO ProjectCodes (ProjectCode, ProjectName, System)
                    VALUES (?, ?, ?)
                    """
                    self.execute_query(insert_query, (code, name, system), fetch=False)
                logger.info("Initialized sample project codes")
        except Exception as e:
            logger.warning(f"Project codes initialization: {e}")

# Enhanced Timesheet Service with Comments Support
class EnhancedTimesheetService:
    def __init__(self, db_manager: EnhancedDatabaseManager):
        self.db_manager = db_manager

    def get_project_codes(self, system: Optional[str] = None) -> ProjectCodeResponse:
        """Get project codes with formatted display"""
        try:
            if system:
                query = """
                SELECT ProjectCode, ProjectName, System
                FROM ProjectCodes 
                WHERE (System = ? OR System = 'Both') AND IsActive = 1
                ORDER BY ProjectCode
                """
                results = self.db_manager.execute_query(query, (system,))
            else:
                query = """
                SELECT ProjectCode, ProjectName, System
                FROM ProjectCodes 
                WHERE IsActive = 1
                ORDER BY System, ProjectCode
                """
                results = self.db_manager.execute_query(query)

            projects = [
                {"code": row[0], "name": row[1], "system": row[2]}
                for row in results
            ] if results else []

            # Format for display
            if projects:
                display_lines = ["\n📋 **AVAILABLE PROJECT CODES**\n"]
                if system:
                    display_lines.append(f"System: **{system}**\n")

                display_lines.append("| Code | Project Name | System |")
                display_lines.append("|------|-------------|---------|")

                for project in projects:
                    display_lines.append(f"| **{project['code']}** | {project['name']} | {project['system']} |")

                display_lines.append(f"\n**Total: {len(projects)} projects available**")
                formatted_display = "\n".join(display_lines)
            else:
                formatted_display = "No project codes found."

            return ProjectCodeResponse(
                system=system or "All",
                projects=projects,
                count=len(projects),
                formatted_display=formatted_display
            )

        except Exception as e:
            logger.error(f"Failed to get project codes: {e}")
            return ProjectCodeResponse(
                system=system or "All",
                projects=[],
                count=0,
                formatted_display="Error retrieving project codes."
            )

    def get_user_timesheet(
        self, 
        user_email: str, 
        system: str, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None
    ) -> TimesheetSummaryResponse:
        """Get user timesheet with enhanced comments display"""
        try:
            table_name = "OracleTimesheet" if system == "Oracle" else "MarsTimesheet"

            query = f"""
            SELECT ID, EntryDate, ProjectCode, TaskCode, Hours, Comments, Status, CreatedAt
            FROM {table_name}
            WHERE UserEmail = ?
            """
            params = [user_email]

            if start_date:
                query += " AND EntryDate >= ?"
                params.append(start_date)
            if end_date:
                query += " AND EntryDate <= ?"
                params.append(end_date)

            query += " ORDER BY EntryDate DESC, CreatedAt DESC"

            results = self.db_manager.execute_query(query, tuple(params))

            entries = []
            total_hours = 0.0

            if results:
                for row in results:
                    entry = {
                        "id": row[0],
                        "date": row[1].isoformat() if hasattr(row[1], 'isoformat') else str(row[1]),
                        "project_code": row[2],
                        "task_code": row[3] or "",
                        "hours": float(row[4]),
                        "comments": row[5] or "",
                        "status": row[6],
                        "created_at": row[7].isoformat() if hasattr(row[7], 'isoformat') else str(row[7])
                    }
                    entries.append(entry)
                    total_hours += entry["hours"]

            # Enhanced format display with full comments
            if entries:
                display_lines = [f"\n📊 **{system.upper()} TIMESHEET SUMMARY**"]
                display_lines.append(f"User: **{user_email}**")
                if start_date or end_date:
                    date_range = f"From: {start_date or 'Beginning'} To: {end_date or 'End'}"
                    display_lines.append(f"Period: {date_range}")
                display_lines.append("")

                display_lines.append("| Date | Project | Task | Hours | Comments | Status |")
                display_lines.append("|------|---------|------|-------|----------|---------|")

                for entry in entries:
                    task = entry["task_code"] or "-"
                    # Show full comments (up to 50 chars in table, full comments below)
                    comments_short = entry["comments"][:47] + "..." if len(entry["comments"]) > 50 else entry["comments"]
                    display_lines.append(
                        f"| {entry['date']} | **{entry['project_code']}** | {task} | "
                        f"**{entry['hours']}** | {comments_short} | {entry['status']} |"
                    )

                display_lines.append("")
                display_lines.append(f"**TOTAL HOURS: {total_hours}** | **ENTRIES: {len(entries)}**")

                # Add detailed comments section
                display_lines.append("\n📝 **DETAILED COMMENTS:**")
                display_lines.append("---")
                for i, entry in enumerate(entries, 1):
                    display_lines.append(f"**{i}. {entry['date']} - {entry['project_code']}:**")
                    display_lines.append(f"   {entry['comments']}")
                    display_lines.append("")

                formatted_display = "\n".join(display_lines)
            else:
                formatted_display = f"No timesheet entries found for {system} system."

            return TimesheetSummaryResponse(
                user_email=user_email,
                system=system,
                entries=entries,
                total_hours=total_hours,
                count=len(entries),
                formatted_display=formatted_display
            )

        except Exception as e:
            logger.error(f"Failed to get timesheet: {e}")
            return TimesheetSummaryResponse(
                user_email=user_email,
                system=system,
                entries=[],
                total_hours=0.0,
                count=0,
                formatted_display="Error retrieving timesheet data."
            )

    def submit_timesheet_entries(self, user_email: str, entries: List[Dict]) -> Dict[str, Any]:
        """Enhanced: Submit entries with mandatory comments validation"""
        try:
            submitted_entries = []

            for entry in entries:
                # Validate comments are provided and not empty
                comments = entry.get('comments', '').strip()
                if not comments or len(comments) < 3:
                    return {
                        "success": False,
                        "error": f"Comments are mandatory and must be at least 3 characters. Entry for {entry.get('project_code', 'Unknown')} has insufficient comments."
                    }

                system = entry['system']
                table_name = "OracleTimesheet" if system == "Oracle" else "MarsTimesheet"

                # Check for existing entry
                check_query = f"""
                SELECT ID FROM {table_name}
                WHERE UserEmail = ? AND EntryDate = ? AND ProjectCode = ?
                """
                existing = self.db_manager.execute_query(
                    check_query,
                    (user_email, entry['date'], entry['project_code'])
                )

                if existing and len(existing) > 0:
                    # Update existing with mandatory comments
                    update_query = f"""
                    UPDATE {table_name}
                    SET Hours = ?, TaskCode = ?, Comments = ?, Status = 'Submitted', UpdatedAt = GETDATE()
                    WHERE ID = ?
                    """
                    self.db_manager.execute_query(
                        update_query,
                        (entry['hours'], entry.get('task_code'), comments, existing[0][0]),
                        fetch=False
                    )
                    entry_id = existing[0][0]
                else:
                    # Insert new with mandatory comments
                    insert_query = f"""
                    INSERT INTO {table_name} (UserEmail, EntryDate, ProjectCode, TaskCode, Hours, Comments, Status)
                    VALUES (?, ?, ?, ?, ?, ?, 'Submitted')
                    """
                    self.db_manager.execute_query(
                        insert_query,
                        (user_email, entry['date'], entry['project_code'], 
                         entry.get('task_code'), entry['hours'], comments),
                        fetch=False
                    )

                    # Get the inserted ID
                    id_query = f"SELECT SCOPE_IDENTITY()"
                    id_result = self.db_manager.execute_query(id_query)
                    entry_id = int(id_result[0][0]) if id_result and id_result[0] else None

                submitted_entries.append({
                    "id": entry_id,
                    "system": system,
                    "date": entry['date'],
                    "project_code": entry['project_code'],
                    "hours": entry['hours'],
                    "comments": comments
                })

            total_hours = sum([entry['hours'] for entry in entries])
            systems_used = list(set([entry['system'] for entry in entries]))

            return {
                "success": True,
                "entries_submitted": len(submitted_entries),
                "total_hours": total_hours,
                "systems_used": systems_used,
                "submitted_entries": submitted_entries
            }

        except Exception as e:
            logger.error(f"Failed to submit entries: {e}")
            return {"success": False, "error": str(e)}

    def save_draft_timesheet(self, user_email: str, entries: List[Dict]) -> Dict[str, Any]:
        """Save timesheet as draft with comments validation"""
        try:
            # Validate all entries have comments
            for entry in entries:
                comments = entry.get('comments', '').strip()
                if not comments or len(comments) < 3:
                    return {
                        "success": False,
                        "error": f"Comments are mandatory. Entry for {entry.get('project_code', 'Unknown')} needs comments (minimum 3 characters)."
                    }

            draft_id = f"draft_{user_email}_{int(datetime.utcnow().timestamp())}"

            systems_used = list(set([entry.get('system') for entry in entries if entry.get('system')]))
            total_hours = sum([entry.get('hours', 0) for entry in entries])

            draft_data = {
                "entries": entries,
                "total_hours": total_hours,
                "systems_used": systems_used,
                "created_at": datetime.utcnow().isoformat()
            }

            insert_query = """
            INSERT INTO TimesheetDrafts (DraftID, UserEmail, DraftData, TotalHours, SystemsUsed)
            VALUES (?, ?, ?, ?, ?)
            """

            self.db_manager.execute_query(
                insert_query,
                (draft_id, user_email, json.dumps(draft_data), total_hours, ",".join(systems_used)),
                fetch=False
            )

            return {
                "success": True,
                "draft_id": draft_id,
                "total_hours": total_hours,
                "systems_used": systems_used,
                "entries_count": len(entries)
            }

        except Exception as e:
            logger.error(f"Failed to save draft: {e}")
            return {"success": False, "error": str(e)}

# Enhanced Session Manager
class EnhancedSessionManager:
    def __init__(self, db_manager: EnhancedDatabaseManager):
        self.db_manager = db_manager
        self.active_sessions: Dict[str, ConversationState] = {}

    def get_or_create_session(self, user_email: str) -> ConversationState:
        session_key = user_email.lower()

        if session_key in self.active_sessions:
            session = self.active_sessions[session_key]
            session.last_interaction = datetime.utcnow()
            return session

        new_session = ConversationState(user_email=user_email)
        self.active_sessions[session_key] = new_session
        return new_session

    def save_session(self, session: ConversationState):
        session_key = session.user_email.lower()
        self.active_sessions[session_key] = session

# Enhanced Parser with Better Date and Comments Processing
class EnhancedTimesheetParser:
    def __init__(self):
        # Month name mappings for better parsing
        self.month_names = {
            'january': 1, 'jan': 1,
            'february': 2, 'feb': 2,
            'march': 3, 'mar': 3,
            'april': 4, 'apr': 4,
            'may': 5,
            'june': 6, 'jun': 6,
            'july': 7, 'jul': 7,
            'august': 8, 'aug': 8,
            'september': 9, 'sep': 9, 'sept': 9,
            'october': 10, 'oct': 10,
            'november': 11, 'nov': 11,
            'december': 12, 'dec': 12
        }

    def parse_user_input(self, user_prompt: str) -> Dict[str, Any]:
        """Enhanced parsing with better date and comments handling"""
        logger.info(f"Enhanced parsing: {user_prompt}")

        data = {}
        prompt_lower = user_prompt.lower()
        original_prompt = user_prompt

        # System extraction
        if re.search(r'\boracle\b', prompt_lower):
            data['system'] = 'Oracle'
        elif re.search(r'\bmars\b', prompt_lower):
            data['system'] = 'Mars'

        # Hours extraction
        hours_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)\b', prompt_lower)
        if hours_match:
            data['hours'] = float(hours_match.group(1))

        # Project code extraction
        project_match = re.search(r'\b([A-Z]{2,4}-\d{3,4})\b', original_prompt.upper())
        if project_match:
            data['project_code'] = project_match.group(1)

        # Enhanced date extraction
        data['date'] = self._parse_date_enhanced(original_prompt, prompt_lower)

        # Enhanced comments extraction
        data['comments'] = self._extract_comments(original_prompt, prompt_lower)

        # Task code extraction
        task_match = re.search(r'task[:\s]+([A-Z0-9-]+)', original_prompt, re.IGNORECASE)
        if task_match:
            data['task_code'] = task_match.group(1)

        logger.info(f"Enhanced parsed result: {data}")
        return data

    def _parse_date_enhanced(self, original_prompt: str, prompt_lower: str) -> Optional[str]:
        """Enhanced date parsing supporting various formats"""

        # 1. Relative dates
        date_keywords = {
            'yesterday': (date.today() - timedelta(days=1)).isoformat(),
            'today': date.today().isoformat(),
            'tomorrow': (date.today() + timedelta(days=1)).isoformat()
        }

        for keyword, date_value in date_keywords.items():
            if keyword in prompt_lower:
                return date_value

        # 2. Standard ISO format
        iso_match = re.search(r'\b(\d{4}-\d{2}-\d{2})\b', original_prompt)
        if iso_match:
            return iso_match.group(1)

        # 3. Enhanced formats: "25 March", "March 25", "25 March 2024", etc.
        current_year = date.today().year

        # Format: "25 March" or "25 March 2024"
        day_month_match = re.search(r'\b(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)(?:\s+(\d{4}))?\b', prompt_lower)
        if day_month_match:
            day = int(day_month_match.group(1))
            month_name = day_month_match.group(2).lower()
            year = int(day_month_match.group(3)) if day_month_match.group(3) else current_year

            if month_name in self.month_names:
                month = self.month_names[month_name]
                try:
                    parsed_date = date(year, month, day)
                    return parsed_date.isoformat()
                except ValueError:
                    pass

        # Format: "March 25" or "March 25 2024"
        month_day_match = re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\s+(\d{1,2})(?:\s+(\d{4}))?\b', prompt_lower)
        if month_day_match:
            month_name = month_day_match.group(1).lower()
            day = int(month_day_match.group(2))
            year = int(month_day_match.group(3)) if month_day_match.group(3) else current_year

            if month_name in self.month_names:
                month = self.month_names[month_name]
                try:
                    parsed_date = date(year, month, day)
                    return parsed_date.isoformat()
                except ValueError:
                    pass

        # 4. Numeric formats: "25/03", "25/03/2024", "03/25", "03/25/2024"
        numeric_match = re.search(r'\b(\d{1,2})/(\d{1,2})(?:/(\d{4}))?\b', original_prompt)
        if numeric_match:
            part1 = int(numeric_match.group(1))
            part2 = int(numeric_match.group(2))
            year = int(numeric_match.group(3)) if numeric_match.group(3) else current_year

            # Try day/month format first (European style)
            if part1 <= 31 and part2 <= 12:
                try:
                    parsed_date = date(year, part2, part1)
                    return parsed_date.isoformat()
                except ValueError:
                    pass

            # Try month/day format (US style)
            if part1 <= 12 and part2 <= 31:
                try:
                    parsed_date = date(year, part1, part2)
                    return parsed_date.isoformat()
                except ValueError:
                    pass

        return None

    def _extract_comments(self, original_prompt: str, prompt_lower: str) -> Optional[str]:
        """Enhanced comments extraction from natural language"""

        # Look for explicit comment markers
        comment_patterns = [
            r'comment[s]?[:\s]+([^\n\r]+)',
            r'description[s]?[:\s]+([^\n\r]+)',
            r'note[s]?[:\s]+([^\n\r]+)',
            r'work[ed]?\s+on[:\s]+([^\n\r]+)',
            r'task[:\s]+[A-Z0-9-]+[,\s]+(.+)',  # After task code
            r'for[:\s]+([^\n\r]+)',
            r'doing[:\s]+([^\n\r]+)',
            r'activity[:\s]+([^\n\r]+)'
        ]

        for pattern in comment_patterns:
            match = re.search(pattern, original_prompt, re.IGNORECASE)
            if match:
                comment = match.group(1).strip()
                # Clean up the comment
                comment = re.sub(r'^[,;:\s]+', '', comment)  # Remove leading punctuation
                comment = re.sub(r'[,;\s]+$', '', comment)  # Remove trailing punctuation
                if len(comment) >= 3:
                    return comment

        # Look for descriptive phrases after project code
        project_match = re.search(r'\b[A-Z]{2,4}-\d{3,4}\b[,\s]+(.+)', original_prompt, re.IGNORECASE)
        if project_match:
            potential_comment = project_match.group(1).strip()
            # Remove task code if present
            potential_comment = re.sub(r'task[:\s]+[A-Z0-9-]+[,\s]*', '', potential_comment, flags=re.IGNORECASE)
            # Clean and validate
            potential_comment = re.sub(r'^[,;:\s]+', '', potential_comment)
            potential_comment = re.sub(r'[,;\s]+$', '', potential_comment)

            if len(potential_comment) >= 3 and not re.match(r'^\d+\s*hours?', potential_comment.lower()):
                return potential_comment

        # Look for work descriptions in natural language
        work_descriptions = [
            r'(developed?|developing|development)\s+(.+)',
            r'(fixed?|fixing|fix)\s+(.+)',
            r'(implemented?|implementing|implementation)\s+(.+)',
            r'(tested?|testing|test)\s+(.+)',
            r'(analyzed?|analyzing|analysis)\s+(.+)',
            r'(designed?|designing|design)\s+(.+)',
            r'(created?|creating|creation)\s+(.+)',
            r'(updated?|updating|update)\s+(.+)',
            r'(maintained?|maintaining|maintenance)\s+(.+)',
            r'(reviewed?|reviewing|review)\s+(.+)'
        ]

        for pattern in work_descriptions:
            match = re.search(pattern, prompt_lower)
            if match:
                activity = match.group(1).title()
                description = match.group(2).strip()
                if len(description) >= 3:
                    return f"{activity} {description}"

        return None

# Enhanced Chatbot Controller with Comments Validation
class EnhancedChatbotController:
    def __init__(self):
        self.db_manager = EnhancedDatabaseManager()
        self.session_manager = EnhancedSessionManager(self.db_manager)
        self.parser = EnhancedTimesheetParser()
        self.timesheet_service = EnhancedTimesheetService(self.db_manager)

        logger.info("Enhanced Chatbot Controller initialized with comments validation")

    async def process_chat_message(self, chat_request: ChatRequest) -> ChatResponse:
        """Process chat with enhanced comments validation"""
        try:
            session = self.session_manager.get_or_create_session(chat_request.email)
            user_prompt = chat_request.user_prompt.strip().lower()

            logger.info(f"Processing: {chat_request.email} -> {chat_request.user_prompt}")

            # Handle special commands
            if self._is_command(user_prompt):
                return await self._handle_command(session, chat_request.user_prompt)

            # Handle confirmation
            if session.conversation_phase == "confirmation":
                return await self._handle_confirmation(session, user_prompt)

            # Parse user input
            parsed_data = self.parser.parse_user_input(chat_request.user_prompt)

            # Update session
            self._update_session_data(session, parsed_data)

            # Check missing fields (including mandatory comments)
            missing_fields = self._get_missing_fields_with_comments(session)

            if not missing_fields and session.current_entries:
                session.conversation_phase = "confirmation"
                response = self._generate_confirmation_response(session)
            else:
                session.conversation_phase = "gathering"
                response = self._generate_gathering_response_with_comments(session, missing_fields)

            self.session_manager.save_session(session)

            return ChatResponse(
                response=response,
                tabular_data=response if "**" in response else None,
                conversation_phase=session.conversation_phase,
                missing_fields=missing_fields,
                current_data={"entries": session.current_entries},
                suggestions=self._generate_suggestions_with_comments(session, missing_fields),
                session_id=f"session_{chat_request.email}"
            )

        except Exception as e:
            logger.error(f"Chat processing failed: {e}")
            return ChatResponse(
                response=f"I encountered an error: {str(e)}. Please try again or type 'start fresh'.",
                tabular_data=None,
                conversation_phase="error",
                missing_fields=[],
                suggestions=["start fresh", "help"],
                session_id=f"session_{chat_request.email}_error"
            )

    def _is_command(self, prompt: str) -> bool:
        commands = ['show projects', 'show timesheet', 'help', 'start fresh', 'projects', 'timesheet']
        return any(cmd in prompt for cmd in commands)

    async def _handle_command(self, session: ConversationState, prompt: str) -> ChatResponse:
        prompt_lower = prompt.lower().strip()

        if 'show projects' in prompt_lower or prompt_lower == 'projects':
            system = None
            if 'oracle' in prompt_lower:
                system = 'Oracle'
            elif 'mars' in prompt_lower:
                system = 'Mars'

            project_response = self.timesheet_service.get_project_codes(system)

            return ChatResponse(
                response=project_response.formatted_display,
                tabular_data=project_response.formatted_display,
                conversation_phase=session.conversation_phase,
                missing_fields=[],
                suggestions=["Use a project code in your entry", "Include comments about your work"],
                session_id=f"session_{session.user_email}"
            )

        elif 'show timesheet' in prompt_lower or prompt_lower == 'timesheet':
            system = 'Oracle'
            if 'mars' in prompt_lower:
                system = 'Mars'

            timesheet_response = self.timesheet_service.get_user_timesheet(session.user_email, system)

            return ChatResponse(
                response=timesheet_response.formatted_display,
                tabular_data=timesheet_response.formatted_display,
                conversation_phase=session.conversation_phase,
                missing_fields=[],
                suggestions=["Add new entry with comments", "Show other system"],
                session_id=f"session_{session.user_email}"
            )

        elif 'help' in prompt_lower:
            help_text = """
🎯 **ENHANCED TIMESHEET ASSISTANT - HELP**

**✅ MANDATORY FIELDS:**
• System: Oracle or Mars
• Date: 25 March, March 25, yesterday, today, 2024-03-25
• Hours: 8, 6.5, etc.
• Project Code: ORG-001, MRS-002, etc.
• **Comments: MANDATORY - Describe your work (minimum 3 characters)**

**📝 Entry Examples with Comments:**
• "8 hours Oracle ORG-001 yesterday, developed new authentication system"
• "6 hours Mars MRS-002 25 March, fixed data processing bugs"
• "Oracle: 4 hours ORG-003, task DEV-001, implemented API endpoints"

**💬 Comment Examples:**
• "Database optimization work"
• "Fixed critical security vulnerabilities"
• "Developed user interface components"
• "Code review and testing activities"

**📋 Available Commands:**
• `show projects [Oracle/Mars]` - View project codes
• `show timesheet [Oracle/Mars]` - View entries with comments
• `start fresh` - Begin new entry
• `help` - Show this help
"""

            return ChatResponse(
                response=help_text,
                tabular_data=help_text,
                conversation_phase=session.conversation_phase,
                missing_fields=[],
                suggestions=["Try: '8 hours Oracle ORG-001 today, database work'"],
                session_id=f"session_{session.user_email}"
            )

        elif 'start fresh' in prompt_lower:
            session.current_entries = []
            session.conversation_phase = "gathering"
            session.missing_fields = []

            return ChatResponse(
                response="✨ **Fresh start!** Ready for your timesheet entry.\n\n**Remember: Comments are mandatory!**\n\nTell me what you worked on with a description.",
                tabular_data=None,
                conversation_phase="gathering",
                missing_fields=[],
                suggestions=["8 hours Oracle ORG-001 today, database optimization", "Show projects"],
                session_id=f"session_{session.user_email}"
            )

        return ChatResponse(
            response="Command not recognized. Type 'help' for available commands.",
            tabular_data=None,
            conversation_phase=session.conversation_phase,
            missing_fields=[],
            suggestions=["help", "start fresh"],
            session_id=f"session_{session.user_email}"
        )

    async def _handle_confirmation(self, session: ConversationState, user_prompt: str) -> ChatResponse:
        if any(word in user_prompt for word in ['yes', 'confirm', 'submit', 'ok', 'y']):
            # Enhanced: Submit entries with comments validation
            result = self.timesheet_service.submit_timesheet_entries(
                session.user_email, 
                session.current_entries
            )

            if result["success"]:
                success_message = f"""
🎉 **TIMESHEET SUBMITTED SUCCESSFULLY!**

**Entries Submitted:** {result['entries_submitted']}
**Total Hours:** {result['total_hours']}
**Systems Used:** {', '.join(result['systems_used'])}

✅ All entries with comments have been saved to the database.

📝 **Comments Included:**
"""

                for i, entry in enumerate(result['submitted_entries'], 1):
                    success_message += f"\n{i}. **{entry['project_code']}**: {entry['comments']}"

                # Reset session
                session.current_entries = []
                session.conversation_phase = "completed"
                session.missing_fields = []

                return ChatResponse(
                    response=success_message,
                    tabular_data=success_message,
                    conversation_phase="completed",
                    missing_fields=[],
                    suggestions=["Add another entry", "Show timesheet"],
                    session_id=f"session_{session.user_email}"
                )
            else:
                return ChatResponse(
                    response=f"❌ **Error submitting:** {result.get('error', 'Unknown error')}",
                    tabular_data=None,
                    conversation_phase="confirmation",
                    missing_fields=[],
                    suggestions=["Try again", "Add comments"],
                    session_id=f"session_{session.user_email}"
                )

        elif any(word in user_prompt for word in ['no', 'cancel', 'abort', 'n']):
            session.current_entries = []
            session.conversation_phase = "gathering"

            return ChatResponse(
                response="❌ **Cancelled.** Let's start over.\n\nRemember to include comments about your work!",
                tabular_data=None,
                conversation_phase="gathering",
                missing_fields=[],
                suggestions=["8 hours Oracle ORG-001 today, development work"],
                session_id=f"session_{session.user_email}"
            )

        return ChatResponse(
            response="Please respond with **'YES'** to submit or **'NO'** to cancel.",
            tabular_data=None,
            conversation_phase="confirmation",
            missing_fields=[],
            suggestions=["YES", "NO"],
            session_id=f"session_{session.user_email}"
        )

    def _update_session_data(self, session: ConversationState, parsed_data: Dict):
        if not parsed_data:
            return

        if len(session.current_entries) == 0:
            session.current_entries.append({})

        current_entry = session.current_entries[-1]

        for key, value in parsed_data.items():
            if value is not None:
                current_entry[key] = value

    def _get_missing_fields_with_comments(self, session: ConversationState) -> List[str]:
        """Enhanced: Include comments as mandatory field"""
        if not session.current_entries:
            return ['system', 'date', 'hours', 'project_code', 'comments']

        required_fields = ['system', 'date', 'hours', 'project_code', 'comments']
        missing = []

        for entry in session.current_entries:
            for field in required_fields:
                if field not in entry or entry[field] is None:
                    if field not in missing:
                        missing.append(field)
                elif field == 'comments' and (not entry[field] or len(entry[field].strip()) < 3):
                    if field not in missing:
                        missing.append(field)

        return missing

    def _generate_gathering_response_with_comments(self, session: ConversationState, missing_fields: List[str]) -> str:
        """Enhanced response generation with comments emphasis"""
        if not session.current_entries:
            return "👋 Hello! I'm ready to help with your timesheet.\n\n**📝 Remember: Comments are mandatory!**\n\nTell me what you worked on (e.g., '8 hours Oracle ORG-001 today, database optimization work')."

        current_entry = session.current_entries[-1]
        response_parts = ["I have the following information:\n"]

        for field, value in current_entry.items():
            if value is not None:
                display_field = field.replace('_', ' ').title()
                if field == 'comments':
                    response_parts.append(f"• {display_field}: **"{value}"**")
                else:
                    response_parts.append(f"• {display_field}: **{value}**")

        if missing_fields:
            response_parts.append("\n🔍 I still need:\n")
            field_questions = {
                'system': "Which system? (Oracle or Mars)",
                'date': "What date? (25 March, yesterday, today, 2024-03-25)",
                'hours': "How many hours? (8, 6.5, etc.)",
                'project_code': "Project code? (ORG-001, MRS-002, etc.)",
                'comments': "**Comments about your work? (MANDATORY - minimum 3 characters)**"
            }

            for field in missing_fields:
                question = field_questions.get(field, f"Please provide {field}")
                if field == 'comments':
                    response_parts.append(f"• {question} ⭐")
                else:
                    response_parts.append(f"• {question}")

        if 'comments' in missing_fields:
            response_parts.append("\n💡 **Comment examples:**")
            response_parts.append("• "Database optimization and performance tuning"")
            response_parts.append("• "Fixed critical bugs in authentication system"")
            response_parts.append("• "Developed new API endpoints for user management"")

        return "\n".join(response_parts)

    def _generate_confirmation_response(self, session: ConversationState) -> str:
        """Enhanced confirmation with comments display"""
        response_parts = ["✅ **READY TO SUBMIT**\n"]

        for i, entry in enumerate(session.current_entries, 1):
            response_parts.append(f"**Entry {i}:**")
            response_parts.append(f"• System: **{entry.get('system')}**")
            response_parts.append(f"• Date: **{entry.get('date')}**")
            response_parts.append(f"• Hours: **{entry.get('hours')}**")
            response_parts.append(f"• Project: **{entry.get('project_code')}**")
            if entry.get('task_code'):
                response_parts.append(f"• Task: **{entry.get('task_code')}**")
            response_parts.append(f"• Comments: **"{entry.get('comments')}"** ✅")
            response_parts.append("")

        response_parts.append("**Please respond with 'YES' to submit or 'NO' to cancel.**")
        return "\n".join(response_parts)

    def _generate_suggestions_with_comments(self, session: ConversationState, missing_fields: List[str]) -> List[str]:
        """Enhanced suggestions with comments focus"""
        if session.conversation_phase == "gathering":
            if 'system' in missing_fields:
                return ["Oracle", "Mars"]
            elif 'project_code' in missing_fields:
                return ["Show projects", "ORG-001", "MRS-002"]
            elif 'date' in missing_fields:
                return ["yesterday", "today", "25 March"]
            elif 'hours' in missing_fields:
                return ["8 hours", "6 hours"]
            elif 'comments' in missing_fields:
                return [
                    "Database optimization work",
                    "Bug fixes and testing",
                    "Development of new features",
                    "Code review activities"
                ]
        elif session.conversation_phase == "confirmation":
            return ["YES", "NO"]

        return ["Help", "Start fresh"]

# FastAPI Application Setup (Enhanced)
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting ENHANCED Ultimate Timesheet API with Mandatory Comments")

    try:
        db_manager = EnhancedDatabaseManager()
        db_manager.execute_query("SELECT 1", fetch=True)
        logger.info("✅ Database connection successful")
        logger.info("🎯 ENHANCED API ready with mandatory comments validation!")
        logger.info("📝 Comments are now required for all timesheet entries")

    except Exception as e:
        logger.error(f"❌ Startup error: {e}")

    yield
    logger.info("🛑 Shutting down ENHANCED API")

app = FastAPI(
    title="Ultimate Expert Timesheet API - Enhanced with Mandatory Comments",
    description="Enhanced with mandatory comments validation and improved date parsing",
    version="3.1.0-ENHANCED",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize enhanced controller
controller = EnhancedChatbotController()

# API Endpoints (Enhanced)
@app.get("/")
async def root():
    return {
        "message": "Ultimate Expert Timesheet API - Enhanced with Mandatory Comments",
        "version": "3.1.0-ENHANCED",
        "status": "operational",
        "new_features": [
            "Mandatory comments validation",
            "Enhanced date parsing (25 March, March 25, etc.)",
            "Full comments display in timesheet view",
            "Better natural language processing"
        ]
    }

@app.get("/health")
async def health_check():
    try:
        db_manager = EnhancedDatabaseManager()
        db_manager.execute_query("SELECT 1", fetch=True)
        db_healthy = True
        db_message = "Database connected and operational"
    except Exception as e:
        db_healthy = False
        db_message = f"Database error: {str(e)}"

    return {
        "status": "healthy" if db_healthy else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "3.1.0-ENHANCED",
        "components": {
            "database": {
                "status": "healthy" if db_healthy else "unhealthy",
                "message": db_message
            },
            "comments_validation": {
                "status": "enabled",
                "message": "Comments are mandatory for all entries"
            }
        },
        "enhancements": [
            "Mandatory comments validation",
            "Enhanced date parsing",
            "Improved natural language processing"
        ]
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    return await controller.process_chat_message(chat_request)

@app.get("/projects")
@app.get("/projects/{system}")
async def get_project_codes(system: Optional[str] = None):
    if system and system not in ["Oracle", "Mars"]:
        raise HTTPException(status_code=400, detail="Invalid system")

    try:
        return controller.timesheet_service.get_project_codes(system)
    except Exception as e:
        logger.error(f"Failed to get project codes: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve project codes")

@app.get("/timesheet/{email}/{system}")
async def get_user_timesheet(
    email: str, 
    system: str,
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None)
):
    if system not in ["Oracle", "Mars"]:
        raise HTTPException(status_code=400, detail="Invalid system")

    try:
        return controller.timesheet_service.get_user_timesheet(email, system, start_date, end_date)
    except Exception as e:
        logger.error(f"Failed to get timesheet: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve timesheet")

@app.post("/timesheet/submit")
async def submit_timesheet_entries(entries: List[TimesheetEntry], user_email: str):
    try:
        entry_dicts = []
        for entry in entries:
            entry_dicts.append({
                "system": entry.system,
                "date": entry.date,
                "hours": entry.hours,
                "project_code": entry.project_code,
                "task_code": entry.task_code,
                "comments": entry.comments  # Now mandatory
            })

        result = controller.timesheet_service.submit_timesheet_entries(user_email, entry_dicts)

        if result["success"]:
            return {
                "message": "Timesheet entries with comments submitted successfully",
                "entries_submitted": result["entries_submitted"],
                "total_hours": result["total_hours"],
                "systems_used": result["systems_used"],
                "comments_included": True
            }
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "Submission failed"))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit entries: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit timesheet entries")

@app.post("/timesheet/draft")
async def save_draft_timesheet(user_email: str, entries: List[Dict]):
    try:
        result = controller.timesheet_service.save_draft_timesheet(user_email, entries)

        if result["success"]:
            return {
                "message": "Draft with comments saved successfully",
                "draft_id": result["draft_id"],
                "total_hours": result["total_hours"],
                "systems_used": result["systems_used"]
            }
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "Draft save failed"))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save draft: {e}")
        raise HTTPException(status_code=500, detail="Failed to save draft timesheet")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
