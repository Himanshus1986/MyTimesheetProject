
"""
Ultimate Expert Conversational Timesheet API - SQL Server Fixed Version
Fixed SQL syntax issues for timesheet submission and database operations
"""

import os
import json
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Union
from uuid import uuid4
import re

import pyodbc
import dateparser
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
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

# Pydantic Models (Fixed)
class ChatRequest(BaseModel):
    email: str = Field(..., min_length=5, pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    user_prompt: str = Field(..., min_length=1, max_length=2000)

class TimesheetEntry(BaseModel):
    system: str = Field(..., pattern=r'^(Oracle|Mars)$')
    date: str = Field(..., pattern=r'^\d{4}-\d{2}-\d{2}$')
    hours: float = Field(..., ge=0.25, le=24.0)
    project_code: str = Field(..., min_length=3, max_length=50)
    task_code: Optional[str] = Field(None, max_length=50)
    comments: Optional[str] = Field(None, max_length=500)

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

# FIXED: Database Manager with corrected SQL syntax
class FixedDatabaseManager:
    def __init__(self):
        self.connection_string = self._build_connection_string()
        self._init_connection_pool()
        self._initialize_all_tables()
        logger.info("Fixed Database Manager initialized successfully")

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
        """Initialize all required tables with FIXED SQL syntax"""
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
                    Comments NVARCHAR(500),
                    Status NVARCHAR(20) DEFAULT 'Draft' CHECK (Status IN ('Draft', 'Submitted', 'Approved')),
                    CreatedAt DATETIME2 DEFAULT GETDATE(),
                    UpdatedAt DATETIME2 DEFAULT GETDATE()
                );
                CREATE INDEX IX_OracleTimesheet_UserEmail_Date ON OracleTimesheet(UserEmail, EntryDate);
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
                    Comments NVARCHAR(500),
                    Status NVARCHAR(20) DEFAULT 'Draft' CHECK (Status IN ('Draft', 'Submitted', 'Approved')),
                    CreatedAt DATETIME2 DEFAULT GETDATE(),
                    UpdatedAt DATETIME2 DEFAULT GETDATE()
                );
                CREATE INDEX IX_MarsTimesheet_UserEmail_Date ON MarsTimesheet(UserEmail, EntryDate);
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
        """Initialize sample project codes with FIXED SQL syntax"""
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

# FIXED: Timesheet Service with corrected SQL
class FixedTimesheetService:
    def __init__(self, db_manager: FixedDatabaseManager):
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
        """Get user timesheet with formatted display"""
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

            # Format display
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
                    comments = entry["comments"][:30] + "..." if len(entry["comments"]) > 30 else entry["comments"] or "-"
                    display_lines.append(
                        f"| {entry['date']} | **{entry['project_code']}** | {task} | "
                        f"**{entry['hours']}** | {comments} | {entry['status']} |"
                    )

                display_lines.append("")
                display_lines.append(f"**TOTAL HOURS: {total_hours}** | **ENTRIES: {len(entries)}**")
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
        """FIXED: Submit multiple timesheet entries with corrected SQL syntax"""
        try:
            submitted_entries = []

            for entry in entries:
                system = entry['system']
                table_name = "OracleTimesheet" if system == "Oracle" else "MarsTimesheet"

                # FIXED: Check for existing entry with simple SELECT
                check_query = f"""
                SELECT ID FROM {table_name}
                WHERE UserEmail = ? AND EntryDate = ? AND ProjectCode = ?
                """
                existing = self.db_manager.execute_query(
                    check_query,
                    (user_email, entry['date'], entry['project_code'])
                )

                if existing and len(existing) > 0:
                    # FIXED: Update existing with proper SQL syntax
                    update_query = f"""
                    UPDATE {table_name}
                    SET Hours = ?, TaskCode = ?, Comments = ?, Status = 'Submitted', UpdatedAt = GETDATE()
                    WHERE ID = ?
                    """
                    self.db_manager.execute_query(
                        update_query,
                        (entry['hours'], entry.get('task_code'), entry.get('comments'), existing[0][0]),
                        fetch=False
                    )
                    entry_id = existing[0][0]
                else:
                    # FIXED: Insert new with proper SQL syntax
                    insert_query = f"""
                    INSERT INTO {table_name} (UserEmail, EntryDate, ProjectCode, TaskCode, Hours, Comments, Status)
                    VALUES (?, ?, ?, ?, ?, ?, 'Submitted')
                    """
                    self.db_manager.execute_query(
                        insert_query,
                        (user_email, entry['date'], entry['project_code'], 
                         entry.get('task_code'), entry['hours'], entry.get('comments')),
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
                    "hours": entry['hours']
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
        """FIXED: Save timesheet as draft with corrected SQL"""
        try:
            draft_id = f"draft_{user_email}_{int(datetime.utcnow().timestamp())}"

            systems_used = list(set([entry.get('system') for entry in entries if entry.get('system')]))
            total_hours = sum([entry.get('hours', 0) for entry in entries])

            draft_data = {
                "entries": entries,
                "total_hours": total_hours,
                "systems_used": systems_used,
                "created_at": datetime.utcnow().isoformat()
            }

            # FIXED: Simple INSERT statement
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

# Session Manager (Simplified for SQL Server compatibility)
class FixedSessionManager:
    def __init__(self, db_manager: FixedDatabaseManager):
        self.db_manager = db_manager
        self.active_sessions: Dict[str, ConversationState] = {}

    def get_or_create_session(self, user_email: str) -> ConversationState:
        session_key = user_email.lower()

        if session_key in self.active_sessions:
            session = self.active_sessions[session_key]
            session.last_interaction = datetime.utcnow()
            return session

        # Create new session (simplified)
        new_session = ConversationState(user_email=user_email)
        self.active_sessions[session_key] = new_session
        return new_session

    def save_session(self, session: ConversationState):
        # Keep in memory for now (can be enhanced later)
        session_key = session.user_email.lower()
        self.active_sessions[session_key] = session

# Simple Parser (without complex LLM dependencies)
class SimpleTimesheetParser:
    def parse_user_input(self, user_prompt: str) -> Dict[str, Any]:
        """Simple but effective parsing"""
        logger.info(f"Parsing: {user_prompt}")

        data = {}
        prompt_lower = user_prompt.lower()

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
        project_match = re.search(r'\b([A-Z]{2,4}-\d{3,4})\b', user_prompt.upper())
        if project_match:
            data['project_code'] = project_match.group(1)

        # Date extraction
        date_keywords = {
            'yesterday': (date.today() - timedelta(days=1)).isoformat(),
            'today': date.today().isoformat(),
            'tomorrow': (date.today() + timedelta(days=1)).isoformat()
        }

        for keyword, date_value in date_keywords.items():
            if keyword in prompt_lower:
                data['date'] = date_value
                break

        # Specific date pattern
        date_match = re.search(r'\b(\d{4}-\d{2}-\d{2})\b', user_prompt)
        if date_match:
            data['date'] = date_match.group(1)

        logger.info(f"Parsed result: {data}")
        return data

# Simple Chatbot Controller
class SimpleChatbotController:
    def __init__(self):
        self.db_manager = FixedDatabaseManager()
        self.session_manager = FixedSessionManager(self.db_manager)
        self.parser = SimpleTimesheetParser()
        self.timesheet_service = FixedTimesheetService(self.db_manager)

        logger.info("Fixed Chatbot Controller initialized")

    async def process_chat_message(self, chat_request: ChatRequest) -> ChatResponse:
        """Process chat with fixed SQL operations"""
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

            # Check missing fields
            missing_fields = self._get_missing_fields(session)

            if not missing_fields and session.current_entries:
                session.conversation_phase = "confirmation"
                response = self._generate_confirmation_response(session)
            else:
                session.conversation_phase = "gathering"
                response = self._generate_gathering_response(session, missing_fields)

            self.session_manager.save_session(session)

            return ChatResponse(
                response=response,
                tabular_data=response if "**" in response else None,
                conversation_phase=session.conversation_phase,
                missing_fields=missing_fields,
                current_data={"entries": session.current_entries},
                suggestions=self._generate_suggestions(session, missing_fields),
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
                suggestions=["Use a project code in your entry", "Type your timesheet details"],
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
                suggestions=["Add new entry", "Show other system"],
                session_id=f"session_{session.user_email}"
            )

        elif 'help' in prompt_lower:
            help_text = """
🎯 **FIXED TIMESHEET ASSISTANT - HELP**

**Available Commands:**
• `show projects [Oracle/Mars]` - View project codes
• `show timesheet [Oracle/Mars]` - View your entries
• `start fresh` - Begin new entry
• `help` - Show this help

**Entry Examples:**
• "8 hours Oracle ORG-001 yesterday"
• "6 hours Mars MRS-002 today"
• "Oracle: 4 hours ORG-003, task DEV-001"

**Required Fields:**
• System: Oracle or Mars
• Date: yesterday, today, 2024-01-15
• Hours: 8, 6.5, etc.
• Project Code: ORG-001, MRS-002, etc.
"""

            return ChatResponse(
                response=help_text,
                tabular_data=help_text,
                conversation_phase=session.conversation_phase,
                missing_fields=[],
                suggestions=["Try: '8 hours Oracle ORG-001 today'"],
                session_id=f"session_{session.user_email}"
            )

        elif 'start fresh' in prompt_lower:
            session.current_entries = []
            session.conversation_phase = "gathering"
            session.missing_fields = []

            return ChatResponse(
                response="✨ **Fresh start!** Ready for your timesheet entry.\n\nTell me what you worked on.",
                tabular_data=None,
                conversation_phase="gathering",
                missing_fields=[],
                suggestions=["8 hours Oracle ORG-001 today", "Show projects"],
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
            # FIXED: Submit entries with corrected SQL
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

✅ All entries have been saved to the database.
"""

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
                    suggestions=["Try again", "Start fresh"],
                    session_id=f"session_{session.user_email}"
                )

        elif any(word in user_prompt for word in ['no', 'cancel', 'abort', 'n']):
            session.current_entries = []
            session.conversation_phase = "gathering"

            return ChatResponse(
                response="❌ **Cancelled.** Let's start over.\n\nTell me about your timesheet entry.",
                tabular_data=None,
                conversation_phase="gathering",
                missing_fields=[],
                suggestions=["8 hours Oracle ORG-001 today"],
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

    def _get_missing_fields(self, session: ConversationState) -> List[str]:
        if not session.current_entries:
            return ['system', 'date', 'hours', 'project_code']

        required_fields = ['system', 'date', 'hours', 'project_code']
        missing = []

        for entry in session.current_entries:
            for field in required_fields:
                if field not in entry or entry[field] is None:
                    if field not in missing:
                        missing.append(field)

        return missing

    def _generate_gathering_response(self, session: ConversationState, missing_fields: List[str]) -> str:
        if not session.current_entries:
            return "👋 Hello! I'm ready to help with your timesheet.\n\nTell me what you worked on (e.g., '8 hours Oracle ORG-001 today')."

        current_entry = session.current_entries[-1]
        response_parts = ["I have the following information:\n"]

        for field, value in current_entry.items():
            if value is not None:
                display_field = field.replace('_', ' ').title()
                response_parts.append(f"• {display_field}: **{value}**")

        if missing_fields:
            response_parts.append("\nI still need:\n")
            field_questions = {
                'system': "Which system? (Oracle or Mars)",
                'date': "What date? (yesterday, today, 2024-01-15)",
                'hours': "How many hours? (8, 6.5, etc.)",
                'project_code': "Project code? (ORG-001, MRS-002, etc.)"
            }

            for field in missing_fields:
                question = field_questions.get(field, f"Please provide {field}")
                response_parts.append(f"• {question}")

        return "\n".join(response_parts)

    def _generate_confirmation_response(self, session: ConversationState) -> str:
        response_parts = ["✅ **READY TO SUBMIT**\n"]

        for i, entry in enumerate(session.current_entries, 1):
            response_parts.append(f"**Entry {i}:**")
            response_parts.append(f"• System: **{entry.get('system')}**")
            response_parts.append(f"• Date: **{entry.get('date')}**")
            response_parts.append(f"• Hours: **{entry.get('hours')}**")
            response_parts.append(f"• Project: **{entry.get('project_code')}**")
            if entry.get('task_code'):
                response_parts.append(f"• Task: **{entry.get('task_code')}**")
            if entry.get('comments'):
                response_parts.append(f"• Comments: **{entry.get('comments')}**")
            response_parts.append("")

        response_parts.append("**Please respond with 'YES' to submit or 'NO' to cancel.**")
        return "\n".join(response_parts)

    def _generate_suggestions(self, session: ConversationState, missing_fields: List[str]) -> List[str]:
        if session.conversation_phase == "gathering":
            if 'system' in missing_fields:
                return ["Oracle", "Mars"]
            elif 'project_code' in missing_fields:
                return ["Show projects", "ORG-001", "MRS-002"]
            elif 'date' in missing_fields:
                return ["yesterday", "today"]
            elif 'hours' in missing_fields:
                return ["8 hours", "6 hours"]
        elif session.conversation_phase == "confirmation":
            return ["YES", "NO"]

        return ["Help", "Start fresh"]

# FastAPI Application Setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting FIXED Ultimate Timesheet API")

    try:
        db_manager = FixedDatabaseManager()
        db_manager.execute_query("SELECT 1", fetch=True)
        logger.info("✅ Database connection successful")
        logger.info("🎯 FIXED API ready with corrected SQL syntax!")

    except Exception as e:
        logger.error(f"❌ Startup error: {e}")

    yield
    logger.info("🛑 Shutting down FIXED API")

app = FastAPI(
    title="Ultimate Expert Timesheet API - FIXED VERSION",
    description="Fixed SQL Server compatibility issues for timesheet operations",
    version="3.0.1-FIXED",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize controller
controller = SimpleChatbotController()

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Ultimate Expert Timesheet API - FIXED VERSION",
        "version": "3.0.1-FIXED",
        "status": "operational",
        "fixes": [
            "SQL Server MERGE statement syntax fixed",
            "INSERT/UPDATE operations corrected",
            "Simplified database operations",
            "Improved error handling"
        ]
    }

@app.get("/health")
async def health_check():
    try:
        db_manager = FixedDatabaseManager()
        db_manager.execute_query("SELECT 1", fetch=True)
        db_healthy = True
        db_message = "Database connected and operational"
    except Exception as e:
        db_healthy = False
        db_message = f"Database error: {str(e)}"

    return {
        "status": "healthy" if db_healthy else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "3.0.1-FIXED",
        "components": {
            "database": {
                "status": "healthy" if db_healthy else "unhealthy",
                "message": db_message
            }
        },
        "fixes_applied": [
            "SQL MERGE statement corrected",
            "INSERT/UPDATE syntax fixed",
            "Error handling improved"
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
                "comments": entry.comments
            })

        result = controller.timesheet_service.submit_timesheet_entries(user_email, entry_dicts)

        if result["success"]:
            return {
                "message": "Timesheet entries submitted successfully",
                "entries_submitted": result["entries_submitted"],
                "total_hours": result["total_hours"],
                "systems_used": result["systems_used"]
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
                "message": "Draft saved successfully",
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
