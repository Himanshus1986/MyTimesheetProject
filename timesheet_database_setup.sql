
/*
================================================================
Ultimate Expert Conversational Timesheet API - Database Setup
SQL Server Database Creation Script
================================================================

This script creates all necessary tables, indexes, constraints,
and sample data for the Ultimate Timesheet API system.

Author: Ultimate Timesheet Expert Team (50+ Years Experience)
Date: October 2025
Version: 3.0.0
Database: SQL Server 2017+
*/

-- ================================================================
-- DATABASE CREATION AND CONFIGURATION
-- ================================================================

-- Create database if it doesn't exist
IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = N'TimesheetDB')
BEGIN
    CREATE DATABASE TimesheetDB
    COLLATE SQL_Latin1_General_CP1_CI_AS;
    PRINT '✅ Database TimesheetDB created successfully';
END
ELSE
BEGIN
    PRINT '⚠️ Database TimesheetDB already exists';
END
GO

-- Switch to the Timesheet database
USE TimesheetDB;
GO

-- Configure database settings for optimal performance
ALTER DATABASE TimesheetDB SET RECOVERY SIMPLE;
ALTER DATABASE TimesheetDB SET AUTO_CLOSE OFF;
ALTER DATABASE TimesheetDB SET AUTO_SHRINK OFF;
ALTER DATABASE TimesheetDB SET AUTO_CREATE_STATISTICS ON;
ALTER DATABASE TimesheetDB SET AUTO_UPDATE_STATISTICS ON;
GO

PRINT '🎯 Switched to TimesheetDB database';

-- ================================================================
-- DROP EXISTING TABLES (FOR CLEAN INSTALLATION)
-- ================================================================

-- Drop tables in reverse dependency order
IF EXISTS (SELECT * FROM sys.objects WHERE name = 'ConversationSessions' AND type = 'U')
    DROP TABLE ConversationSessions;

IF EXISTS (SELECT * FROM sys.objects WHERE name = 'TimesheetDrafts' AND type = 'U')
    DROP TABLE TimesheetDrafts;

IF EXISTS (SELECT * FROM sys.objects WHERE name = 'OracleTimesheet' AND type = 'U')
    DROP TABLE OracleTimesheet;

IF EXISTS (SELECT * FROM sys.objects WHERE name = 'MarsTimesheet' AND type = 'U')
    DROP TABLE MarsTimesheet;

IF EXISTS (SELECT * FROM sys.objects WHERE name = 'ProjectCodes' AND type = 'U')
    DROP TABLE ProjectCodes;

IF EXISTS (SELECT * FROM sys.objects WHERE name = 'TaskCodes' AND type = 'U')
    DROP TABLE TaskCodes;

IF EXISTS (SELECT * FROM sys.objects WHERE name = 'SystemConfiguration' AND type = 'U')
    DROP TABLE SystemConfiguration;

IF EXISTS (SELECT * FROM sys.objects WHERE name = 'AuditLog' AND type = 'U')
    DROP TABLE AuditLog;

PRINT '🗑️ Existing tables dropped (if they existed)';

-- ================================================================
-- CORE SYSTEM TABLES
-- ================================================================

-- System Configuration Table
CREATE TABLE SystemConfiguration (
    ID INT IDENTITY(1,1) PRIMARY KEY,
    ConfigKey NVARCHAR(100) NOT NULL UNIQUE,
    ConfigValue NVARCHAR(500) NOT NULL,
    Description NVARCHAR(500),
    IsActive BIT DEFAULT 1,
    CreatedAt DATETIME2 DEFAULT GETDATE(),
    UpdatedAt DATETIME2 DEFAULT GETDATE(),
    CreatedBy NVARCHAR(255) DEFAULT 'SYSTEM'
);

-- Project Codes Master Table (Enhanced)
CREATE TABLE ProjectCodes (
    ID INT IDENTITY(1,1) PRIMARY KEY,
    ProjectCode NVARCHAR(50) NOT NULL UNIQUE,
    ProjectName NVARCHAR(255) NOT NULL,
    System NVARCHAR(20) NOT NULL CHECK (System IN ('Oracle', 'Mars', 'Both')),
    ClientName NVARCHAR(255),
    ProjectManager NVARCHAR(255),
    StartDate DATE,
    EndDate DATE,
    BudgetHours DECIMAL(10,2),
    Description NVARCHAR(1000),
    IsActive BIT DEFAULT 1,
    Priority NVARCHAR(20) DEFAULT 'Medium' CHECK (Priority IN ('Low', 'Medium', 'High', 'Critical')),
    CreatedAt DATETIME2 DEFAULT GETDATE(),
    UpdatedAt DATETIME2 DEFAULT GETDATE(),
    CreatedBy NVARCHAR(255) DEFAULT 'SYSTEM'
);

-- Task Codes Table (New)
CREATE TABLE TaskCodes (
    ID INT IDENTITY(1,1) PRIMARY KEY,
    TaskCode NVARCHAR(50) NOT NULL,
    TaskName NVARCHAR(255) NOT NULL,
    TaskDescription NVARCHAR(500),
    ProjectCodeID INT,
    System NVARCHAR(20) NOT NULL CHECK (System IN ('Oracle', 'Mars', 'Both')),
    EstimatedHours DECIMAL(8,2),
    IsActive BIT DEFAULT 1,
    CreatedAt DATETIME2 DEFAULT GETDATE(),
    UpdatedAt DATETIME2 DEFAULT GETDATE(),
    FOREIGN KEY (ProjectCodeID) REFERENCES ProjectCodes(ID)
);

-- ================================================================
-- TIMESHEET TABLES (ORACLE SYSTEM)
-- ================================================================

CREATE TABLE OracleTimesheet (
    ID BIGINT IDENTITY(1,1) PRIMARY KEY,
    UserEmail NVARCHAR(255) NOT NULL,
    EntryDate DATE NOT NULL,
    ProjectCode NVARCHAR(50) NOT NULL,
    TaskCode NVARCHAR(50),
    Hours DECIMAL(5,2) NOT NULL,
    Comments NVARCHAR(500),
    Status NVARCHAR(20) DEFAULT 'Draft',
    ApprovedBy NVARCHAR(255),
    ApprovedAt DATETIME2,
    SubmittedAt DATETIME2,
    CreatedAt DATETIME2 DEFAULT GETDATE(),
    UpdatedAt DATETIME2 DEFAULT GETDATE(),
    CreatedBy NVARCHAR(255),
    UpdatedBy NVARCHAR(255),

    -- Constraints
    CONSTRAINT CK_OracleTimesheet_Hours CHECK (Hours > 0 AND Hours <= 24),
    CONSTRAINT CK_OracleTimesheet_Status CHECK (Status IN ('Draft', 'Submitted', 'Approved', 'Rejected', 'Cancelled')),
    CONSTRAINT FK_OracleTimesheet_ProjectCode FOREIGN KEY (ProjectCode) REFERENCES ProjectCodes(ProjectCode)
);

-- ================================================================
-- TIMESHEET TABLES (MARS SYSTEM)
-- ================================================================

CREATE TABLE MarsTimesheet (
    ID BIGINT IDENTITY(1,1) PRIMARY KEY,
    UserEmail NVARCHAR(255) NOT NULL,
    EntryDate DATE NOT NULL,
    ProjectCode NVARCHAR(50) NOT NULL,
    TaskCode NVARCHAR(50),
    Hours DECIMAL(5,2) NOT NULL,
    Comments NVARCHAR(500),
    Status NVARCHAR(20) DEFAULT 'Draft',
    ApprovedBy NVARCHAR(255),
    ApprovedAt DATETIME2,
    SubmittedAt DATETIME2,
    CreatedAt DATETIME2 DEFAULT GETDATE(),
    UpdatedAt DATETIME2 DEFAULT GETDATE(),
    CreatedBy NVARCHAR(255),
    UpdatedBy NVARCHAR(255),

    -- Constraints
    CONSTRAINT CK_MarsTimesheet_Hours CHECK (Hours > 0 AND Hours <= 24),
    CONSTRAINT CK_MarsTimesheet_Status CHECK (Status IN ('Draft', 'Submitted', 'Approved', 'Rejected', 'Cancelled')),
    CONSTRAINT FK_MarsTimesheet_ProjectCode FOREIGN KEY (ProjectCode) REFERENCES ProjectCodes(ProjectCode)
);

-- ================================================================
-- SESSION AND CONVERSATION MANAGEMENT
-- ================================================================

-- Conversation Sessions Table (Enhanced)
CREATE TABLE ConversationSessions (
    SessionID NVARCHAR(50) PRIMARY KEY,
    UserEmail NVARCHAR(255) NOT NULL,
    SessionData NVARCHAR(MAX),
    ConversationPhase NVARCHAR(50),
    SystemsInProgress NVARCHAR(100),
    LastInteraction DATETIME2 DEFAULT GETDATE(),
    CreatedAt DATETIME2 DEFAULT GETDATE(),
    IPAddress NVARCHAR(50),
    UserAgent NVARCHAR(500),
    IsActive BIT DEFAULT 1
);

-- Timesheet Drafts Table (Enhanced)
CREATE TABLE TimesheetDrafts (
    DraftID NVARCHAR(50) PRIMARY KEY,
    UserEmail NVARCHAR(255) NOT NULL,
    DraftData NVARCHAR(MAX),
    TotalHours DECIMAL(8,2),
    SystemsUsed NVARCHAR(100),
    EntryCount INT DEFAULT 0,
    Status NVARCHAR(20) DEFAULT 'Draft' CHECK (Status IN ('Draft', 'Submitted', 'Cancelled')),
    CreatedAt DATETIME2 DEFAULT GETDATE(),
    UpdatedAt DATETIME2 DEFAULT GETDATE(),
    ExpiresAt DATETIME2 DEFAULT DATEADD(DAY, 7, GETDATE())
);

-- ================================================================
-- AUDIT AND LOGGING
-- ================================================================

-- Comprehensive Audit Log
CREATE TABLE AuditLog (
    ID BIGINT IDENTITY(1,1) PRIMARY KEY,
    TableName NVARCHAR(50) NOT NULL,
    RecordID NVARCHAR(50),
    Action NVARCHAR(20) NOT NULL CHECK (Action IN ('INSERT', 'UPDATE', 'DELETE', 'SELECT')),
    OldValues NVARCHAR(MAX),
    NewValues NVARCHAR(MAX),
    UserEmail NVARCHAR(255),
    IPAddress NVARCHAR(50),
    UserAgent NVARCHAR(500),
    Timestamp DATETIME2 DEFAULT GETDATE(),
    AdditionalInfo NVARCHAR(1000)
);

-- ================================================================
-- INDEXES FOR OPTIMAL PERFORMANCE
-- ================================================================

-- Oracle Timesheet Indexes
CREATE INDEX IX_OracleTimesheet_UserEmail_Date ON OracleTimesheet(UserEmail, EntryDate);
CREATE INDEX IX_OracleTimesheet_ProjectCode ON OracleTimesheet(ProjectCode);
CREATE INDEX IX_OracleTimesheet_Status ON OracleTimesheet(Status);
CREATE INDEX IX_OracleTimesheet_CreatedAt ON OracleTimesheet(CreatedAt);

-- Mars Timesheet Indexes
CREATE INDEX IX_MarsTimesheet_UserEmail_Date ON MarsTimesheet(UserEmail, EntryDate);
CREATE INDEX IX_MarsTimesheet_ProjectCode ON MarsTimesheet(ProjectCode);
CREATE INDEX IX_MarsTimesheet_Status ON MarsTimesheet(Status);
CREATE INDEX IX_MarsTimesheet_CreatedAt ON MarsTimesheet(CreatedAt);

-- Project Codes Indexes
CREATE INDEX IX_ProjectCodes_System ON ProjectCodes(System);
CREATE INDEX IX_ProjectCodes_IsActive ON ProjectCodes(IsActive);
CREATE INDEX IX_ProjectCodes_ProjectCode ON ProjectCodes(ProjectCode);

-- Session Management Indexes
CREATE INDEX IX_ConversationSessions_UserEmail ON ConversationSessions(UserEmail);
CREATE INDEX IX_ConversationSessions_LastInteraction ON ConversationSessions(LastInteraction);
CREATE INDEX IX_TimesheetDrafts_UserEmail ON TimesheetDrafts(UserEmail);
CREATE INDEX IX_TimesheetDrafts_CreatedAt ON TimesheetDrafts(CreatedAt);

-- Audit Log Indexes
CREATE INDEX IX_AuditLog_TableName_Timestamp ON AuditLog(TableName, Timestamp);
CREATE INDEX IX_AuditLog_UserEmail ON AuditLog(UserEmail);

PRINT '📊 Indexes created for optimal performance';

-- ================================================================
-- INSERT SYSTEM CONFIGURATION DATA
-- ================================================================

INSERT INTO SystemConfiguration (ConfigKey, ConfigValue, Description) VALUES
('API_VERSION', '3.0.0', 'Current API version'),
('EXPERTISE_LEVEL', '50+ Years', 'Professional expertise level'),
('MAX_HOURS_PER_DAY', '24.0', 'Maximum hours allowed per day'),
('MIN_HOURS_PER_ENTRY', '0.25', 'Minimum hours per timesheet entry'),
('DEFAULT_TIMESHEET_STATUS', 'Draft', 'Default status for new timesheet entries'),
('SESSION_TIMEOUT_HOURS', '24', 'Session timeout in hours'),
('DRAFT_EXPIRY_DAYS', '7', 'Days before draft entries expire'),
('SUPPORTED_SYSTEMS', 'Oracle,Mars', 'Comma-separated list of supported systems'),
('COMPANY_NAME', 'Ultimate Timesheet Solutions', 'Company name'),
('ADMIN_EMAIL', 'admin@ultimatetimesheet.com', 'Administrator email address');

-- ================================================================
-- INSERT PROJECT CODES DATA (COMPREHENSIVE)
-- ================================================================

-- Oracle System Projects
INSERT INTO ProjectCodes (ProjectCode, ProjectName, System, ClientName, ProjectManager, StartDate, EndDate, BudgetHours, Description, Priority) VALUES
-- Core Oracle Projects
('ORG-001', 'Oracle Core Development', 'Oracle', 'Internal IT', 'John Smith', '2024-01-01', '2024-12-31', 2000.00, 'Core Oracle database development and maintenance', 'High'),
('ORG-002', 'Oracle Database Maintenance', 'Oracle', 'Internal IT', 'Sarah Johnson', '2024-01-01', '2024-12-31', 1500.00, 'Regular database maintenance and optimization', 'Medium'),
('ORG-003', 'Oracle Integration Services', 'Oracle', 'External Client A', 'Mike Wilson', '2024-03-01', '2024-09-30', 800.00, 'Integration services for external client systems', 'High'),
('ORG-004', 'Oracle Security Implementation', 'Oracle', 'Security Team', 'Lisa Chen', '2024-02-01', '2024-08-31', 600.00, 'Security enhancements and compliance implementation', 'Critical'),
('ORG-005', 'Oracle Performance Optimization', 'Oracle', 'Internal IT', 'David Brown', '2024-04-01', '2024-10-31', 1000.00, 'Database performance tuning and optimization', 'High'),
('ORG-006', 'Oracle Backup & Recovery', 'Oracle', 'Internal IT', 'Jennifer Davis', '2024-01-01', '2024-12-31', 400.00, 'Backup and disaster recovery systems', 'Critical'),
('ORG-007', 'Oracle Analytics Platform', 'Oracle', 'Business Intelligence', 'Robert Taylor', '2024-05-01', '2024-11-30', 1200.00, 'Analytics and reporting platform development', 'Medium'),
('ORG-008', 'Oracle Cloud Migration', 'Oracle', 'Cloud Team', 'Amanda White', '2024-06-01', '2025-02-28', 2500.00, 'Migration of Oracle systems to cloud infrastructure', 'High'),

-- Mars System Projects  
('MRS-001', 'Mars Data Processing', 'Mars', 'Data Science Team', 'Chris Anderson', '2024-01-01', '2024-12-31', 1800.00, 'Advanced data processing and ETL operations', 'High'),
('MRS-002', 'Mars Analytics Platform', 'Mars', 'Analytics Team', 'Michelle Garcia', '2024-02-01', '2024-10-31', 1600.00, 'Business analytics and intelligence platform', 'High'),
('MRS-003', 'Mars Reporting Services', 'Mars', 'Business Operations', 'Kevin Martinez', '2024-01-01', '2024-12-31', 1200.00, 'Automated reporting and dashboard services', 'Medium'),
('MRS-004', 'Mars Machine Learning', 'Mars', 'AI/ML Team', 'Emily Rodriguez', '2024-03-01', '2024-12-31', 2200.00, 'Machine learning model development and deployment', 'Critical'),
('MRS-005', 'Mars Data Visualization', 'Mars', 'UX Team', 'Jason Lee', '2024-04-01', '2024-11-30', 800.00, 'Interactive data visualization tools', 'Medium'),
('MRS-006', 'Mars Real-time Processing', 'Mars', 'Engineering Team', 'Nicole Thompson', '2024-05-01', '2025-01-31', 1500.00, 'Real-time data streaming and processing', 'High'),
('MRS-007', 'Mars API Development', 'Mars', 'Development Team', 'Daniel Jackson', '2024-02-01', '2024-09-30', 1000.00, 'REST API development for Mars services', 'Medium'),
('MRS-008', 'Mars Mobile Application', 'Mars', 'Mobile Team', 'Rachel Kim', '2024-06-01', '2024-12-31', 1400.00, 'Mobile application for Mars data access', 'Medium'),

-- Common/Shared Projects
('CMN-001', 'Common Documentation', 'Both', 'Documentation Team', 'Steven Wilson', '2024-01-01', '2024-12-31', 600.00, 'Technical documentation and user guides', 'Medium'),
('CMN-002', 'Common Training', 'Both', 'Training Team', 'Laura Martinez', '2024-01-01', '2024-12-31', 800.00, 'Training materials and sessions for staff', 'Medium'),
('CMN-003', 'Common Testing', 'Both', 'QA Team', 'Mark Johnson', '2024-01-01', '2024-12-31', 1000.00, 'Quality assurance and testing procedures', 'High'),
('CMN-004', 'Common Architecture', 'Both', 'Architecture Team', 'Carol Davis', '2024-01-01', '2024-12-31', 1500.00, 'System architecture and design standards', 'Critical'),
('CMN-005', 'Common Security', 'Both', 'Security Team', 'Brian Taylor', '2024-01-01', '2024-12-31', 1200.00, 'Security policies and implementation', 'Critical'),
('CMN-006', 'Common Infrastructure', 'Both', 'Infrastructure Team', 'Helen Brown', '2024-01-01', '2024-12-31', 2000.00, 'Shared infrastructure and platform services', 'High'),
('CMN-007', 'Common Monitoring', 'Both', 'DevOps Team', 'Frank Miller', '2024-01-01', '2024-12-31', 800.00, 'System monitoring and alerting', 'High'),
('CMN-008', 'Common Research', 'Both', 'R&D Team', 'Grace Lee', '2024-01-01', '2024-12-31', 1000.00, 'Research and development initiatives', 'Medium');

-- ================================================================
-- INSERT TASK CODES DATA
-- ================================================================

-- Development Tasks
INSERT INTO TaskCodes (TaskCode, TaskName, TaskDescription, System, EstimatedHours) VALUES
('DEV-001', 'Feature Development', 'New feature development and implementation', 'Both', 40.0),
('DEV-002', 'Bug Fixing', 'Bug investigation and resolution', 'Both', 8.0),
('DEV-003', 'Code Review', 'Code review and quality assurance', 'Both', 4.0),
('DEV-004', 'Unit Testing', 'Writing and executing unit tests', 'Both', 16.0),
('DEV-005', 'Integration Testing', 'Integration testing and validation', 'Both', 12.0),
('DEV-006', 'Performance Optimization', 'Performance tuning and optimization', 'Both', 20.0),
('DEV-007', 'Database Design', 'Database schema design and modeling', 'Both', 24.0),
('DEV-008', 'API Development', 'REST API development and testing', 'Both', 32.0),

-- Maintenance Tasks
('MNT-001', 'System Maintenance', 'Regular system maintenance activities', 'Both', 8.0),
('MNT-002', 'Database Maintenance', 'Database cleanup and optimization', 'Both', 12.0),
('MNT-003', 'Security Updates', 'Security patches and updates', 'Both', 6.0),
('MNT-004', 'Performance Monitoring', 'System performance monitoring', 'Both', 4.0),
('MNT-005', 'Backup Operations', 'Data backup and recovery operations', 'Both', 4.0),

-- Documentation Tasks
('DOC-001', 'Technical Documentation', 'Writing technical documentation', 'Both', 16.0),
('DOC-002', 'User Guide Creation', 'Creating user guides and manuals', 'Both', 20.0),
('DOC-003', 'API Documentation', 'REST API documentation', 'Both', 12.0),
('DOC-004', 'Process Documentation', 'Business process documentation', 'Both', 8.0),

-- Analysis Tasks
('ANA-001', 'Requirements Analysis', 'Business requirements analysis', 'Both', 24.0),
('ANA-002', 'Data Analysis', 'Data analysis and reporting', 'Mars', 16.0),
('ANA-003', 'Performance Analysis', 'System performance analysis', 'Both', 12.0),
('ANA-004', 'Security Analysis', 'Security assessment and analysis', 'Both', 16.0),

-- Training Tasks
('TRN-001', 'Training Preparation', 'Preparing training materials', 'Both', 20.0),
('TRN-002', 'Training Delivery', 'Conducting training sessions', 'Both', 8.0),
('TRN-003', 'Knowledge Transfer', 'Knowledge transfer sessions', 'Both', 12.0),

-- Meeting Tasks
('MTG-001', 'Project Meetings', 'Project status and planning meetings', 'Both', 2.0),
('MTG-002', 'Client Meetings', 'Client consultation meetings', 'Both', 3.0),
('MTG-003', 'Team Meetings', 'Regular team meetings and standups', 'Both', 1.0),
('MTG-004', 'Architecture Reviews', 'Architecture and design reviews', 'Both', 4.0);

-- ================================================================
-- INSERT SAMPLE TIMESHEET DATA
-- ================================================================

-- Sample Oracle Timesheet Entries
INSERT INTO OracleTimesheet (UserEmail, EntryDate, ProjectCode, TaskCode, Hours, Comments, Status, SubmittedAt) VALUES
('demo.user@company.com', '2024-10-01', 'ORG-001', 'DEV-001', 8.0, 'Implemented new user authentication features', 'Submitted', '2024-10-01 17:30:00'),
('demo.user@company.com', '2024-10-02', 'ORG-001', 'DEV-002', 6.5, 'Fixed critical bugs in login system', 'Approved', '2024-10-02 17:15:00'),
('demo.user@company.com', '2024-10-03', 'ORG-003', 'DEV-008', 7.0, 'Developed REST API endpoints for integration', 'Submitted', '2024-10-03 18:00:00'),
('demo.user@company.com', '2024-10-04', 'CMN-003', 'DEV-004', 8.0, 'Created comprehensive unit tests', 'Approved', '2024-10-04 17:45:00'),
('test.user@company.com', '2024-10-01', 'ORG-002', 'MNT-002', 4.0, 'Database performance optimization', 'Submitted', '2024-10-01 16:30:00'),
('test.user@company.com', '2024-10-01', 'CMN-001', 'DOC-001', 4.0, 'Updated technical documentation', 'Submitted', '2024-10-01 16:30:00'),
('test.user@company.com', '2024-10-02', 'ORG-005', 'MNT-004', 8.0, 'Performance monitoring and analysis', 'Approved', '2024-10-02 17:00:00');

-- Sample Mars Timesheet Entries
INSERT INTO MarsTimesheet (UserEmail, EntryDate, ProjectCode, TaskCode, Hours, Comments, Status, SubmittedAt) VALUES
('demo.user@company.com', '2024-10-01', 'MRS-001', 'ANA-002', 4.0, 'Data analysis for customer behavior patterns', 'Submitted', '2024-10-01 17:30:00'),
('demo.user@company.com', '2024-10-03', 'MRS-002', 'DEV-001', 8.0, 'Developed new analytics dashboard features', 'Approved', '2024-10-03 18:00:00'),
('demo.user@company.com', '2024-10-04', 'MRS-004', 'DEV-001', 6.0, 'Machine learning model development', 'Submitted', '2024-10-04 17:45:00'),
('test.user@company.com', '2024-10-02', 'MRS-003', 'DEV-008', 8.0, 'Enhanced reporting API endpoints', 'Approved', '2024-10-02 17:00:00'),
('test.user@company.com', '2024-10-03', 'CMN-005', 'MNT-003', 4.0, 'Applied security updates to Mars systems', 'Submitted', '2024-10-03 16:45:00'),
('admin@company.com', '2024-10-01', 'MRS-006', 'DEV-001', 8.0, 'Real-time data processing pipeline development', 'Approved', '2024-10-01 18:00:00'),
('admin@company.com', '2024-10-02', 'CMN-007', 'MNT-004', 6.0, 'System monitoring setup and configuration', 'Approved', '2024-10-02 17:30:00');

-- ================================================================
-- INSERT SAMPLE CONVERSATION SESSIONS
-- ================================================================

INSERT INTO ConversationSessions (SessionID, UserEmail, SessionData, ConversationPhase, SystemsInProgress, IPAddress, UserAgent) VALUES
('session_demo_001', 'demo.user@company.com', '{"current_entries": [], "conversation_phase": "gathering", "last_system": "Oracle"}', 'gathering', 'Oracle', '192.168.1.100', 'UltimateTimesheetApp/3.0'),
('session_test_001', 'test.user@company.com', '{"current_entries": [], "conversation_phase": "completed", "last_system": "Mars"}', 'completed', 'Mars', '192.168.1.101', 'UltimateTimesheetApp/3.0'),
('session_admin_001', 'admin@company.com', '{"current_entries": [], "conversation_phase": "gathering", "last_system": "Both"}', 'gathering', 'Oracle,Mars', '192.168.1.102', 'UltimateTimesheetApp/3.0');

-- ================================================================
-- INSERT SAMPLE DRAFT DATA
-- ================================================================

INSERT INTO TimesheetDrafts (DraftID, UserEmail, DraftData, TotalHours, SystemsUsed, EntryCount, Status) VALUES
('draft_demo_001', 'demo.user@company.com', '{"entries": [{"system": "Oracle", "project_code": "ORG-001", "hours": 8.0, "date": "2024-10-05"}]}', 8.0, 'Oracle', 1, 'Draft'),
('draft_test_001', 'test.user@company.com', '{"entries": [{"system": "Mars", "project_code": "MRS-001", "hours": 4.0, "date": "2024-10-05"}, {"system": "Oracle", "project_code": "CMN-001", "hours": 4.0, "date": "2024-10-05"}]}', 8.0, 'Oracle,Mars', 2, 'Draft');

-- ================================================================
-- CREATE VIEWS FOR REPORTING
-- ================================================================

-- Comprehensive Timesheet View
CREATE VIEW vw_AllTimesheetEntries AS
SELECT 
    'Oracle' as System,
    ID,
    UserEmail,
    EntryDate,
    ProjectCode,
    TaskCode,
    Hours,
    Comments,
    Status,
    ApprovedBy,
    ApprovedAt,
    SubmittedAt,
    CreatedAt,
    UpdatedAt
FROM OracleTimesheet
UNION ALL
SELECT 
    'Mars' as System,
    ID,
    UserEmail,
    EntryDate,
    ProjectCode,
    TaskCode,
    Hours,
    Comments,
    Status,
    ApprovedBy,
    ApprovedAt,
    SubmittedAt,
    CreatedAt,
    UpdatedAt
FROM MarsTimesheet;

-- Project Summary View
CREATE VIEW vw_ProjectSummary AS
SELECT 
    p.ProjectCode,
    p.ProjectName,
    p.System,
    p.ClientName,
    p.ProjectManager,
    p.BudgetHours,
    COALESCE(oracle_hours.TotalHours, 0) + COALESCE(mars_hours.TotalHours, 0) as ActualHours,
    p.BudgetHours - (COALESCE(oracle_hours.TotalHours, 0) + COALESCE(mars_hours.TotalHours, 0)) as RemainingHours,
    p.StartDate,
    p.EndDate,
    p.IsActive
FROM ProjectCodes p
LEFT JOIN (
    SELECT ProjectCode, SUM(Hours) as TotalHours
    FROM OracleTimesheet 
    WHERE Status IN ('Submitted', 'Approved')
    GROUP BY ProjectCode
) oracle_hours ON p.ProjectCode = oracle_hours.ProjectCode
LEFT JOIN (
    SELECT ProjectCode, SUM(Hours) as TotalHours
    FROM MarsTimesheet 
    WHERE Status IN ('Submitted', 'Approved')
    GROUP BY ProjectCode
) mars_hours ON p.ProjectCode = mars_hours.ProjectCode;

-- User Activity Summary View
CREATE VIEW vw_UserActivitySummary AS
SELECT 
    UserEmail,
    COUNT(*) as TotalEntries,
    SUM(Hours) as TotalHours,
    AVG(Hours) as AverageHours,
    MIN(EntryDate) as FirstEntry,
    MAX(EntryDate) as LastEntry,
    COUNT(DISTINCT ProjectCode) as ProjectsWorked
FROM vw_AllTimesheetEntries
WHERE Status IN ('Submitted', 'Approved')
GROUP BY UserEmail;

-- ================================================================
-- CREATE STORED PROCEDURES
-- ================================================================

-- Stored Procedure: Get User Timesheet with Filters
CREATE PROCEDURE sp_GetUserTimesheet
    @UserEmail NVARCHAR(255),
    @System NVARCHAR(20) = NULL,
    @StartDate DATE = NULL,
    @EndDate DATE = NULL,
    @Status NVARCHAR(20) = NULL
AS
BEGIN
    SET NOCOUNT ON;

    SELECT 
        System,
        ID,
        UserEmail,
        EntryDate,
        ProjectCode,
        TaskCode,
        Hours,
        Comments,
        Status,
        ApprovedBy,
        ApprovedAt,
        SubmittedAt,
        CreatedAt,
        UpdatedAt
    FROM vw_AllTimesheetEntries
    WHERE UserEmail = @UserEmail
        AND (@System IS NULL OR System = @System)
        AND (@StartDate IS NULL OR EntryDate >= @StartDate)
        AND (@EndDate IS NULL OR EntryDate <= @EndDate)
        AND (@Status IS NULL OR Status = @Status)
    ORDER BY EntryDate DESC, CreatedAt DESC;
END;

-- Stored Procedure: Get Project Utilization Report
CREATE PROCEDURE sp_GetProjectUtilization
    @StartDate DATE = NULL,
    @EndDate DATE = NULL
AS
BEGIN
    SET NOCOUNT ON;

    SELECT 
        p.ProjectCode,
        p.ProjectName,
        p.System,
        p.ClientName,
        p.BudgetHours,
        s.ActualHours,
        s.RemainingHours,
        CASE 
            WHEN p.BudgetHours > 0 THEN (s.ActualHours / p.BudgetHours) * 100 
            ELSE 0 
        END as UtilizationPercentage,
        COUNT(t.ID) as TotalEntries,
        COUNT(DISTINCT t.UserEmail) as UniqueUsers
    FROM vw_ProjectSummary s
    JOIN ProjectCodes p ON s.ProjectCode = p.ProjectCode
    LEFT JOIN vw_AllTimesheetEntries t ON p.ProjectCode = t.ProjectCode 
        AND t.Status IN ('Submitted', 'Approved')
        AND (@StartDate IS NULL OR t.EntryDate >= @StartDate)
        AND (@EndDate IS NULL OR t.EntryDate <= @EndDate)
    WHERE p.IsActive = 1
    GROUP BY p.ProjectCode, p.ProjectName, p.System, p.ClientName, p.BudgetHours, s.ActualHours, s.RemainingHours
    ORDER BY UtilizationPercentage DESC;
END;

-- ================================================================
-- CREATE TRIGGERS FOR AUDIT LOGGING
-- ================================================================

-- Audit Trigger for Oracle Timesheet
CREATE TRIGGER tr_OracleTimesheet_Audit
ON OracleTimesheet
AFTER INSERT, UPDATE, DELETE
AS
BEGIN
    SET NOCOUNT ON;

    -- Insert operations
    INSERT INTO AuditLog (TableName, RecordID, Action, NewValues, UserEmail, AdditionalInfo)
    SELECT 
        'OracleTimesheet',
        CAST(i.ID as NVARCHAR(50)),
        'INSERT',
        (SELECT * FROM inserted i2 WHERE i2.ID = i.ID FOR JSON AUTO),
        i.UserEmail,
        'Timesheet entry created'
    FROM inserted i;

    -- Update operations
    INSERT INTO AuditLog (TableName, RecordID, Action, OldValues, NewValues, UserEmail, AdditionalInfo)
    SELECT 
        'OracleTimesheet',
        CAST(i.ID as NVARCHAR(50)),
        'UPDATE',
        (SELECT * FROM deleted d WHERE d.ID = i.ID FOR JSON AUTO),
        (SELECT * FROM inserted i2 WHERE i2.ID = i.ID FOR JSON AUTO),
        i.UserEmail,
        'Timesheet entry updated'
    FROM inserted i
    INNER JOIN deleted d ON i.ID = d.ID;

    -- Delete operations
    INSERT INTO AuditLog (TableName, RecordID, Action, OldValues, UserEmail, AdditionalInfo)
    SELECT 
        'OracleTimesheet',
        CAST(d.ID as NVARCHAR(50)),
        'DELETE',
        (SELECT * FROM deleted d2 WHERE d2.ID = d.ID FOR JSON AUTO),
        NULL,
        d.UserEmail,
        'Timesheet entry deleted'
    FROM deleted d;
END;

-- Audit Trigger for Mars Timesheet
CREATE TRIGGER tr_MarsTimesheet_Audit
ON MarsTimesheet
AFTER INSERT, UPDATE, DELETE
AS
BEGIN
    SET NOCOUNT ON;

    -- Insert operations
    INSERT INTO AuditLog (TableName, RecordID, Action, NewValues, UserEmail, AdditionalInfo)
    SELECT 
        'MarsTimesheet',
        CAST(i.ID as NVARCHAR(50)),
        'INSERT',
        (SELECT * FROM inserted i2 WHERE i2.ID = i.ID FOR JSON AUTO),
        i.UserEmail,
        'Timesheet entry created'
    FROM inserted i;

    -- Update operations
    INSERT INTO AuditLog (TableName, RecordID, Action, OldValues, NewValues, UserEmail, AdditionalInfo)
    SELECT 
        'MarsTimesheet',
        CAST(i.ID as NVARCHAR(50)),
        'UPDATE',
        (SELECT * FROM deleted d WHERE d.ID = i.ID FOR JSON AUTO),
        (SELECT * FROM inserted i2 WHERE i2.ID = i.ID FOR JSON AUTO),
        i.UserEmail,
        'Timesheet entry updated'
    FROM inserted i
    INNER JOIN deleted d ON i.ID = d.ID;

    -- Delete operations
    INSERT INTO AuditLog (TableName, RecordID, Action, OldValues, UserEmail, AdditionalInfo)
    SELECT 
        'MarsTimesheet',
        CAST(d.ID as NVARCHAR(50)),
        'DELETE',
        (SELECT * FROM deleted d2 WHERE d2.ID = d.ID FOR JSON AUTO),
        NULL,
        d.UserEmail,
        'Timesheet entry deleted'
    FROM deleted d;
END;

-- ================================================================
-- FINAL VERIFICATION AND STATISTICS
-- ================================================================

PRINT '📊 Database setup completed successfully!';
PRINT '================================';
PRINT 'Tables created: ' + CAST((SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE') AS VARCHAR(10));
PRINT 'Views created: ' + CAST((SELECT COUNT(*) FROM INFORMATION_SCHEMA.VIEWS) AS VARCHAR(10));
PRINT 'Indexes created: ' + CAST((SELECT COUNT(*) FROM sys.indexes WHERE is_primary_key = 0 AND is_unique_constraint = 0) AS VARCHAR(10));
PRINT 'Stored procedures: ' + CAST((SELECT COUNT(*) FROM INFORMATION_SCHEMA.ROUTINES WHERE ROUTINE_TYPE = 'PROCEDURE') AS VARCHAR(10));
PRINT 'Triggers created: ' + CAST((SELECT COUNT(*) FROM sys.triggers) AS VARCHAR(10));

-- Show sample data counts
SELECT 'ProjectCodes' as TableName, COUNT(*) as RecordCount FROM ProjectCodes
UNION ALL
SELECT 'TaskCodes', COUNT(*) FROM TaskCodes
UNION ALL
SELECT 'OracleTimesheet', COUNT(*) FROM OracleTimesheet
UNION ALL
SELECT 'MarsTimesheet', COUNT(*) FROM MarsTimesheet
UNION ALL
SELECT 'ConversationSessions', COUNT(*) FROM ConversationSessions
UNION ALL
SELECT 'TimesheetDrafts', COUNT(*) FROM TimesheetDrafts
UNION ALL
SELECT 'SystemConfiguration', COUNT(*) FROM SystemConfiguration;

PRINT '================================';
PRINT '🎯 Ultimate Timesheet Database is ready!';
PRINT '✅ All tables, indexes, and sample data created';
PRINT '✅ Audit logging enabled';
PRINT '✅ Views and stored procedures available';
PRINT '🚀 Ready for Ultimate Expert Timesheet API!';
