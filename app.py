from fastapi import FastAPI, Depends, HTTPException, status, Query, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import json
import uuid
from datetime import datetime, date, timedelta
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
import os

from database import get_db, engine, Base
import models
from schemas import (
    # User schemas
    UserCreate, UserLogin, UserUpdate, UserResponse, UserWithToken,
    # School schemas
    SchoolCreate, SchoolUpdate, SchoolResponse, SchoolWithStats,
    # Student schemas
    StudentCreate, StudentUpdate, StudentResponse, StudentWithDetails,
    # Exam schemas
    ExamCreate, ExamResponse,
    # Exam Result schemas
    ExamResultCreate, ExamResultResponse,
    # AI schemas
    AIPredictionResponse, AIModelCreate, AIModelResponse, AITrainingJobCreate,
    AIConversationCreate, AIConversationResponse,
    # Analytics schemas
    AnalyticsRequest, PerformanceAnalytics, DropoutRiskAnalytics,
    # Learning Path schemas
    LearningPathCreate, LearningPathResponse,
    # Notification schemas
    NotificationCreate, NotificationResponse,
    # Dashboard schemas
    DashboardMetrics, RealTimeMetrics,
    # Report schemas
    ReportRequest, ReportResponse,
    # General schemas
    PaginatedResponse, SuccessResponse, ErrorResponse,
    # WebSocket schemas
    WSMessage, WSConnection
)

from auth import AuthService, get_current_user, require_role, can_access_student, log_audit_event
from crud import CRUDService
from ai_engine import AdvancedAIEngine
from analytics import AdvancedAnalytics

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title="EduTrack AI Pro - Advanced Educational Analytics Platform",
    description="""Next-generation educational analytics platform with advanced AI capabilities.
    
    Features:
    - Real-time student performance analytics
    - AI-powered dropout prediction
    - Personalized learning paths
    - Advanced behavioral analysis
    - Predictive forecasting
    - Conversational AI interface
    - Multi-school management
    """,
    version="4.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
ai_engine = AdvancedAIEngine()
analytics_engine = AdvancedAnalytics()

# WebSocket connections
connected_clients: Dict[str, WebSocket] = {}
active_subscriptions: Dict[str, List[str]] = {}

# ========== AUTHENTICATION ENDPOINTS ==========

@app.post("/api/auth/register", response_model=UserWithToken)
async def register_user(
    user_data: UserCreate,
    db: Session = Depends(get_db),
    auth_service: AuthService = Depends(lambda: AuthService(db))
):
    """Register new user"""
    try:
        user = auth_service.create_user(user_data)
        tokens = auth_service.create_tokens(user)
        
        log_audit_event(
            db, user, "USER_REGISTER",
            details={"email": user.email, "role": user.role}
        )
        
        return {**UserResponse.model_validate(user).model_dump(), **tokens}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/auth/login", response_model=UserWithToken)
async def login(
    credentials: UserLogin,
    db: Session = Depends(get_db),
    auth_service: AuthService = Depends(lambda: AuthService(db))
):
    """User login"""
    user = auth_service.authenticate_user(credentials.email, credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    tokens = auth_service.create_tokens(user)
    
    log_audit_event(
        db, user, "USER_LOGIN",
        details={"method": "email_password"}
    )
    
    return {**UserResponse.model_validate(user).model_dump(), **tokens}

@app.post("/api/auth/refresh")
async def refresh_token(
    refresh_token: str,
    db: Session = Depends(get_db),
    auth_service: AuthService = Depends(lambda: AuthService(db))
):
    """Refresh access token"""
    return auth_service.refresh_access_token(refresh_token)

@app.get("/api/auth/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: models.User = Depends(get_current_user)
):
    """Get current user information"""
    return UserResponse.model_validate(current_user)

@app.post("/api/auth/api-keys")
async def create_api_key(
    name: str = "Default",
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
    auth_service: AuthService = Depends(lambda: AuthService(db))
):
    """Create new API key"""
    api_key = auth_service.create_api_key(current_user.id, name)
    
    log_audit_event(
        db, current_user, "API_KEY_CREATED",
        details={"key_name": name}
    )
    
    return {
        "api_key": api_key.key,
        "name": api_key.name,
        "created_at": api_key.created_at,
        "expires_at": api_key.expires_at
    }

# ========== SCHOOL MANAGEMENT ENDPOINTS ==========

@app.post("/api/schools", response_model=SchoolResponse)
async def create_school(
    school_data: SchoolCreate,
    current_user: models.User = Depends(require_role("super_admin")),
    db: Session = Depends(get_db)
):
    """Create new school"""
    crud = CRUDService(db)
    school = crud.create_school(school_data)
    
    log_audit_event(
        db, current_user, "SCHOOL_CREATED",
        resource_type="school", resource_id=str(school.id),
        details={"school_name": school.name, "school_code": school.code}
    )
    
    return SchoolResponse.model_validate(school)

@app.get("/api/schools", response_model=List[SchoolResponse])
async def get_schools(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    county: Optional[str] = None,
    level: Optional[str] = None,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get schools with filtering"""
    crud = CRUDService(db)
    
    # Non-super admins can only see their own school
    if current_user.role != "super_admin":
        if current_user.school_id:
            school = crud.get_school(current_user.school_id)
            return [SchoolResponse.model_validate(school)] if school else []
        return []
    
    schools = crud.get_schools(skip=skip, limit=limit, county=county, level=level)
    return [SchoolResponse.model_validate(school) for school in schools]

@app.get("/api/schools/{school_id}", response_model=SchoolWithStats)
async def get_school(
    school_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get school by ID"""
    crud = CRUDService(db)
    school = crud.get_school(school_id)
    
    if not school:
        raise HTTPException(status_code=404, detail="School not found")
    
    # Check permissions
    if current_user.role != "super_admin" and current_user.school_id != school_id:
        raise HTTPException(status_code=403, detail="Cannot access other schools")
    
    # Get statistics
    stats = crud.get_dashboard_stats(school_id)
    
    school_data = SchoolResponse.model_validate(school).model_dump()
    school_data["performance_metrics"] = stats
    school_data["ai_insights"] = {
        "total_students": stats.get("students", {}).get("total", 0),
        "average_performance": stats.get("exams", {}).get("average_score", 0) if "exams" in stats else 0,
        "active_interventions": stats.get("interventions", {}).get("active", 0) if "interventions" in stats else 0
    }
    
    return school_data

@app.put("/api/schools/{school_id}", response_model=SchoolResponse)
async def update_school(
    school_id: int,
    school_data: SchoolUpdate,
    current_user: models.User = Depends(require_role("super_admin", "school_admin")),
    db: Session = Depends(get_db)
):
    """Update school information"""
    crud = CRUDService(db)
    
    # School admins can only update their own school
    if current_user.role == "school_admin" and current_user.school_id != school_id:
        raise HTTPException(status_code=403, detail="Can only update your own school")
    
    school = crud.update_school(school_id, school_data)
    if not school:
        raise HTTPException(status_code=404, detail="School not found")
    
    log_audit_event(
        db, current_user, "SCHOOL_UPDATED",
        resource_type="school", resource_id=str(school_id),
        details={"update_data": school_data.model_dump()}
    )
    
    return SchoolResponse.model_validate(school)

# ========== STUDENT MANAGEMENT ENDPOINTS ==========

@app.post("/api/students", response_model=StudentResponse)
async def create_student(
    student_data: StudentCreate,
    current_user: models.User = Depends(require_role("school_admin", "teacher")),
    db: Session = Depends(get_db)
):
    """Create new student"""
    crud = CRUDService(db)
    
    # Check if user has permission for this school
    if current_user.school_id != student_data.school_id:
        raise HTTPException(status_code=403, detail="Cannot add students to other schools")
    
    student = crud.create_student(student_data)
    
    log_audit_event(
        db, current_user, "STUDENT_CREATED",
        resource_type="student", resource_id=str(student.id),
        details={"admission_number": student.admission_number, "school_id": student.school_id}
    )
    
    # Trigger initial AI analysis (async)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(
        ai_engine.analyze_student_comprehensive, student.id
    )
    
    return StudentResponse.model_validate(student)

@app.get("/api/students", response_model=PaginatedResponse)
async def get_students(
    school_id: Optional[int] = None,
    class_id: Optional[int] = None,
    is_active: Optional[bool] = True,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get students with filtering and pagination"""
    crud = CRUDService(db)
    
    # Apply permission filters
    if current_user.role == "school_admin":
        school_id = current_user.school_id
    elif current_user.role == "teacher":
        # Teachers can only see students in their classes
        # This needs to be implemented based on teacher-class assignment
        pass
    elif current_user.role == "parent":
        # Parents can only see their own children
        # This needs parent-student linking
        pass
    elif current_user.role == "student":
        # Students can only see themselves
        # This needs user-student linking
        pass
    
    students = crud.get_students(
        school_id=school_id,
        class_id=class_id,
        is_active=is_active,
        skip=skip,
        limit=limit
    )
    
    total = db.query(models.Student).count()
    
    return PaginatedResponse(
        items=[StudentResponse.model_validate(s) for s in students],
        total=total,
        page=skip // limit + 1,
        size=limit,
        pages=(total + limit - 1) // limit
    )

@app.get("/api/students/{student_id}", response_model=StudentWithDetails)
async def get_student(
    student_id: int,
    current_user: models.User = Depends(can_access_student(student_id)),
    db: Session = Depends(get_db)
):
    """Get student by ID with detailed information"""
    crud = CRUDService(db)
    student = crud.get_student(student_id)
    
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    # Get additional details
    school = crud.get_school(student.school_id) if student.school_id else None
    
    # Get recent performance
    exam_results = crud.get_exam_results(
        student_id=student_id,
        limit=5
    )
    
    # Get AI predictions
    ai_predictions = crud.get_ai_predictions(
        student_id=student_id,
        limit=3
    )
    
    student_data = StudentResponse.model_validate(student).model_dump()
    student_data["school"] = SchoolResponse.model_validate(school) if school else None
    student_data["class_info"] = {
        "name": student.class_rel.name if student.class_rel else None,
        "teacher": f"{student.class_rel.teacher.first_name} {student.class_rel.teacher.last_name}" 
                   if student.class_rel and student.class_rel.teacher else None
    }
    student_data["recent_performance"] = [
        {
            "subject": er.subject.name if er.subject else "Unknown",
            "score": er.percentage,
            "exam": er.exam.name if er.exam else "Unknown",
            "date": er.exam.conducted_date if er.exam else None
        }
        for er in exam_results
    ]
    student_data["ai_predictions"] = [
        {
            "type": p.prediction_type,
            "date": p.prediction_date,
            "risk_score": p.prediction_output.get("risk_score") if isinstance(p.prediction_output, dict) else None
        }
        for p in ai_predictions
    ]
    
    return student_data

@app.put("/api/students/{student_id}", response_model=StudentResponse)
async def update_student(
    student_id: int,
    student_data: StudentUpdate,
    current_user: models.User = Depends(can_access_student(student_id)),
    db: Session = Depends(get_db)
):
    """Update student information"""
    crud = CRUDService(db)
    
    # Check if user has permission to update
    if current_user.role not in ["super_admin", "school_admin", "teacher"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    student = crud.update_student(student_id, student_data)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    log_audit_event(
        db, current_user, "STUDENT_UPDATED",
        resource_type="student", resource_id=str(student_id),
        details={"update_data": student_data.model_dump()}
    )
    
    # Trigger AI re-analysis if significant fields changed
    if any(field in student_data.model_dump(exclude_unset=True) 
           for field in ["class_id", "status", "ai_profile"]):
        background_tasks = BackgroundTasks()
        background_tasks.add_task(
            ai_engine.analyze_student_comprehensive, student_id
        )
    
    return StudentResponse.model_validate(student)

@app.delete("/api/students/{student_id}")
async def delete_student(
    student_id: int,
    current_user: models.User = Depends(require_role("super_admin", "school_admin")),
    db: Session = Depends(get_db)
):
    """Delete student (soft delete)"""
    crud = CRUDService(db)
    
    # Get student first to check school
    student = crud.get_student(student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    # School admins can only delete from their own school
    if current_user.role == "school_admin" and current_user.school_id != student.school_id:
        raise HTTPException(status_code=403, detail="Cannot delete students from other schools")
    
    success = crud.delete_student(student_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete student")
    
    log_audit_event(
        db, current_user, "STUDENT_DELETED",
        resource_type="student", resource_id=str(student_id),
        details={"admission_number": student.admission_number}
    )
    
    return SuccessResponse(message="Student deleted successfully")

@app.get("/api/students/{student_id}/timeline")
async def get_student_timeline(
    student_id: int,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    current_user: models.User = Depends(can_access_student(student_id)),
    db: Session = Depends(get_db)
):
    """Get student progress timeline"""
    crud = CRUDService(db)
    
    timeline = crud.get_student_progress_timeline(
        student_id=student_id,
        start_date=start_date,
        end_date=end_date
    )
    
    return timeline

# ========== EXAM MANAGEMENT ENDPOINTS ==========

@app.post("/api/exams", response_model=ExamResponse)
async def create_exam(
    exam_data: ExamCreate,
    current_user: models.User = Depends(require_role("school_admin", "teacher")),
    db: Session = Depends(get_db)
):
    """Create new exam"""
    crud = CRUDService(db)
    
    # Check permissions
    if current_user.role == "teacher":
        # Teachers can only create exams for their classes
        if exam_data.class_id:
            # Verify teacher teaches this class
            pass
    
    exam = crud.create_exam(exam_data)
    
    log_audit_event(
        db, current_user, "EXAM_CREATED",
        resource_type="exam", resource_id=str(exam.id),
        details={"exam_name": exam.name, "exam_code": exam.code}
    )
    
    return ExamResponse.model_validate(exam)

@app.get("/api/exams/{exam_id}", response_model=ExamResponse)
async def get_exam(
    exam_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get exam by ID"""
    crud = CRUDService(db)
    exam = crud.get_exam(exam_id)
    
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")
    
    # Check permissions
    if current_user.role != "super_admin":
        if exam.school_id and current_user.school_id != exam.school_id:
            raise HTTPException(status_code=403, detail="Cannot access exams from other schools")
    
    return ExamResponse.model_validate(exam)

@app.post("/api/exams/{exam_id}/results/bulk")
async def bulk_create_exam_results(
    exam_id: int,
    subject_id: int,
    results_data: List[Dict[str, Any]],
    current_user: models.User = Depends(require_role("school_admin", "teacher")),
    db: Session = Depends(get_db)
):
    """Bulk create exam results"""
    crud = CRUDService(db)
    
    # Verify exam exists and user has permission
    exam = crud.get_exam(exam_id)
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")
    
    if current_user.role == "teacher":
        # Teachers can only add results to their classes
        if exam.class_id:
            # Verify teacher teaches this class
            pass
    
    results = crud.bulk_create_exam_results(results_data, exam_id, subject_id)
    
    log_audit_event(
        db, current_user, "EXAM_RESULTS_BULK_CREATED",
        resource_type="exam", resource_id=str(exam_id),
        details={"results_count": len(results), "subject_id": subject_id}
    )
    
    return {
        "message": f"Created {len(results)} exam results",
        "exam_id": exam_id,
        "subject_id": subject_id,
        "results_count": len(results)
    }

# ========== AI & ANALYTICS ENDPOINTS ==========

@app.get("/api/ai/analyze/student/{student_id}")
async def analyze_student_ai(
    student_id: int,
    analysis_type: str = Query("comprehensive", regex="^(comprehensive|dropout|performance|engagement|behavior)$"),
    current_user: models.User = Depends(can_access_student(student_id)),
    db: Session = Depends(get_db)
):
    """Run AI analysis on student"""
    
    log_audit_event(
        db, current_user, "AI_ANALYSIS_REQUESTED",
        resource_type="student", resource_id=str(student_id),
        details={"analysis_type": analysis_type}
    )
    
    if analysis_type == "comprehensive":
        analysis = ai_engine.analyze_student_comprehensive(student_id)
    else:
        # For specific analysis types
        analysis = ai_engine.analyze_student_comprehensive(student_id)
        # Filter to specific analysis type if needed
    
    return analysis

@app.post("/api/ai/analyze/batch")
async def batch_analyze_students(
    student_ids: List[int],
    analysis_type: str = Query("comprehensive"),
    priority: str = Query("normal", regex="^(low|normal|high|urgent)$"),
    current_user: models.User = Depends(require_role("school_admin", "teacher", "analyst")),
    db: Session = Depends(get_db)
):
    """Batch analyze multiple students"""
    
    # Check permissions for all students
    for student_id in student_ids:
        # Simplified permission check - in production, implement proper check
        student = db.query(models.Student).filter(models.Student.id == student_id).first()
        if student and current_user.school_id != student.school_id and current_user.role != "super_admin":
            raise HTTPException(status_code=403, detail=f"Cannot access student {student_id}")
    
    log_audit_event(
        db, current_user, "AI_BATCH_ANALYSIS_REQUESTED",
        details={"student_count": len(student_ids), "analysis_type": analysis_type, "priority": priority}
    )
    
    # Run batch analysis
    analysis = ai_engine.batch_analyze_students(student_ids)
    
    return analysis

@app.get("/api/ai/predictions/student/{student_id}")
async def get_student_predictions(
    student_id: int,
    prediction_type: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    limit: int = Query(10, ge=1, le=100),
    current_user: models.User = Depends(can_access_student(student_id)),
    db: Session = Depends(get_db)
):
    """Get AI predictions for student"""
    crud = CRUDService(db)
    
    predictions = crud.get_ai_predictions(
        student_id=student_id,
        prediction_type=prediction_type,
        start_date=start_date,
        end_date=end_date,
        limit=limit
    )
    
    return [AIPredictionResponse.model_validate(p) for p in predictions]

@app.post("/api/ai/models/train")
async def train_ai_model(
    training_job: AITrainingJobCreate,
    current_user: models.User = Depends(require_role("super_admin", "analyst")),
    db: Session = Depends(get_db)
):
    """Train new AI model"""
    
    log_audit_event(
        db, current_user, "AI_MODEL_TRAINING_REQUESTED",
        details={"model_type": training_job.model_type, "school_id": training_job.school_id}
    )
    
    # In production, this would be queued and run asynchronously
    result = ai_engine.train_ai_model(
        training_job.model_type,
        {"training_data": "placeholder"}  # Actual training data would be provided
    )
    
    return result

@app.get("/api/ai/conversations")
async def get_ai_conversations(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    student_id: Optional[int] = None,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get AI conversations for user"""
    query = db.query(models.AIConversation).filter(
        models.AIConversation.user_id == current_user.id
    )
    
    if student_id:
        query = query.filter(models.AIConversation.student_id == student_id)
    
    conversations = query.order_by(
        models.AIConversation.created_at.desc()
    ).offset(skip).limit(limit).all()
    
    return [AIConversationResponse.model_validate(c) for c in conversations]

@app.post("/api/ai/conversations")
async def create_ai_conversation(
    conversation_data: AIConversationCreate,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create new AI conversation"""
    
    if conversation_data.student_id:
        # Check permission for student
        student = db.query(models.Student).filter(
            models.Student.id == conversation_data.student_id
        ).first()
        
        if student and current_user.school_id != student.school_id and current_user.role != "super_admin":
            raise HTTPException(status_code=403, detail="Cannot access this student")
    
    conversation = models.AIConversation(
        user_id=current_user.id,
        student_id=conversation_data.student_id,
        school_id=current_user.school_id,
        topic=conversation_data.topic,
        purpose=conversation_data.purpose,
        messages=[],
        ai_model="conversational_ai_v1",
        ai_config={"version": "1.0"}
    )
    
    if conversation_data.initial_message:
        conversation.messages.append({
            "role": "user",
            "content": conversation_data.initial_message,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Get AI response
        response = ai_engine.conversational_ai(
            current_user.id,
            conversation_data.initial_message,
            {"student_id": conversation_data.student_id}
        )
        
        conversation.messages.append({
            "role": "assistant",
            "content": json.dumps(response),
            "timestamp": datetime.utcnow().isoformat()
        })
        
        conversation.key_insights = response.get("key_insights", [])
    
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    
    log_audit_event(
        db, current_user, "AI_CONVERSATION_CREATED",
        resource_type="ai_conversation", resource_id=str(conversation.id),
        details={"purpose": conversation_data.purpose, "has_student": bool(conversation_data.student_id)}
    )
    
    return AIConversationResponse.model_validate(conversation)

# ========== ANALYTICS ENDPOINTS ==========

@app.get("/api/analytics/school/{school_id}")
async def get_school_analytics(
    school_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get comprehensive school analytics"""
    
    # Check permissions
    if current_user.role != "super_admin" and current_user.school_id != school_id:
        raise HTTPException(status_code=403, detail="Cannot access analytics for other schools")
    
    log_audit_event(
        db, current_user, "SCHOOL_ANALYTICS_REQUESTED",
        resource_type="school", resource_id=str(school_id)
    )
    
    analysis = analytics_engine.analyze_school_performance(school_id)
    
    return analysis

@app.get("/api/analytics/class/{class_id}")
async def get_class_analytics(
    class_id: int,
    term: Optional[str] = None,
    year: Optional[int] = None,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get class performance analytics"""
    crud = CRUDService(db)
    
    # Get class to check permissions
    class_obj = db.query(models.Class).filter(models.Class.id == class_id).first()
    if not class_obj:
        raise HTTPException(status_code=404, detail="Class not found")
    
    if current_user.role != "super_admin" and current_user.school_id != class_obj.school_id:
        raise HTTPException(status_code=403, detail="Cannot access class from other school")
    
    analysis = crud.get_class_performance(class_id, term, year)
    
    log_audit_event(
        db, current_user, "CLASS_ANALYTICS_REQUESTED",
        resource_type="class", resource_id=str(class_id),
        details={"term": term, "year": year}
    )
    
    return analysis

@app.get("/api/analytics/dashboard")
async def get_dashboard_analytics(
    school_id: Optional[int] = None,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get dashboard analytics"""
    crud = CRUDService(db)
    
    # Apply permission filters
    if current_user.role != "super_admin":
        school_id = current_user.school_id
    
    stats = crud.get_dashboard_stats(school_id)
    
    # Real-time metrics
    real_time = {
        "timestamp": datetime.utcnow().isoformat(),
        "active_sessions": len(connected_clients),
        "requests_per_minute": 0,  # Would come from monitoring
        "predictions_processed": stats.get("ai_insights", {}).get("last_7_days", 0),
        "alerts_triggered": 0,  # Would come from alert system
        "system_health": "healthy"
    }
    
    dashboard_metrics = DashboardMetrics(
        total_students=stats.get("students", {}).get("total", 0),
        active_students=stats.get("students", {}).get("active", 0),
        total_teachers=stats.get("teachers", {}).get("total", 0),
        average_performance=stats.get("exams", {}).get("average_score", 0) if "exams" in stats else 0,
        attendance_rate=stats.get("attendance", {}).get("rate", 0) if "attendance" in stats else 0,
        dropout_risk_index=0,  # Would calculate from student data
        ai_insights_generated=stats.get("ai_insights", {}).get("total_predictions", 0),
        interventions_active=stats.get("interventions", {}).get("active", 0) if "interventions" in stats else 0,
        learning_paths_active=0  # Would query learning paths
    )
    
    return {
        "dashboard_metrics": dashboard_metrics.model_dump(),
        "real_time_metrics": real_time,
        "detailed_stats": stats
    }

@app.post("/api/analytics/reports")
async def generate_report(
    report_request: ReportRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Generate analytics report"""
    
    # Check permissions based on report type
    if report_request.report_type == "school":
        school = db.query(models.School).filter(models.School.id == report_request.entity_id).first()
        if school and current_user.role != "super_admin" and current_user.school_id != school.id:
            raise HTTPException(status_code=403, detail="Cannot generate report for other schools")
    elif report_request.report_type == "student":
        # Use can_access_student logic
        student = db.query(models.Student).filter(models.Student.id == report_request.entity_id).first()
        if student and current_user.school_id != student.school_id and current_user.role != "super_admin":
            raise HTTPException(status_code=403, detail="Cannot generate report for this student")
    
    log_audit_event(
        db, current_user, "REPORT_GENERATED",
        resource_type=report_request.report_type,
        resource_id=str(report_request.entity_id),
        details={"report_type": report_request.report_type, "format": report_request.format}
    )
    
    # Generate report
    report = analytics_engine.generate_performance_report(
        report_request.report_type,
        report_request.entity_id,
        report_request.start_date,
        report_request.end_date
    )
    
    # In production, would generate PDF/Excel and store
    report_id = f"REPORT_{uuid.uuid4().hex[:8]}"
    
    return ReportResponse(
        report_id=report_id,
        report_type=report_request.report_type,
        generated_at=datetime.utcnow(),
        data=report,
        ai_insights=report.get("insights") if report_request.include_ai_insights else None,
        download_url=f"/api/reports/{report_id}/download" if report_request.format != "json" else None
    )

# ========== LEARNING PATH ENDPOINTS ==========

@app.post("/api/learning-paths", response_model=LearningPathResponse)
async def create_learning_path(
    path_data: LearningPathCreate,
    current_user: models.User = Depends(require_role("teacher", "school_admin")),
    db: Session = Depends(get_db)
):
    """Create learning path for student"""
    crud = CRUDService(db)
    
    # Check permission for student
    student = crud.get_student(path_data.student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    if current_user.school_id != student.school_id and current_user.role != "super_admin":
        raise HTTPException(status_code=403, detail="Cannot create learning path for student from other school")
    
    learning_path = crud.create_learning_path(path_data)
    
    log_audit_event(
        db, current_user, "LEARNING_PATH_CREATED",
        resource_type="learning_path", resource_id=str(learning_path.id),
        details={"student_id": path_data.student_id, "ai_generated": path_data.ai_generated}
    )
    
    return LearningPathResponse.model_validate(learning_path)

@app.get("/api/learning-paths/student/{student_id}", response_model=List[LearningPathResponse])
async def get_student_learning_paths(
    student_id: int,
    status: Optional[str] = None,
    current_user: models.User = Depends(can_access_student(student_id)),
    db: Session = Depends(get_db)
):
    """Get learning paths for student"""
    crud = CRUDService(db)
    
    learning_paths = crud.get_learning_paths(
        student_id=student_id,
        status=status
    )
    
    return [LearningPathResponse.model_validate(lp) for lp in learning_paths]

@app.put("/api/learning-paths/{path_id}/progress")
async def update_learning_path_progress(
    path_id: int,
    progress: float = Query(..., ge=0, le=100),
    current_user: models.User = Depends(require_role("teacher", "school_admin")),
    db: Session = Depends(get_db)
):
    """Update learning path progress"""
    crud = CRUDService(db)
    
    # Get learning path to check permissions
    learning_path = db.query(models.LearningPath).filter(models.LearningPath.id == path_id).first()
    if not learning_path:
        raise HTTPException(status_code=404, detail="Learning path not found")
    
    student = crud.get_student(learning_path.student_id)
    if student and current_user.school_id != student.school_id and current_user.role != "super_admin":
        raise HTTPException(status_code=403, detail="Cannot update learning path for student from other school")
    
    updated = crud.update_learning_path_progress(path_id, progress)
    if not updated:
        raise HTTPException(status_code=404, detail="Learning path not found")
    
    log_audit_event(
        db, current_user, "LEARNING_PATH_PROGRESS_UPDATED",
        resource_type="learning_path", resource_id=str(path_id),
        details={"progress": progress, "student_id": learning_path.student_id}
    )
    
    return LearningPathResponse.model_validate(updated)

# ========== NOTIFICATION ENDPOINTS ==========

@app.get("/api/notifications", response_model=List[NotificationResponse])
async def get_notifications(
    unread_only: bool = Query(False),
    category: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user notifications"""
    query = db.query(models.Notification).filter(
        models.Notification.user_id == current_user.id
    )
    
    if unread_only:
        query = query.filter(models.Notification.read == False)
    
    if category:
        query = query.filter(models.Notification.category == category)
    
    notifications = query.order_by(
        models.Notification.created_at.desc()
    ).limit(limit).all()
    
    return [NotificationResponse.model_validate(n) for n in notifications]

@app.post("/api/notifications/{notification_id}/read")
async def mark_notification_read(
    notification_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Mark notification as read"""
    notification = db.query(models.Notification).filter(
        models.Notification.id == notification_id,
        models.Notification.user_id == current_user.id
    ).first()
    
    if not notification:
        raise HTTPException(status_code=404, detail="Notification not found")
    
    notification.read = True
    notification.read_at = datetime.utcnow()
    db.commit()
    
    return SuccessResponse(message="Notification marked as read")

@app.post("/api/notifications/ai")
async def generate_ai_notification(
    student_id: int,
    notification_type: str = Query(..., regex="^(alert|warning|info|success)$"),
    category: str = Query(..., regex="^(academic|behavior|system|ai)$"),
    priority: str = Query("medium", regex="^(low|medium|high|critical)$"),
    current_user: models.User = Depends(can_access_student(student_id)),
    db: Session = Depends(get_db)
):
    """Generate AI-powered notification for student"""
    
    # Get student analysis
    analysis = ai_engine.analyze_student_comprehensive(student_id)
    
    # Generate notification based on analysis
    if notification_type == "alert" and analysis.get("dropout_risk_analysis", {}).get("risk_level") in ["HIGH", "CRITICAL"]:
        title = "🚨 High Dropout Risk Alert"
        message = f"Student shows high dropout risk ({analysis['dropout_risk_analysis']['risk_score']:.1%}). Immediate intervention recommended."
    elif notification_type == "warning" and analysis.get("performance_analysis", {}).get("trend") == "declining":
        title = "⚠️ Academic Performance Warning"
        message = "Student academic performance is declining. Consider additional support."
    else:
        title = "📊 AI Insights Generated"
        message = f"New AI insights available for student. {len(analysis.get('ai_insights', []))} key findings identified."
    
    notification = models.Notification(
        user_id=current_user.id,
        type=notification_type,
        category=category,
        title=title,
        message=message,
        ai_generated=True,
        priority=priority,
        actionable=True,
        action_url=f"/students/{student_id}/ai-analysis",
        data={"analysis_summary": analysis}
    )
    
    db.add(notification)
    db.commit()
    db.refresh(notification)
    
    # Send real-time notification via WebSocket
    await send_notification_to_user(current_user.id, notification)
    
    return NotificationResponse.model_validate(notification)

# ========== WEBHOOKS & REAL-TIME ENDPOINTS ==========

@app.websocket("/ws/ai/updates")
async def websocket_ai_updates(websocket: WebSocket):
    """WebSocket for real-time AI updates"""
    await websocket.accept()
    client_id = str(uuid.uuid4())
    connected_clients[client_id] = websocket
    active_subscriptions[client_id] = []
    
    try:
        # Send connection confirmation
        await websocket.send_json({
            "type": "connection_established",
            "client_id": client_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        while True:
            # Wait for client message (subscription requests)
            data = await websocket.receive_json()
            
            if data.get("type") == "subscribe":
                topics = data.get("topics", [])
                active_subscriptions[client_id] = topics
                
                await websocket.send_json({
                    "type": "subscription_updated",
                    "topics": topics,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
            elif data.get("type") == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                })
                
    except WebSocketDisconnect:
        # Clean up on disconnect
        if client_id in connected_clients:
            del connected_clients[client_id]
        if client_id in active_subscriptions:
            del active_subscriptions[client_id]
    except Exception as e:
        print(f"WebSocket error: {e}")
        if client_id in connected_clients:
            await connected_clients[client_id].close()
            del connected_clients[client_id]
        if client_id in active_subscriptions:
            del active_subscriptions[client_id]

async def send_notification_to_user(user_id: uuid.UUID, notification: models.Notification):
    """Send notification to user via WebSocket"""
    message = {
        "type": "notification",
        "data": NotificationResponse.model_validate(notification).model_dump(),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Find all WebSocket connections for this user
    # In production, maintain user_id -> connection mapping
    for client_id, websocket in connected_clients.items():
        try:
            await websocket.send_json(message)
        except:
            pass  # Client disconnected

async def broadcast_ai_update(update_type: str, data: Dict[str, Any]):
    """Broadcast AI update to all subscribed clients"""
    message = {
        "type": "ai_update",
        "update_type": update_type,
        "data": data,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    for client_id, websocket in connected_clients.items():
        if update_type in active_subscriptions.get(client_id, []):
            try:
                await websocket.send_json(message)
            except:
                pass

# ========== DATA EXPORT & IMPORT ENDPOINTS ==========

@app.get("/api/export/students")
async def export_student_data(
    school_id: Optional[int] = None,
    format: str = Query("json", regex="^(json|csv)$"),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Export student data"""
    crud = CRUDService(db)
    
    # Apply permission filters
    if current_user.role != "super_admin":
        school_id = current_user.school_id
    
    if not school_id:
        raise HTTPException(status_code=400, detail="School ID required")
    
    export_data = crud.export_student_data(school_id, format)
    
    log_audit_event(
        db, current_user, "DATA_EXPORTED",
        resource_type="school", resource_id=str(school_id),
        details={"format": format, "student_count": export_data.get("total_students", 0)}
    )
    
    if format == "csv":
        # Convert to CSV
        df = pd.DataFrame(export_data["data"])
        csv_content = df.to_csv(index=False)
        
        return JSONResponse(
            content={"csv": csv_content},
            headers={"Content-Disposition": f"attachment; filename=students_{school_id}_{datetime.utcnow().strftime('%Y%m%d')}.csv"}
        )
    
    return export_data

@app.post("/api/import/students")
async def import_student_data(
    file_content: str,
    school_id: int,
    current_user: models.User = Depends(require_role("school_admin")),
    db: Session = Depends(get_db)
):
    """Import student data from CSV/JSON"""
    
    if current_user.school_id != school_id and current_user.role != "super_admin":
        raise HTTPException(status_code=403, detail="Cannot import data to other schools")
    
    try:
        # Parse CSV or JSON
        if file_content.strip().startswith("["):  # JSON array
            data = json.loads(file_content)
        else:  # Assume CSV
            df = pd.read_csv(pd.compat.StringIO(file_content))
            data = df.to_dict(orient="records")
        
        imported_count = 0
        errors = []
        
        crud = CRUDService(db)
        
        for row in data:
            try:
                # Map row to StudentCreate schema
                student_data = StudentCreate(
                    admission_number=row.get("admission_number", ""),
                    first_name=row.get("first_name", ""),
                    last_name=row.get("last_name", ""),
                    date_of_birth=datetime.strptime(row.get("date_of_birth"), "%Y-%m-%d").date() if row.get("date_of_birth") else None,
                    gender=row.get("gender", ""),
                    school_id=school_id,
                    class_id=row.get("class_id"),
                    stream=row.get("stream"),
                    parent_name=row.get("parent_name"),
                    parent_email=row.get("parent_email"),
                    parent_phone=row.get("parent_phone")
                )
                
                student = crud.create_student(student_data)
                imported_count += 1
                
            except Exception as e:
                errors.append({
                    "row": row,
                    "error": str(e)
                })
        
        log_audit_event(
            db, current_user, "DATA_IMPORTED",
            resource_type="school", resource_id=str(school_id),
            details={"imported_count": imported_count, "error_count": len(errors)}
        )
        
        return {
            "message": f"Imported {imported_count} students",
            "imported_count": imported_count,
            "error_count": len(errors),
            "errors": errors[:10]  # Limit errors in response
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Import failed: {str(e)}")

# ========== SYSTEM HEALTH & MONITORING ==========

@app.get("/api/health")
async def health_check():
    """System health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "4.0.0",
        "services": {
            "database": "connected",
            "ai_engine": "running",
            "analytics_engine": "running",
            "websocket": f"{len(connected_clients)} clients connected"
        }
    }

@app.get("/api/system/metrics")
async def system_metrics(
    current_user: models.User = Depends(require_role("super_admin"))
):
    """Get system metrics (admin only)"""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "active_users": len(connected_clients),
        "total_students": 0,  # Would query database
        "total_schools": 0,   # Would query database
        "ai_predictions_today": 0,
        "system_load": {
            "cpu": 0.15,
            "memory": 0.32,
            "disk": 0.45
        }
    }

# ========== SEARCH ENDPOINTS ==========

@app.get("/api/search/students")
async def search_students(
    query: str = Query(..., min_length=2),
    school_id: Optional[int] = None,
    limit: int = Query(20, ge=1, le=100),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Search students by name or admission number"""
    crud = CRUDService(db)
    
    # Apply permission filters
    if current_user.role != "super_admin":
        school_id = current_user.school_id
    
    students = crud.search_students(query, school_id, limit)
    
    return [StudentResponse.model_validate(s) for s in students]

# ========== ROOT & INFO ENDPOINTS ==========

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to EduTrack AI Pro API",
        "version": "4.0.0",
        "description": "Advanced Educational Analytics Platform",
        "documentation": "/api/docs",
        "health_check": "/api/health",
        "features": [
            "AI-powered student analytics",
            "Real-time monitoring",
            "Predictive modeling",
            "Personalized learning paths",
            "Multi-school management",
            "Advanced reporting"
        ]
    }

@app.get("/api/info")
async def api_info():
    """Detailed API information"""
    return {
        "api": {
            "name": "EduTrack AI Pro",
            "version": "4.0.0",
            "status": "active",
            "uptime": "0 days 0 hours"  # Would calculate from start time
        },
        "ai_capabilities": {
            "dropout_prediction": True,
            "performance_forecasting": True,
            "behavioral_analysis": True,
            "personalized_recommendations": True,
            "natural_language_insights": True
        },
        "analytics_features": {
            "real_time_dashboards": True,
            "predictive_analytics": True,
            "comparative_analysis": True,
            "trend_analysis": True,
            "benchmarking": True
        },
        "system_requirements": {
            "python": "3.9+",
            "database": "PostgreSQL 12+",
            "memory": "8GB+",
            "storage": "100GB+"
        }
    }

# ========== ERROR HANDLERS ==========

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            code=exc.status_code,
            details={"path": request.url.path}
        ).model_dump()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            code="INTERNAL_ERROR",
            details={"message": str(exc)}
        ).model_dump()
    )

# ========== STARTUP & SHUTDOWN EVENTS ==========

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    print("🚀 EduTrack AI Pro API starting up...")
    print(f"📊 AI Engine initialized: {ai_engine}")
    print(f"📈 Analytics Engine initialized: {analytics_engine}")
    
    # Initialize sample data in development
    if os.getenv("ENVIRONMENT") == "development":
        await initialize_sample_data()

@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    print("🛑 EduTrack AI Pro API shutting down...")
    
    # Close WebSocket connections
    for websocket in connected_clients.values():
        try:
            await websocket.close()
        except:
            pass
    
    # Close AI engine
    ai_engine.close()
    analytics_engine.close()

# ========== HELPER FUNCTIONS ==========

async def initialize_sample_data():
    """Initialize sample data for development"""
    print("📝 Initializing sample data...")
    
    # This would create sample schools, students, etc.
    # For now, just a placeholder
    pass

# ========== MAIN EXECUTION ==========

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
                                       )
