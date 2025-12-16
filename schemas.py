from pydantic import BaseModel, Field, EmailStr, validator, ConfigDict
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, date
from enum import Enum
import uuid
from decimal import Decimal

# Base Schemas
class BaseSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

# UUID Mixin
class UUIDMixin(BaseModel):
    uuid: uuid.UUID

# Timestamp Mixin
class TimestampMixin(BaseModel):
    created_at: datetime
    updated_at: Optional[datetime] = None

# User Schemas
class UserRole(str, Enum):
    SUPER_ADMIN = "super_admin"
    SCHOOL_ADMIN = "school_admin"
    TEACHER = "teacher"
    STUDENT = "student"
    PARENT = "parent"
    ANALYST = "analyst"

class UserBase(BaseSchema):
    email: EmailStr
    username: Optional[str] = None
    full_name: Optional[str] = None
    role: UserRole = UserRole.TEACHER
    school_id: Optional[int] = None
    is_active: bool = True
    ai_preferences: Optional[Dict[str, Any]] = None

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserUpdate(BaseSchema):
    full_name: Optional[str] = None
    ai_preferences: Optional[Dict[str, Any]] = None
    notification_settings: Optional[Dict[str, Any]] = None

class UserResponse(UserBase, UUIDMixin, TimestampMixin):
    id: uuid.UUID
    last_login: Optional[datetime] = None
    is_verified: bool = False

class UserWithToken(UserResponse):
    access_token: str
    token_type: str = "bearer"
    refresh_token: Optional[str] = None

# School Schemas
class SchoolBase(BaseSchema):
    name: str = Field(..., min_length=2, max_length=255)
    code: str = Field(..., min_length=2, max_length=50)
    type: Optional[str] = None
    level: Optional[str] = None
    county: Optional[str] = None
    subcounty: Optional[str] = None
    principal_name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    established_year: Optional[int] = None
    
    # AI Configuration
    ai_enabled: bool = True
    ai_config: Optional[Dict[str, Any]] = None

class SchoolCreate(SchoolBase):
    pass

class SchoolUpdate(BaseSchema):
    name: Optional[str] = None
    ai_config: Optional[Dict[str, Any]] = None
    settings: Optional[Dict[str, Any]] = None

class SchoolResponse(SchoolBase, TimestampMixin):
    id: int
    uuid: uuid.UUID
    total_students: int = 0
    total_teachers: int = 0
    subscription_tier: str = "free"
    subscription_expires: Optional[datetime] = None

class SchoolWithStats(SchoolResponse):
    performance_metrics: Optional[Dict[str, Any]] = None
    ai_insights: Optional[Dict[str, Any]] = None

# Student Schemas
class StudentBase(BaseSchema):
    admission_number: str
    first_name: str
    last_name: str
    date_of_birth: date
    gender: str
    class_id: Optional[int] = None
    stream: Optional[str] = None
    parent_name: Optional[str] = None
    parent_email: Optional[EmailStr] = None
    parent_phone: Optional[str] = None

class StudentCreate(StudentBase):
    school_id: int

class StudentUpdate(BaseSchema):
    class_id: Optional[int] = None
    stream: Optional[str] = None
    status: Optional[str] = None
    ai_profile: Optional[Dict[str, Any]] = None

class StudentResponse(StudentBase, UUIDMixin, TimestampMixin):
    id: int
    school_id: int
    status: str = "active"
    is_active: bool = True
    
    # AI Metrics
    dropout_risk_score: float = 0.0
    performance_score: float = 0.0
    engagement_score: float = 0.0
    wellbeing_score: float = 0.0
    
    ai_profile: Optional[Dict[str, Any]] = None

class StudentWithDetails(StudentResponse):
    school: Optional[SchoolResponse] = None
    class_info: Optional[Dict[str, Any]] = None
    recent_performance: Optional[Dict[str, Any]] = None
    ai_predictions: Optional[List[Dict[str, Any]]] = None

# Exam Schemas
class ExamBase(BaseSchema):
    name: str
    code: str
    type: str
    term: str
    year: int
    total_score: float = 100.0
    passing_score: float = 40.0
    weight: float = 1.0

class ExamCreate(ExamBase):
    school_id: int
    class_id: Optional[int] = None

class ExamResponse(ExamBase, UUIDMixin, TimestampMixin):
    id: int
    school_id: Optional[int] = None
    class_id: Optional[int] = None
    
    # AI Analysis
    difficulty_index: Optional[float] = None
    discrimination_index: Optional[float] = None
    reliability_score: Optional[float] = None
    
    results_published: bool = False
    analysis_complete: bool = False

# Exam Result Schemas
class ExamResultBase(BaseSchema):
    student_id: int
    exam_id: int
    subject_id: int
    raw_score: float
    total_score: float = 100.0
    percentage: Optional[float] = None
    grade: Optional[str] = None
    points: Optional[int] = None
    
    @validator('percentage', always=True)
    def calculate_percentage(cls, v, values):
        if v is None:
            raw_score = values.get('raw_score')
            total_score = values.get('total_score')
            if raw_score is not None and total_score is not None:
                return (raw_score / total_score) * 100
        return v

class ExamResultCreate(ExamResultBase):
    pass

class ExamResultUpdate(BaseSchema):
    raw_score: Optional[float] = None
    comments: Optional[str] = None

class ExamResultResponse(ExamResultBase, UUIDMixin, TimestampMixin):
    id: int
    position: Optional[int] = None
    position_out_of: Optional[int] = None
    
    # AI Enhanced Fields
    normalized_score: Optional[float] = None
    percentile_rank: Optional[float] = None
    growth_score: Optional[float] = None
    expected_score: Optional[float] = None
    performance_gap: Optional[float] = None
    
    verified: bool = False

# AI Prediction Schemas
class AIPredictionBase(BaseSchema):
    student_id: int
    model_id: int
    prediction_type: str
    prediction_date: date
    input_features: Dict[str, Any]
    prediction_output: Dict[str, Any]
    confidence_scores: Dict[str, float]
    explanation: Optional[Dict[str, Any]] = None
    key_factors: Optional[List[Dict[str, Any]]] = None
    recommendations: Optional[List[Dict[str, Any]]] = None

class AIPredictionCreate(AIPredictionBase):
    pass

class AIPredictionResponse(AIPredictionBase, UUIDMixin):
    id: int
    valid_until: Optional[date] = None
    reviewed: bool = False
    review_notes: Optional[str] = None

# Learning Path Schemas
class LearningPathBase(BaseSchema):
    student_id: int
    name: str
    description: Optional[str] = None
    objectives: Optional[List[Dict[str, Any]]] = None
    modules: Optional[List[Dict[str, Any]]] = None
    resources: Optional[List[Dict[str, Any]]] = None
    difficulty_level: Optional[str] = None
    pace: Optional[str] = None
    learning_style: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None

class LearningPathCreate(LearningPathBase):
    ai_generated: bool = True

class LearningPathUpdate(BaseSchema):
    progress: Optional[float] = None
    status: Optional[str] = None
    effectiveness_score: Optional[float] = None

class LearningPathResponse(LearningPathBase, UUIDMixin, TimestampMixin):
    id: int
    ai_generated: bool = True
    progress: float = 0.0
    status: str = "active"
    effectiveness_score: float = 0.0
    completion_rate: float = 0.0

# Analytics Schemas
class AnalyticsRequest(BaseSchema):
    school_id: Optional[int] = None
    class_id: Optional[int] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    metrics: Optional[List[str]] = None
    granularity: str = "daily"  # "daily", "weekly", "monthly", "yearly"

class PerformanceAnalytics(BaseSchema):
    period: str
    average_score: float
    top_performer: Optional[Dict[str, Any]] = None
    needs_attention: Optional[List[Dict[str, Any]]] = None
    subject_performance: Dict[str, float]
    trend: str  # "improving", "declining", "stable"

class DropoutRiskAnalytics(BaseSchema):
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    risk_factors: Dict[str, int]
    interventions_needed: int
    predicted_dropouts: int

class EngagementAnalytics(BaseSchema):
    attendance_rate: float
    participation_score: float
    behavioral_incidents: int
    extracurricular_involvement: float
    overall_engagement: float

# AI Model Schemas
class AIModelBase(BaseSchema):
    name: str
    version: str
    model_type: str
    framework: str
    description: Optional[str] = None
    input_features: List[str]
    output_labels: List[str]
    hyperparameters: Dict[str, Any]

class AIModelCreate(AIModelBase):
    school_id: Optional[int] = None

class AIModelUpdate(BaseSchema):
    is_active: Optional[bool] = None
    is_production: Optional[bool] = None
    accuracy: Optional[float] = None

class AIModelResponse(AIModelBase, UUIDMixin, TimestampMixin):
    id: int
    school_id: Optional[int] = None
    
    # Performance Metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    
    # Training Info
    training_data_size: Optional[int] = None
    training_duration: Optional[float] = None
    last_trained: Optional[datetime] = None
    
    # Deployment
    is_active: bool = True
    is_production: bool = False
    deployment_date: Optional[datetime] = None

# AI Training Job Schemas
class AITrainingJobCreate(BaseSchema):
    model_type: str
    school_id: Optional[int] = None
    parameters: Dict[str, Any]
    training_data_range: Dict[str, date]

class AITrainingJobResponse(BaseSchema):
    id: int
    uuid: uuid.UUID
    model_type: str
    status: str
    progress: float
    results: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

# AI Conversation Schemas
class AIMessage(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class AIConversationCreate(BaseSchema):
    student_id: Optional[int] = None
    topic: Optional[str] = None
    purpose: str = "analysis"
    initial_message: Optional[str] = None

class AIConversationResponse(BaseSchema):
    id: int
    uuid: uuid.UUID
    student_id: Optional[int] = None
    topic: Optional[str] = None
    purpose: str
    messages: List[AIMessage]
    ai_model: str
    key_insights: Optional[List[str]] = None
    ended: bool = False

# Notification Schemas
class NotificationBase(BaseSchema):
    type: str
    category: str
    title: str
    message: str
    priority: str = "medium"
    actionable: bool = False
    action_url: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

class NotificationCreate(NotificationBase):
    user_id: uuid.UUID

class NotificationResponse(NotificationBase, UUIDMixin):
    id: int
    user_id: uuid.UUID
    ai_generated: bool = False
    read: bool = False
    read_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

# Dashboard Schemas
class DashboardMetrics(BaseSchema):
    total_students: int
    active_students: int
    total_teachers: int
    average_performance: float
    attendance_rate: float
    dropout_risk_index: float
    ai_insights_generated: int
    interventions_active: int
    learning_paths_active: int

class RealTimeMetrics(BaseSchema):
    timestamp: datetime
    active_sessions: int
    requests_per_minute: int
    predictions_processed: int
    alerts_triggered: int
    system_health: str

# Report Schemas
class ReportRequest(BaseSchema):
    report_type: str  # "student", "class", "school", "subject"
    entity_id: int
    start_date: date
    end_date: date
    include_ai_insights: bool = True
    format: str = "json"  # "json", "pdf", "excel"

class ReportResponse(BaseSchema):
    report_id: str
    report_type: str
    generated_at: datetime
    data: Dict[str, Any]
    ai_insights: Optional[Dict[str, Any]] = None
    download_url: Optional[str] = None

# API Response Schemas
class PaginatedResponse(BaseSchema):
    items: List[Any]
    total: int
    page: int
    size: int
    pages: int

class SuccessResponse(BaseSchema):
    success: bool = True
    message: str
    data: Optional[Any] = None

class ErrorResponse(BaseSchema):
    success: bool = False
    error: str
    code: str
    details: Optional[Dict[str, Any]] = None

# WebSocket Schemas
class WSMessage(BaseModel):
    type: str  # "update", "alert", "notification", "prediction"
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class WSConnection(BaseModel):
    user_id: uuid.UUID
    session_id: str
    subscribed_topics: List[str]

# AI Engine Request/Response
class AIAnalysisRequest(BaseSchema):
    student_id: int
    analysis_type: str  # "comprehensive", "dropout", "performance", "engagement"
    include_recommendations: bool = True
    time_horizon: str = "30d"  # "7d", "30d", "90d", "1y"

class AIAnalysisResponse(BaseSchema):
    student_id: int
    analysis_type: str
    generated_at: datetime
    risk_assessment: Dict[str, Any]
    performance_analysis: Dict[str, Any]
    behavioral_insights: Optional[Dict[str, Any]] = None
    personalized_recommendations: List[Dict[str, Any]]
    predicted_outcomes: Dict[str, Any]
    confidence_scores: Dict[str, float]
    next_review_date: date

class BatchAnalysisRequest(BaseSchema):
    student_ids: List[int]
    analysis_type: str
    priority: str = "normal"  # "low", "normal", "high", "urgent"

class BatchAnalysisResponse(BaseSchema):
    job_id: str
    total_students: int
    estimated_completion: datetime
    status: str = "queued"
