from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Date,
    ForeignKey, Text, JSON, ARRAY, LargeBinary, Enum,
    func, Index, BigInteger, Numeric, text
)
from sqlalchemy.orm import relationship, backref
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY as PG_ARRAY
from sqlalchemy.ext.mutable import MutableDict, MutableList
import uuid
from datetime import datetime, timezone
from database import Base
import enum

class UserRole(str, enum.Enum):
    SUPER_ADMIN = "super_admin"
    SCHOOL_ADMIN = "school_admin"
    TEACHER = "teacher"
    STUDENT = "student"
    PARENT = "parent"
    ANALYST = "analyst"

class User(Base):
    _tablename_ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True)
    full_name = Column(String(255))
    hashed_password = Column(String(255), nullable=False)
    role = Column(Enum(UserRole), default=UserRole.TEACHER)
    school_id = Column(Integer, ForeignKey("schools.id"), nullable=True)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    last_login = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # AI-specific fields
    ai_preferences = Column(JSONB, default=lambda: {"notifications": True, "personalized_insights": True})
    notification_settings = Column(JSONB, default=lambda: {"email": True, "push": True, "sms": False})
    
    school = relationship("School", back_populates="users")
    api_keys = relationship("APIKey", back_populates="user")
    audit_logs = relationship("AuditLog", back_populates="user")

class School(Base):
    _tablename_ = "schools"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True)
    name = Column(String(255), nullable=False)
    code = Column(String(50), unique=True, nullable=False)
    type = Column(String(50))  # "Public", "Private", "International"
    level = Column(String(50))  # "Primary", "Secondary", "Mixed"
    county = Column(String(100))
    subcounty = Column(String(100))
    ward = Column(String(100))
    principal_name = Column(String(255))
    principal_email = Column(String(255))
    principal_phone = Column(String(20))
    email = Column(String(255))
    phone = Column(String(20))
    address = Column(Text)
    website = Column(String(255))
    established_year = Column(Integer)
    total_students = Column(Integer, default=0)
    total_teachers = Column(Integer, default=0)
    subscription_tier = Column(String(50), default="free")  # "free", "basic", "pro", "enterprise"
    subscription_expires = Column(DateTime(timezone=True))
    settings = Column(JSONB, default=lambda: {})
    
    # AI Configuration
    ai_enabled = Column(Boolean, default=True)
    ai_config = Column(JSONB, default=lambda: {
        "dropout_prediction": True,
        "performance_forecasting": True,
        "sentiment_analysis": True,
        "personalized_learning": True,
        "automated_reporting": True
    })
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    users = relationship("User", back_populates="school")
    students = relationship("Student", back_populates="school")
    teachers = relationship("Teacher", back_populates="school")
    classes = relationship("Class", back_populates="school")
    exams = relationship("Exam", back_populates="school")
    ai_models = relationship("AIModel", back_populates="school")

class Student(Base):
    _tablename_ = "students"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True)
    admission_number = Column(String(50), unique=True, nullable=False)
    first_name = Column(String(100), nullable=False)
    middle_name = Column(String(100))
    last_name = Column(String(100), nullable=False)
    date_of_birth = Column(Date, nullable=False)
    gender = Column(String(10), nullable=False)
    nationality = Column(String(50), default="Kenyan")
    id_number = Column(String(20))
    birth_certificate = Column(String(50))
    
    # School information
    school_id = Column(Integer, ForeignKey("schools.id"), nullable=False)
    class_id = Column(Integer, ForeignKey("classes.id"))
    stream = Column(String(50))
    admission_date = Column(Date)
    graduation_date = Column(Date, nullable=True)
    is_active = Column(Boolean, default=True)
    status = Column(String(50), default="active")  # "active", "transferred", "graduated", "dropped"
    
    # Contact information
    parent_name = Column(String(255))
    parent_email = Column(String(255))
    parent_phone = Column(String(20))
    home_address = Column(Text)
    emergency_contact = Column(String(20))
    
    # Academic history
    previous_school = Column(String(255))
    previous_performance = Column(JSONB, default=lambda: {})
    
    # AI Profile
    ai_profile = Column(JSONB, default=lambda: {
        "learning_style": None,
        "cognitive_profile": None,
        "engagement_level": None,
        "risk_factors": [],
        "strengths": [],
        "weaknesses": []
    })
    
    # Metrics (updated by AI)
    dropout_risk_score = Column(Float, default=0.0)
    performance_score = Column(Float, default=0.0)
    engagement_score = Column(Float, default=0.0)
    wellbeing_score = Column(Float, default=0.0)
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    school = relationship("School", back_populates="students")
    class_rel = relationship("Class", back_populates="students")
    exam_results = relationship("ExamResult", back_populates="student")
    attendance = relationship("Attendance", back_populates="student")
    behaviors = relationship("StudentBehavior", back_populates="student")
    interventions = relationship("Intervention", back_populates="student")
    ai_predictions = relationship("AIPrediction", back_populates="student")
    learning_paths = relationship("LearningPath", back_populates="student")

class Teacher(Base):
    _tablename_ = "teachers"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True)
    teacher_id = Column(String(50), unique=True, nullable=False)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True)
    phone = Column(String(20))
    qualification = Column(String(100))
    specialization = Column(String(255))
    subjects = Column(PG_ARRAY(String))  # Array of subjects
    years_experience = Column(Integer)
    
    school_id = Column(Integer, ForeignKey("schools.id"), nullable=False)
    is_active = Column(Boolean, default=True)
    
    # AI Teaching Profile
    teaching_style = Column(JSONB, default=lambda: {})
    effectiveness_score = Column(Float, default=0.0)
    student_satisfaction = Column(Float, default=0.0)
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    school = relationship("School", back_populates="teachers")
    classes = relationship("Class", back_populates="teacher")

class Class(Base):
    _tablename_ = "classes"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)  # e.g., "Form 4A"
    level = Column(String(50), nullable=False)  # e.g., "Form 4"
    stream = Column(String(50))  # e.g., "A", "B", "C"
    capacity = Column(Integer, default=40)
    room_number = Column(String(20))
    
    school_id = Column(Integer, ForeignKey("schools.id"), nullable=False)
    teacher_id = Column(Integer, ForeignKey("teachers.id"))
    
    # AI Class Metrics
    class_performance_index = Column(Float, default=0.0)
    engagement_index = Column(Float, default=0.0)
    cohesion_score = Column(Float, default=0.0)
    learning_environment_score = Column(Float, default=0.0)
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    school = relationship("School", back_populates="classes")
    teacher = relationship("Teacher", back_populates="classes")
    students = relationship("Student", back_populates="class_rel")
    subjects = relationship("ClassSubject", back_populates="class_rel")

class Subject(Base):
    _tablename_ = "subjects"
    
    id = Column(Integer, primary_key=True, index=True)
    code = Column(String(20), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    category = Column(String(50))  # "Core", "Science", "Humanities", "Technical", "Languages"
    difficulty_level = Column(String(20))  # "Basic", "Intermediate", "Advanced"
    credits = Column(Integer, default=1)
    description = Column(Text)
    
    # AI Metadata
    knowledge_graph = Column(JSONB, default=lambda: {})
    prerequisites = Column(PG_ARRAY(String))
    related_subjects = Column(PG_ARRAY(String))
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    exam_results = relationship("ExamResult", back_populates="subject")
    class_subjects = relationship("ClassSubject", back_populates="subject")

class ClassSubject(Base):
    _tablename_ = "class_subjects"
    
    id = Column(Integer, primary_key=True, index=True)
    class_id = Column(Integer, ForeignKey("classes.id"), nullable=False)
    subject_id = Column(Integer, ForeignKey("subjects.id"), nullable=False)
    teacher_id = Column(Integer, ForeignKey("teachers.id"))
    
    # Schedule
    schedule = Column(JSONB, default=lambda: {})
    academic_year = Column(Integer, nullable=False)
    term = Column(String(20), nullable=False)  # "Term 1", "Term 2", "Term 3"
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    class_rel = relationship("Class", back_populates="subjects")
    subject = relationship("Subject", back_populates="class_subjects")

class Exam(Base):
    _tablename_ = "exams"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True)
    name = Column(String(255), nullable=False)
    code = Column(String(50), unique=True, nullable=False)
    type = Column(String(50))  # "Mid-term", "End-term", "National", "Mock", "Continuous"
    term = Column(String(20))
    year = Column(Integer, nullable=False)
    
    school_id = Column(Integer, ForeignKey("schools.id"))
    class_id = Column(Integer, ForeignKey("classes.id"))
    
    total_score = Column(Float, default=100.0)
    passing_score = Column(Float, default=40.0)
    weight = Column(Float, default=1.0)  # Weight in final grade calculation
    
    # AI Exam Analysis
    difficulty_index = Column(Float, default=0.0)
    discrimination_index = Column(Float, default=0.0)
    reliability_score = Column(Float, default=0.0)
    
    conducted_date = Column(Date)
    results_published = Column(Boolean, default=False)
    analysis_complete = Column(Boolean, default=False)
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    school = relationship("School", back_populates="exams")
    exam_results = relationship("ExamResult", back_populates="exam")

class ExamResult(Base):
    _tablename_ = "exam_results"
    
    id = Column(BigInteger, primary_key=True, index=True)
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True)
    
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False)
    exam_id = Column(Integer, ForeignKey("exams.id"), nullable=False)
    subject_id = Column(Integer, ForeignKey("subjects.id"), nullable=False)
    
    raw_score = Column(Float, nullable=False)
    total_score = Column(Float, nullable=False)
    percentage = Column(Float, nullable=False)
    grade = Column(String(5))
    points = Column(Integer)
    position = Column(Integer)  # Class position
    position_out_of = Column(Integer)  # Total students in class
    
    # AI Enhanced Fields
    normalized_score = Column(Float)  # Z-score normalization
    percentile_rank = Column(Float)
    growth_score = Column(Float)  # Compared to previous performance
    expected_score = Column(Float)  # AI predicted score
    performance_gap = Column(Float)  # Difference between expected and actual
    
    comments = Column(Text)
    entered_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    verified = Column(Boolean, default=False)
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    student = relationship("Student", back_populates="exam_results")
    exam = relationship("Exam", back_populates="exam_results")
    subject = relationship("Subject", back_populates="exam_results")

class Attendance(Base):
    _tablename_ = "attendance"
    
    id = Column(BigInteger, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False)
    date = Column(Date, nullable=False)
    status = Column(String(20), nullable=False)  # "present", "absent", "late", "excused"
    reason = Column(Text)
    session = Column(String(20))  # "morning", "afternoon", "full_day"
    
    # AI Analysis
    pattern_flag = Column(Boolean, default=False)  # Flag for concerning patterns
    trend_score = Column(Float, default=0.0)
    
    recorded_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    student = relationship("Student", back_populates="attendance")
    
    _table_args_ = (
        Index('idx_attendance_student_date', 'student_id', 'date'),
    )

class StudentBehavior(Base):
    _tablename_ = "student_behaviors"
    
    id = Column(BigInteger, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False)
    
    behavior_type = Column(String(50), nullable=False)  # "positive", "negative", "neutral"
    category = Column(String(100))  # "academic", "social", "disciplinary", "extracurricular"
    description = Column(Text, nullable=False)
    severity = Column(String(20))  # "low", "medium", "high", "critical"
    
    # AI Sentiment Analysis
    sentiment_score = Column(Float)
    keywords = Column(PG_ARRAY(String))
    
    reported_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    date_observed = Column(Date)
    resolved = Column(Boolean, default=False)
    resolution_notes = Column(Text)
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    student = relationship("Student", back_populates="behaviors")

class Intervention(Base):
    _tablename_ = "interventions"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True)
    
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False)
    type = Column(String(50), nullable=False)  # "academic", "behavioral", "counseling", "parental"
    title = Column(String(255), nullable=False)
    description = Column(Text)
    reason = Column(Text, nullable=False)
    
    # AI Recommended Fields
    ai_recommended = Column(Boolean, default=False)
    recommendation_confidence = Column(Float)
    expected_outcome = Column(Text)
    success_probability = Column(Float)
    
    assigned_to = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    start_date = Column(Date)
    end_date = Column(Date)
    status = Column(String(20), default="pending")  # "pending", "active", "completed", "cancelled"
    outcome = Column(String(20))  # "successful", "partial", "failed"
    outcome_notes = Column(Text)
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    student = relationship("Student", back_populates="interventions")
    sessions = relationship("InterventionSession", back_populates="intervention")

class LearningPath(Base):
    _tablename_ = "learning_paths"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True)
    
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # AI-Generated Learning Path
    ai_generated = Column(Boolean, default=True)
    objectives = Column(JSONB, default=lambda: [])
    modules = Column(JSONB, default=lambda: [])
    resources = Column(JSONB, default=lambda: [])
    milestones = Column(JSONB, default=lambda: [])
    
    # Adaptive Learning Parameters
    difficulty_level = Column(String(20))
    pace = Column(String(20))  # "slow", "moderate", "fast"
    learning_style = Column(String(50))
    
    start_date = Column(Date)
    end_date = Column(Date)
    progress = Column(Float, default=0.0)  # 0 to 100
    status = Column(String(20), default="active")  # "active", "completed", "paused"
    
    effectiveness_score = Column(Float, default=0.0)
    completion_rate = Column(Float, default=0.0)
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    student = relationship("Student", back_populates="learning_paths")
    activities = relationship("LearningActivity", back_populates="learning_path")

# AI-SPECIFIC MODELS

class AIModel(Base):
    _tablename_ = "ai_models"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True)
    
    name = Column(String(255), nullable=False)
    version = Column(String(50), nullable=False)
    model_type = Column(String(50), nullable=False)  # "dropout", "performance", "sentiment", "recommendation"
    framework = Column(String(50))  # "tensorflow", "pytorch", "sklearn", "custom"
    
    school_id = Column(Integer, ForeignKey("schools.id"))
    
    # Model Metadata
    description = Column(Text)
    input_features = Column(JSONB, default=lambda: [])
    output_labels = Column(JSONB, default=lambda: [])
    hyperparameters = Column(JSONB, default=lambda: {})
    
    # Performance Metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    auc_roc = Column(Float)
    mse = Column(Float)
    
    # Training Info
    training_data_size = Column(Integer)
    training_duration = Column(Float)  # in seconds
    last_trained = Column(DateTime(timezone=True))
    
    # Deployment
    is_active = Column(Boolean, default=True)
    is_production = Column(Boolean, default=False)
    deployment_date = Column(DateTime(timezone=True))
    
    # Model Storage
    model_path = Column(String(500))
    artifact_path = Column(String(500))
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    school = relationship("School", back_populates="ai_models")
    predictions = relationship("AIPrediction", back_populates="model")

class AIPrediction(Base):
    _tablename_ = "ai_predictions"
    
    id = Column(BigInteger, primary_key=True, index=True)
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True)
    
    model_id = Column(Integer, ForeignKey("ai_models.id"), nullable=False)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False)
    
    prediction_type = Column(String(50), nullable=False)  # "dropout", "performance", "engagement", "recommendation"
    
    # Prediction Data
    input_features = Column(JSONB, default=lambda: {})
    prediction_output = Column(JSONB, default=lambda: {})
    confidence_scores = Column(JSONB, default=lambda: {})
    
    # Interpretation
    explanation = Column(JSONB, default=lambda: {})  # SHAP/LIME explanations
    key_factors = Column(JSONB, default=lambda: [])
    recommendations = Column(JSONB, default=lambda: [])
    
    # Metadata
    prediction_date = Column(Date, nullable=False)
    valid_until = Column(Date)
    reviewed = Column(Boolean, default=False)
    reviewed_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    review_notes = Column(Text)
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    model = relationship("AIModel", back_populates="predictions")
    student = relationship("Student", back_populates="ai_predictions")

class AITrainingJob(Base):
    _tablename_ = "ai_training_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True)
    
    model_type = Column(String(50), nullable=False)
    school_id = Column(Integer, ForeignKey("schools.id"))
    
    status = Column(String(20), default="pending")  # "pending", "running", "completed", "failed"
    progress = Column(Float, default=0.0)  # 0 to 100
    
    # Training Parameters
    parameters = Column(JSONB, default=lambda: {})
    training_data_range = Column(JSONB, default=lambda: {})
    
    # Results
    results = Column(JSONB, default=lambda: {})
    metrics = Column(JSONB, default=lambda: {})
    model_path = Column(String(500))
    
    # Logs
    logs = Column(Text)
    error_message = Column(Text)
    
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

class AIConversation(Base):
    _tablename_ = "ai_conversations"
    
    id = Column(BigInteger, primary_key=True, index=True)
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True)
    
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    student_id = Column(Integer, ForeignKey("students.id"), nullable=True)
    school_id = Column(Integer, ForeignKey("schools.id"))
    
    # Conversation Context
    context = Column(JSONB, default=lambda: {})
    topic = Column(String(255))
    purpose = Column(String(50))  # "analysis", "advice", "reporting", "planning"
    
    # Messages (stored as JSON array)
    messages = Column(JSONB, default=lambda: [])
    
    # AI Model Used
    ai_model = Column(String(100))
    ai_config = Column(JSONB, default=lambda: {})
    
    # Analytics
    token_count = Column(Integer, default=0)
    sentiment_score = Column(Float)
    key_insights = Column(JSONB, default=lambda: [])
    
    ended = Column(Boolean, default=False)
    ended_at = Column(DateTime(timezone=True))
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

# SYSTEM MODELS

class APIKey(Base):
    _tablename_ = "api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(100), unique=True, nullable=False)
    name = Column(String(100))
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    permissions = Column(JSONB, default=lambda: ["read"])
    rate_limit = Column(Integer, default=100)  # requests per minute
    is_active = Column(Boolean, default=True)
    
    last_used = Column(DateTime(timezone=True))
    expires_at = Column(DateTime(timezone=True))
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    user = relationship("User", back_populates="api_keys")

class AuditLog(Base):
    _tablename_ = "audit_logs"
    
    id = Column(BigInteger, primary_key=True, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    school_id = Column(Integer, ForeignKey("schools.id"), nullable=True)
    
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50))
    resource_id = Column(String(100))
    
    details = Column(JSONB, default=lambda: {})
    ip_address = Column(String(45))
    user_agent = Column(Text)
    
    status = Column(String(20))  # "success", "failure"
    error_message = Column(Text)
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    user = relationship("User", back_populates="audit_logs")

class Notification(Base):
    _tablename_ = "notifications"
    
    id = Column(BigInteger, primary_key=True, index=True)
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True)
    
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    type = Column(String(50), nullable=False)  # "alert", "warning", "info", "success"
    category = Column(String(50))  # "academic", "behavior", "system", "ai"
    
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    
    # AI-Generated Notification
    ai_generated = Column(Boolean, default=False)
    priority = Column(String(20))  # "low", "medium", "high", "critical"
    actionable = Column(Boolean, default=False)
    action_url = Column(String(500))
    
    data = Column(JSONB, default=lambda: {})
    
    read = Column(Boolean, default=False)
    read_at = Column(DateTime(timezone=True))
    
    expires_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

# Create indexes for performance
Index('idx_student_school', Student.school_id)
Index('idx_student_class', Student.class_id)
Index('idx_examresult_student_exam', ExamResult.student_id, ExamResult.exam_id)
Index('idx_attendance_student_year', Attendance.student_id, func.extract('year', Attendance.date))
Index('idx_ai_prediction_student_type', AIPrediction.student_id, AIPrediction.prediction_type)
Index('idx_ai_prediction_date', AIPrediction.prediction_date)
