from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
from sqlalchemy.orm import Session
import uuid

from database import get_db
import models
from schemas import UserCreate, UserResponse, UserWithToken

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# API Key scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# JWT Configuration
SECRET_KEY = "your-secret-key-here-change-in-production"  # Change in production!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days
REFRESH_TOKEN_EXPIRE_DAYS = 30

class AuthService:
    def __init__(self, db: Session):
        self.db = db
    
    def create_user(self, user_data: UserCreate) -> models.User:
        """Create new user with hashed password"""
        # Check if user already exists
        existing_user = self.db.query(models.User).filter(
            (models.User.email == user_data.email) | 
            (models.User.username == user_data.username)
        ).first()
        
        if existing_user:
            if existing_user.email == user_data.email:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username already taken"
                )
        
        # Hash password
        hashed_password = pwd_context.hash(user_data.password)
        
        # Create user
        user = models.User(
            email=user_data.email,
            username=user_data.username,
            full_name=user_data.full_name,
            hashed_password=hashed_password,
            role=user_data.role,
            school_id=user_data.school_id,
            is_active=user_data.is_active,
            ai_preferences=user_data.ai_preferences or {"notifications": True, "personalized_insights": True}
        )
        
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        
        # Create default API key
        self.create_api_key(user.id, "Default Key")
        
        return user
    
    def authenticate_user(self, email: str, password: str) -> Optional[models.User]:
        """Authenticate user with email and password"""
        user = self.db.query(models.User).filter(models.User.email == email).first()
        
        if not user:
            return None
        
        if not pwd_context.verify(password, user.hashed_password):
            return None
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User account is inactive"
            )
        
        # Update last login
        user.last_login = datetime.utcnow()
        self.db.commit()
        
        return user
    
    def create_access_token(self, user: models.User, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = {
            "sub": str(user.id),
            "email": user.email,
            "role": user.role,
            "school_id": user.school_id,
            "exp": datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
        }
        
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    def create_refresh_token(self, user: models.User) -> str:
        """Create JWT refresh token"""
        to_encode = {
            "sub": str(user.id),
            "type": "refresh",
            "exp": datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        }
        
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    def create_tokens(self, user: models.User) -> Dict[str, str]:
        """Create both access and refresh tokens"""
        access_token = self.create_access_token(user)
        refresh_token = self.create_refresh_token(user)
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer"
        }
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    def refresh_access_token(self, refresh_token: str) -> Dict[str, str]:
        """Refresh access token using refresh token"""
        try:
            payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
            
            if payload.get("type") != "refresh":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid refresh token"
                )
            
            user_id = uuid.UUID(payload.get("sub"))
            user = self.db.query(models.User).filter(models.User.id == user_id).first()
            
            if not user or not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found or inactive"
                )
            
            access_token = self.create_access_token(user)
            
            return {
                "access_token": access_token,
                "token_type": "bearer"
            }
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
    
    def create_api_key(self, user_id: uuid.UUID, name: str = "Default") -> models.APIKey:
        """Create API key for user"""
        import secrets
        
        key = f"edutrack_{secrets.token_urlsafe(32)}"
        
        api_key = models.APIKey(
            key=key,
            name=name,
            user_id=user_id,
            permissions=["read", "write"],
            rate_limit=100,
            is_active=True,
            expires_at=datetime.utcnow() + timedelta(days=365)
        )
        
        self.db.add(api_key)
        self.db.commit()
        self.db.refresh(api_key)
        
        return api_key
    
    def verify_api_key(self, api_key: str) -> Optional[models.User]:
        """Verify API key and return user"""
        if not api_key:
            return None
        
        key_record = self.db.query(models.APIKey).filter(
            models.APIKey.key == api_key,
            models.APIKey.is_active == True,
            (models.APIKey.expires_at == None) | (models.APIKey.expires_at > datetime.utcnow())
        ).first()
        
        if not key_record:
            return None
        
        # Update last used
        key_record.last_used = datetime.utcnow()
        self.db.commit()
        
        return key_record.user
    
    def get_current_user(
        self, 
        token: Optional[str] = Depends(oauth2_scheme),
        api_key: Optional[str] = Depends(api_key_header),
        db: Session = Depends(get_db)
    ) -> models.User:
        """Get current user from token or API key"""
        user = None
        
        # Try API key first
        if api_key:
            auth_service = AuthService(db)
            user = auth_service.verify_api_key(api_key)
        
        # Try JWT token
        if not user and token:
            auth_service = AuthService(db)
            payload = auth_service.verify_token(token)
            user_id = uuid.UUID(payload.get("sub"))
            user = db.query(models.User).filter(models.User.id == user_id).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User account is inactive"
            )
        
        return user
    
    def require_role(self, *roles: str):
        """Decorator to require specific roles"""
        def role_checker(current_user: models.User = Depends(get_current_user)):
            if current_user.role not in roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions"
                )
            return current_user
        return role_checker
    
    def require_same_school(self, school_id: int):
        """Require user to be from same school or super admin"""
        def school_checker(current_user: models.User = Depends(get_current_user)):
            if current_user.role != "super_admin" and current_user.school_id != school_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Cannot access data from other schools"
                )
            return current_user
        return school_checker
    
    def can_access_student(self, student_id: int):
        """Check if user can access student data"""
        def access_checker(
            current_user: models.User = Depends(get_current_user),
            db: Session = Depends(get_db)
        ):
            if current_user.role == "super_admin":
                return current_user
            
            # Get student's school
            student = db.query(models.Student).filter(models.Student.id == student_id).first()
            if not student:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Student not found"
                )
            
            if current_user.school_id != student.school_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Cannot access student from other school"
                )
            
            # Teachers can only access their own students
            if current_user.role == "teacher":
                # Check if teacher teaches this student's class
                teacher_class = db.query(models.Class).filter(
                    models.Class.teacher_id == current_user.id,
                    models.Class.id == student.class_id
                ).first()
                
                if not teacher_class:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Can only access students in your classes"
                    )
            
            # Parents can only access their own children
            elif current_user.role == "parent":
                # In production, link parent to student
                # For now, allow if user email matches parent email
                if current_user.email.lower() != student.parent_email.lower():
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Can only access your own children"
                    )
            
            # Students can only access themselves
            elif current_user.role == "student":
                # Link user to student record (in production)
                # For now, check if username matches admission number
                if current_user.username != student.admission_number:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Can only access your own data"
                    )
            
            return current_user
        return access_checker

# Password utilities
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

# Rate limiting
import redis
from fastapi import Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Initialize Redis for rate limiting
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri="redis://localhost:6379",
    default_limits=["100/minute"]
)

def get_rate_limit_key(user: models.User) -> str:
    """Get rate limit key for user"""
    return f"rate_limit:{user.id}"

def check_rate_limit(user: models.User, endpoint: str) -> bool:
    """Check if user has exceeded rate limit"""
    key = f"{get_rate_limit_key(user)}:{endpoint}"
    current = redis_client.get(key)
    
    if not current:
        redis_client.setex(key, 60, 1)  # 1 request, expires in 60 seconds
        return True
    
    current = int(current)
    limit = user.api_keys[0].rate_limit if user.api_keys else 100
    
    if current >= limit:
        return False
    
    redis_client.incr(key)
    return True

# Audit logging
def log_audit_event(
    db: Session,
    user: models.User,
    action: str,
    resource_type: str = None,
    resource_id: str = None,
    details: Dict[str, Any] = None,
    status: str = "success",
    error_message: str = None
):
    """Log audit event"""
    audit_log = models.AuditLog(
        user_id=user.id,
        school_id=user.school_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details or {},
        ip_address="127.0.0.1",  # Get from request in production
        user_agent="API Client",  # Get from request in production
        status=status,
        error_message=error_message
    )
    
    db.add(audit_log)
    db.commit()
