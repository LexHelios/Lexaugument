"""
LexOS Vibe Coder - API Dependencies
Shared dependencies for authentication, database sessions, and security
"""
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext

from ..settings import settings
from ..memory.lmdb_store import memory_store

logger = logging.getLogger(__name__)

# Security setup
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Token blacklist (in production, use Redis or database)
token_blacklist = set()

class AuthenticationError(Exception):
    """Custom authentication error"""
    pass

class AuthorizationError(Exception):
    """Custom authorization error"""
    pass

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token
    
    Args:
        data: Data to encode in the token
        expires_delta: Token expiration time
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=settings.JWT_EXPIRATION_HOURS)
    
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.SECRET_KEY, 
        algorithm=settings.JWT_ALGORITHM
    )
    
    return encoded_jwt

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate password hash"""
    return pwd_context.hash(password)

async def verify_token(token: str) -> Dict[str, Any]:
    """
    Verify JWT token and return payload
    
    Args:
        token: JWT token to verify
        
    Returns:
        Token payload
        
    Raises:
        AuthenticationError: If token is invalid
    """
    try:
        # Check if token is blacklisted
        if token in token_blacklist:
            raise AuthenticationError("Token has been revoked")
        
        # Decode and verify token
        payload = jwt.decode(
            token, 
            settings.SECRET_KEY, 
            algorithms=[settings.JWT_ALGORITHM]
        )
        
        # Check expiration
        exp = payload.get("exp")
        if exp is None:
            raise AuthenticationError("Token missing expiration")
        
        if datetime.utcnow() > datetime.fromtimestamp(exp):
            raise AuthenticationError("Token has expired")
        
        return payload
        
    except JWTError as e:
        raise AuthenticationError(f"Invalid token: {str(e)}")

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """
    Get current user from JWT token
    
    Args:
        credentials: HTTP authorization credentials
        
    Returns:
        User information
        
    Raises:
        HTTPException: If authentication fails
    """
    try:
        token = credentials.credentials
        payload = await verify_token(token)
        
        user_id = payload.get("sub")
        if user_id is None:
            raise AuthenticationError("Token missing user ID")
        
        # In production, fetch user details from database
        user = {
            "user_id": user_id,
            "username": payload.get("username", "unknown"),
            "email": payload.get("email"),
            "roles": payload.get("roles", []),
            "permissions": payload.get("permissions", []),
            "is_active": payload.get("is_active", True),
            "token_issued_at": payload.get("iat"),
            "token_expires_at": payload.get("exp")
        }
        
        if not user["is_active"]:
            raise AuthenticationError("User account is disabled")
        
        return user
        
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"❌ Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_current_active_user(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get current active user (additional validation)
    
    Args:
        current_user: Current user from token
        
    Returns:
        Active user information
        
    Raises:
        HTTPException: If user is not active
    """
    if not current_user.get("is_active", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    return current_user

def require_permission(permission: str):
    """
    Dependency factory for permission-based access control
    
    Args:
        permission: Required permission
        
    Returns:
        Dependency function
    """
    async def permission_checker(
        current_user: Dict[str, Any] = Depends(get_current_active_user)
    ) -> Dict[str, Any]:
        user_permissions = current_user.get("permissions", [])
        user_roles = current_user.get("roles", [])
        
        # Check direct permission
        if permission in user_permissions:
            return current_user
        
        # Check role-based permissions (simplified)
        admin_permissions = ["*", "admin", "all"]
        if any(perm in user_permissions for perm in admin_permissions):
            return current_user
        
        # Check if user has admin role
        if "admin" in user_roles or "superuser" in user_roles:
            return current_user
        
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission '{permission}' required"
        )
    
    return permission_checker

async def require_role(role: str):
    """
    Dependency factory for role-based access control
    
    Args:
        role: Required role
        
    Returns:
        Dependency function
    """
    async def role_checker(
        current_user: Dict[str, Any] = Depends(get_current_active_user)
    ) -> Dict[str, Any]:
        user_roles = current_user.get("roles", [])
        
        if role not in user_roles:
            # Check for admin override
            if "admin" not in user_roles and "superuser" not in user_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Role '{role}' required"
                )
        
        return current_user
    
    return role_checker

async def get_db_session():
    """
    Get database session (placeholder for future database integration)
    
    Returns:
        Database session
    """
    # For now, return memory store
    # In production, this would return a proper database session
    return memory_store

async def validate_api_key(api_key: str) -> bool:
    """
    Validate API key for external integrations
    
    Args:
        api_key: API key to validate
        
    Returns:
        True if valid, False otherwise
    """
    # In production, validate against database
    # For now, simple validation
    valid_keys = [
        settings.SECRET_KEY,
        "lexos-api-key-dev",
        "lexos-api-key-prod"
    ]
    
    return api_key in valid_keys

async def get_api_key_user(api_key: str) -> Optional[Dict[str, Any]]:
    """
    Get user information from API key
    
    Args:
        api_key: API key
        
    Returns:
        User information or None
    """
    if await validate_api_key(api_key):
        return {
            "user_id": "api_user",
            "username": "api_user",
            "email": None,
            "roles": ["api"],
            "permissions": ["api_access"],
            "is_active": True,
            "auth_method": "api_key"
        }
    
    return None

async def optional_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[Dict[str, Any]]:
    """
    Optional authentication - returns user if authenticated, None otherwise
    
    Args:
        credentials: Optional HTTP authorization credentials
        
    Returns:
        User information or None
    """
    if not credentials:
        return None
    
    try:
        token = credentials.credentials
        
        # Try JWT token first
        try:
            payload = await verify_token(token)
            user_id = payload.get("sub")
            if user_id:
                return {
                    "user_id": user_id,
                    "username": payload.get("username", "unknown"),
                    "email": payload.get("email"),
                    "roles": payload.get("roles", []),
                    "permissions": payload.get("permissions", []),
                    "is_active": payload.get("is_active", True),
                    "auth_method": "jwt"
                }
        except AuthenticationError:
            pass
        
        # Try API key
        user = await get_api_key_user(token)
        if user:
            return user
        
        return None
        
    except Exception as e:
        logger.debug(f"Optional auth failed: {e}")
        return None

def revoke_token(token: str) -> None:
    """
    Revoke a JWT token by adding it to blacklist
    
    Args:
        token: Token to revoke
    """
    token_blacklist.add(token)
    logger.info(f"Token revoked: {token[:20]}...")

def cleanup_expired_tokens() -> None:
    """
    Clean up expired tokens from blacklist
    This should be called periodically
    """
    current_time = datetime.utcnow()
    expired_tokens = set()
    
    for token in token_blacklist:
        try:
            payload = jwt.decode(
                token, 
                settings.SECRET_KEY, 
                algorithms=[settings.JWT_ALGORITHM],
                options={"verify_exp": False}  # Don't verify expiration for cleanup
            )
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp) < current_time:
                expired_tokens.add(token)
        except:
            # If we can't decode the token, remove it
            expired_tokens.add(token)
    
    token_blacklist -= expired_tokens
    
    if expired_tokens:
        logger.info(f"Cleaned up {len(expired_tokens)} expired tokens")

# Rate limiting (simple in-memory implementation)
request_counts = {}
rate_limit_window = 60  # 1 minute

async def rate_limit(
    user_id: str, 
    max_requests: int = 100,
    window_seconds: int = 60
) -> bool:
    """
    Simple rate limiting
    
    Args:
        user_id: User identifier
        max_requests: Maximum requests per window
        window_seconds: Time window in seconds
        
    Returns:
        True if request is allowed, False if rate limited
    """
    current_time = datetime.utcnow()
    window_start = current_time - timedelta(seconds=window_seconds)
    
    # Clean old entries
    if user_id in request_counts:
        request_counts[user_id] = [
            req_time for req_time in request_counts[user_id]
            if req_time > window_start
        ]
    else:
        request_counts[user_id] = []
    
    # Check rate limit
    if len(request_counts[user_id]) >= max_requests:
        return False
    
    # Add current request
    request_counts[user_id].append(current_time)
    return True

async def check_rate_limit(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Rate limiting dependency
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User information
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    user_id = current_user["user_id"]
    
    if not await rate_limit(user_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    
    return current_user
