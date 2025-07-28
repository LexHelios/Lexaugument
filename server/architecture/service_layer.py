"""
🏗️ Service Layer and Repository Pattern 🏗️
JAI MAHAKAAL! Clean architecture for maintainable code
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Generic, TypeVar, Protocol
from dataclasses import dataclass
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

# Domain Models
T = TypeVar('T')

@dataclass
class User:
    """User domain model"""
    user_id: str
    username: str
    email: Optional[str] = None
    preferences: Dict[str, Any] = None
    created_at: datetime = None
    last_active: datetime = None

@dataclass
class Conversation:
    """Conversation domain model"""
    conversation_id: str
    user_id: str
    title: str
    messages: List[Dict[str, Any]] = None
    created_at: datetime = None
    updated_at: datetime = None

@dataclass
class ModelRequest:
    """Model request domain model"""
    request_id: str
    user_id: str
    model_name: str
    prompt: str
    parameters: Dict[str, Any]
    response: Optional[str] = None
    processing_time: Optional[float] = None
    created_at: datetime = None
    completed_at: Optional[datetime] = None

# Repository Interfaces
class Repository(ABC, Generic[T]):
    """Base repository interface"""
    
    @abstractmethod
    async def get_by_id(self, id: str) -> Optional[T]:
        """Get entity by ID"""
        pass
    
    @abstractmethod
    async def save(self, entity: T) -> T:
        """Save entity"""
        pass
    
    @abstractmethod
    async def delete(self, id: str) -> bool:
        """Delete entity by ID"""
        pass
    
    @abstractmethod
    async def list(self, limit: int = 100, offset: int = 0) -> List[T]:
        """List entities with pagination"""
        pass

class UserRepository(Repository[User]):
    """User repository interface"""
    
    @abstractmethod
    async def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        pass
    
    @abstractmethod
    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        pass
    
    @abstractmethod
    async def update_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """Update user preferences"""
        pass

class ConversationRepository(Repository[Conversation]):
    """Conversation repository interface"""
    
    @abstractmethod
    async def get_by_user(self, user_id: str, limit: int = 50) -> List[Conversation]:
        """Get conversations for user"""
        pass
    
    @abstractmethod
    async def add_message(self, conversation_id: str, message: Dict[str, Any]) -> bool:
        """Add message to conversation"""
        pass

class ModelRequestRepository(Repository[ModelRequest]):
    """Model request repository interface"""
    
    @abstractmethod
    async def get_by_user(self, user_id: str, limit: int = 100) -> List[ModelRequest]:
        """Get requests for user"""
        pass
    
    @abstractmethod
    async def get_by_model(self, model_name: str, limit: int = 100) -> List[ModelRequest]:
        """Get requests for specific model"""
        pass

# Repository Implementations
class InMemoryUserRepository(UserRepository):
    """In-memory user repository implementation"""
    
    def __init__(self):
        self._users: Dict[str, User] = {}
        self._username_index: Dict[str, str] = {}
        self._email_index: Dict[str, str] = {}
    
    async def get_by_id(self, user_id: str) -> Optional[User]:
        return self._users.get(user_id)
    
    async def get_by_username(self, username: str) -> Optional[User]:
        user_id = self._username_index.get(username)
        return self._users.get(user_id) if user_id else None
    
    async def get_by_email(self, email: str) -> Optional[User]:
        user_id = self._email_index.get(email)
        return self._users.get(user_id) if user_id else None
    
    async def save(self, user: User) -> User:
        if not user.user_id:
            user.user_id = str(uuid.uuid4())
        if not user.created_at:
            user.created_at = datetime.now()
        
        self._users[user.user_id] = user
        self._username_index[user.username] = user.user_id
        if user.email:
            self._email_index[user.email] = user.user_id
        
        return user
    
    async def delete(self, user_id: str) -> bool:
        user = self._users.get(user_id)
        if user:
            del self._users[user_id]
            self._username_index.pop(user.username, None)
            if user.email:
                self._email_index.pop(user.email, None)
            return True
        return False
    
    async def list(self, limit: int = 100, offset: int = 0) -> List[User]:
        users = list(self._users.values())
        return users[offset:offset + limit]
    
    async def update_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        user = self._users.get(user_id)
        if user:
            user.preferences = preferences
            return True
        return False

class InMemoryConversationRepository(ConversationRepository):
    """In-memory conversation repository implementation"""
    
    def __init__(self):
        self._conversations: Dict[str, Conversation] = {}
        self._user_index: Dict[str, List[str]] = {}
    
    async def get_by_id(self, conversation_id: str) -> Optional[Conversation]:
        return self._conversations.get(conversation_id)
    
    async def get_by_user(self, user_id: str, limit: int = 50) -> List[Conversation]:
        conversation_ids = self._user_index.get(user_id, [])
        conversations = [self._conversations[cid] for cid in conversation_ids if cid in self._conversations]
        return sorted(conversations, key=lambda c: c.updated_at or c.created_at, reverse=True)[:limit]
    
    async def save(self, conversation: Conversation) -> Conversation:
        if not conversation.conversation_id:
            conversation.conversation_id = str(uuid.uuid4())
        if not conversation.created_at:
            conversation.created_at = datetime.now()
        conversation.updated_at = datetime.now()
        
        self._conversations[conversation.conversation_id] = conversation
        
        # Update user index
        if conversation.user_id not in self._user_index:
            self._user_index[conversation.user_id] = []
        if conversation.conversation_id not in self._user_index[conversation.user_id]:
            self._user_index[conversation.user_id].append(conversation.conversation_id)
        
        return conversation
    
    async def delete(self, conversation_id: str) -> bool:
        conversation = self._conversations.get(conversation_id)
        if conversation:
            del self._conversations[conversation_id]
            # Remove from user index
            user_conversations = self._user_index.get(conversation.user_id, [])
            if conversation_id in user_conversations:
                user_conversations.remove(conversation_id)
            return True
        return False
    
    async def list(self, limit: int = 100, offset: int = 0) -> List[Conversation]:
        conversations = list(self._conversations.values())
        return conversations[offset:offset + limit]
    
    async def add_message(self, conversation_id: str, message: Dict[str, Any]) -> bool:
        conversation = self._conversations.get(conversation_id)
        if conversation:
            if not conversation.messages:
                conversation.messages = []
            conversation.messages.append(message)
            conversation.updated_at = datetime.now()
            return True
        return False

# Service Layer
class UserService:
    """User service with business logic"""
    
    def __init__(self, user_repo: UserRepository):
        self.user_repo = user_repo
    
    async def create_user(self, username: str, email: str = None) -> User:
        """Create a new user"""
        # Check if username already exists
        existing_user = await self.user_repo.get_by_username(username)
        if existing_user:
            raise ValueError(f"Username '{username}' already exists")
        
        # Check if email already exists
        if email:
            existing_email = await self.user_repo.get_by_email(email)
            if existing_email:
                raise ValueError(f"Email '{email}' already exists")
        
        # Create user
        user = User(
            user_id=str(uuid.uuid4()),
            username=username,
            email=email,
            preferences={},
            created_at=datetime.now(),
            last_active=datetime.now()
        )
        
        return await self.user_repo.save(user)
    
    async def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return await self.user_repo.get_by_id(user_id)
    
    async def update_user_activity(self, user_id: str) -> bool:
        """Update user's last active timestamp"""
        user = await self.user_repo.get_by_id(user_id)
        if user:
            user.last_active = datetime.now()
            await self.user_repo.save(user)
            return True
        return False
    
    async def update_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """Update user preferences"""
        return await self.user_repo.update_preferences(user_id, preferences)

class ConversationService:
    """Conversation service with business logic"""
    
    def __init__(self, conversation_repo: ConversationRepository, user_service: UserService):
        self.conversation_repo = conversation_repo
        self.user_service = user_service
    
    async def create_conversation(self, user_id: str, title: str) -> Conversation:
        """Create a new conversation"""
        # Verify user exists
        user = await self.user_service.get_user(user_id)
        if not user:
            raise ValueError(f"User '{user_id}' not found")
        
        conversation = Conversation(
            conversation_id=str(uuid.uuid4()),
            user_id=user_id,
            title=title,
            messages=[],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        return await self.conversation_repo.save(conversation)
    
    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Add message to conversation"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        return await self.conversation_repo.add_message(conversation_id, message)
    
    async def get_user_conversations(self, user_id: str, limit: int = 50) -> List[Conversation]:
        """Get conversations for user"""
        return await self.conversation_repo.get_by_user(user_id, limit)

class ModelRequestService:
    """Model request service with business logic"""
    
    def __init__(self, model_repo: ModelRequestRepository):
        self.model_repo = model_repo
    
    async def create_request(
        self,
        user_id: str,
        model_name: str,
        prompt: str,
        parameters: Dict[str, Any] = None
    ) -> ModelRequest:
        """Create a new model request"""
        request = ModelRequest(
            request_id=str(uuid.uuid4()),
            user_id=user_id,
            model_name=model_name,
            prompt=prompt,
            parameters=parameters or {},
            created_at=datetime.now()
        )
        
        return await self.model_repo.save(request)
    
    async def complete_request(
        self,
        request_id: str,
        response: str,
        processing_time: float
    ) -> bool:
        """Mark request as completed"""
        request = await self.model_repo.get_by_id(request_id)
        if request:
            request.response = response
            request.processing_time = processing_time
            request.completed_at = datetime.now()
            await self.model_repo.save(request)
            return True
        return False

# Dependency Injection Container
class ServiceContainer:
    """Simple dependency injection container"""
    
    def __init__(self):
        self._services = {}
        self._repositories = {}
        self._setup_dependencies()
    
    def _setup_dependencies(self):
        """Setup default dependencies"""
        # Repositories
        self._repositories['user'] = InMemoryUserRepository()
        self._repositories['conversation'] = InMemoryConversationRepository()
        
        # Services
        self._services['user'] = UserService(self._repositories['user'])
        self._services['conversation'] = ConversationService(
            self._repositories['conversation'],
            self._services['user']
        )
    
    def get_service(self, name: str):
        """Get service by name"""
        return self._services.get(name)
    
    def get_repository(self, name: str):
        """Get repository by name"""
        return self._repositories.get(name)

# Global container
service_container = ServiceContainer()
