"""
IntelliDesk Database Models
Tables: users, tickets, messages, chat_logs, analytics
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Text, Float, DateTime,
    ForeignKey, Enum, Date, Boolean
)
from sqlalchemy.orm import relationship
from config import Base
import enum


# ---------- Enums ----------

class UserRole(str, enum.Enum):
    USER = "user"
    ADMIN = "admin"

class TicketStatus(str, enum.Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    CLOSED = "closed"

class TicketPriority(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class TicketCategory(str, enum.Enum):
    NETWORK = "network"
    SOFTWARE = "software"
    HARDWARE = "hardware"
    ACCESS = "access"
    PASSWORD = "password"
    GENERAL = "general"

class SenderType(str, enum.Enum):
    USER = "user"
    AI = "ai"
    ADMIN = "admin"


# ---------- Models ----------

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    email = Column(String(150), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(Enum(UserRole), default=UserRole.USER, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Email verification
    is_verified = Column(Boolean, default=False, nullable=False)
    verification_token = Column(String(255), nullable=True)

    # Relationships
    tickets = relationship("Ticket", back_populates="user", cascade="all, delete-orphan")
    chat_logs = relationship("ChatLog", back_populates="user", cascade="all, delete-orphan")


class Ticket(Base):
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    subject = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    category = Column(Enum(TicketCategory), default=TicketCategory.GENERAL, nullable=False)
    priority = Column(Enum(TicketPriority), default=TicketPriority.MEDIUM, nullable=False)
    status = Column(Enum(TicketStatus), default=TicketStatus.OPEN, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="tickets")
    messages = relationship("Message", back_populates="ticket", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    ticket_id = Column(Integer, ForeignKey("tickets.id"), nullable=False)
    sender = Column(Enum(SenderType), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    ticket = relationship("Ticket", back_populates="messages")


class ChatLog(Base):
    __tablename__ = "chat_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    user_input = Column(Text, nullable=False)
    detected_intent = Column(String(100), nullable=True)
    confidence_score = Column(Float, nullable=True)
    ai_response = Column(Text, nullable=True)
    ticket_created = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="chat_logs")


class Analytics(Base):
    __tablename__ = "analytics"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    date = Column(Date, unique=True, nullable=False)
    total_tickets = Column(Integer, default=0)
    open_tickets = Column(Integer, default=0)
    closed_tickets = Column(Integer, default=0)
    ai_resolved_count = Column(Integer, default=0)
    escalated_count = Column(Integer, default=0)
    avg_resolution_hours = Column(Float, default=0.0)