"""
IntelliDesk Ticket Service
Business logic for ticket CRUD operations
"""

from datetime import datetime
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from models import Ticket, Message, User, TicketStatus, TicketPriority, TicketCategory, SenderType


def create_ticket(
    db: Session,
    user_id: int,
    subject: str,
    description: str,
    category: str = "general",
    priority: str = "medium"
) -> Ticket:
    """Create a new support ticket."""
    ticket = Ticket(
        user_id=user_id,
        subject=subject,
        description=description,
        category=TicketCategory(category),
        priority=TicketPriority(priority),
        status=TicketStatus.OPEN
    )
    db.add(ticket)
    db.commit()
    db.refresh(ticket)

    # Add the initial description as the first message
    initial_message = Message(
        ticket_id=ticket.id,
        sender=SenderType.USER,
        content=description
    )
    db.add(initial_message)
    db.commit()

    return ticket


def get_ticket(db: Session, ticket_id: int) -> Optional[Ticket]:
    """Get a ticket by ID."""
    return db.query(Ticket).filter(Ticket.id == ticket_id).first()


def get_user_tickets(db: Session, user_id: int) -> List[Ticket]:
    """Get all tickets for a specific user."""
    return (
        db.query(Ticket)
        .filter(Ticket.user_id == user_id)
        .order_by(desc(Ticket.created_at))
        .all()
    )


def get_all_tickets(
    db: Session,
    status_filter: Optional[str] = None,
    category_filter: Optional[str] = None,
    priority_filter: Optional[str] = None
) -> List[Ticket]:
    """Get all tickets with optional filters (admin)."""
    query = db.query(Ticket)

    if status_filter and status_filter != "all":
        query = query.filter(Ticket.status == TicketStatus(status_filter))
    if category_filter and category_filter != "all":
        query = query.filter(Ticket.category == TicketCategory(category_filter))
    if priority_filter and priority_filter != "all":
        query = query.filter(Ticket.priority == TicketPriority(priority_filter))

    return query.order_by(desc(Ticket.created_at)).all()


def update_ticket_status(db: Session, ticket_id: int, status: str) -> Optional[Ticket]:
    """Update ticket status."""
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if ticket:
        ticket.status = TicketStatus(status)
        ticket.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(ticket)
    return ticket


def update_ticket_priority(db: Session, ticket_id: int, priority: str) -> Optional[Ticket]:
    """Update ticket priority."""
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if ticket:
        ticket.priority = TicketPriority(priority)
        ticket.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(ticket)
    return ticket


def add_message(db: Session, ticket_id: int, sender: str, content: str) -> Message:
    """Add a message to a ticket."""
    message = Message(
        ticket_id=ticket_id,
        sender=SenderType(sender),
        content=content
    )
    db.add(message)
    db.commit()
    db.refresh(message)
    return message


def get_ticket_messages(db: Session, ticket_id: int) -> List[Message]:
    """Get all messages for a ticket."""
    return (
        db.query(Message)
        .filter(Message.ticket_id == ticket_id)
        .order_by(Message.created_at)
        .all()
    )


def get_analytics_data(db: Session) -> dict:
    """Generate analytics data from tickets."""
    total = db.query(Ticket).count()
    open_count = db.query(Ticket).filter(Ticket.status == TicketStatus.OPEN).count()
    in_progress = db.query(Ticket).filter(Ticket.status == TicketStatus.IN_PROGRESS).count()
    closed = db.query(Ticket).filter(Ticket.status == TicketStatus.CLOSED).count()

    # Tickets by category
    by_category = {}
    for cat in TicketCategory:
        count = db.query(Ticket).filter(Ticket.category == cat).count()
        by_category[cat.value] = count

    # Tickets by priority
    by_priority = {}
    for pri in TicketPriority:
        count = db.query(Ticket).filter(Ticket.priority == pri).count()
        by_priority[pri.value] = count

    # Recent tickets
    recent = (
        db.query(Ticket)
        .order_by(desc(Ticket.created_at))
        .limit(10)
        .all()
    )

    # Count AI-resolved (tickets created from chat that were auto-resolved)
    from models import ChatLog
    ai_resolved = db.query(ChatLog).filter(ChatLog.ticket_created.is_(None)).count()
    escalated = db.query(ChatLog).filter(ChatLog.ticket_created.isnot(None)).count()

    return {
        "total_tickets": total,
        "open_tickets": open_count,
        "in_progress_tickets": in_progress,
        "closed_tickets": closed,
        "ai_resolved": ai_resolved,
        "escalated": escalated,
        "avg_resolution_hours": 0.0,
        "tickets_by_category": by_category,
        "tickets_by_priority": by_priority,
        "recent_tickets": recent
    }
