"""
IntelliDesk Ticket Routes
Ticket CRUD operations for users
"""

from fastapi import APIRouter, Depends, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from config import get_db
from models import User
from auth import get_current_user
from services.ticket_service import (
    create_ticket, get_ticket, get_user_tickets,
    get_ticket_messages, add_message, update_ticket_status
)

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/dashboard", response_class=HTMLResponse)
async def user_dashboard(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    tickets = get_user_tickets(db, user.id)
    open_count = sum(1 for t in tickets if t.status.value == "open")
    in_progress_count = sum(1 for t in tickets if t.status.value == "in_progress")
    closed_count = sum(1 for t in tickets if t.status.value == "closed")

    return templates.TemplateResponse(
        "user_dashboard.html",
        {
            "request": request,
            "user": user,
            "tickets": tickets,
            "open_count": open_count,
            "in_progress_count": in_progress_count,
            "closed_count": closed_count
        }
    )


@router.get("/tickets/new", response_class=HTMLResponse)
async def new_ticket_page(
    request: Request,
    user: User = Depends(get_current_user)
):
    return templates.TemplateResponse(
        "create_ticket.html",
        {
            "request": request,
            "user": user
        }
    )


@router.post("/tickets/new")
async def submit_ticket(
    request: Request,
    subject: str = Form(...),
    description: str = Form(...),
    category: str = Form("general"),
    priority: str = Form("medium"),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    ticket = create_ticket(
        db=db,
        user_id=user.id,
        subject=subject,
        description=description,
        category=category,
        priority=priority
    )
    return RedirectResponse(url=f"/tickets/{ticket.id}", status_code=303)


@router.get("/tickets/{ticket_id}", response_class=HTMLResponse)
async def view_ticket(
    request: Request,
    ticket_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    ticket = get_ticket(db, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    if user.role.value != "admin" and ticket.user_id != user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    messages = get_ticket_messages(db, ticket_id)

    return templates.TemplateResponse(
        "view_ticket.html",
        {
            "request": request,
            "user": user,
            "ticket": ticket,
            "messages": messages
        }
    )


@router.post("/tickets/{ticket_id}/message")
async def add_ticket_message(
    ticket_id: int,
    content: str = Form(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    ticket = get_ticket(db, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    sender = "admin" if user.role.value == "admin" else "user"
    add_message(db, ticket_id, sender, content)

    return RedirectResponse(url=f"/tickets/{ticket_id}", status_code=303)