"""
IntelliDesk Admin Routes
Admin dashboard, ticket management, and analytics
"""

from fastapi import APIRouter, Depends, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from config import get_db
from models import User
from auth import require_admin
from services.ticket_service import (
    get_all_tickets, get_ticket, update_ticket_status,
    update_ticket_priority, add_message, get_ticket_messages,
    get_analytics_data
)

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(
    request: Request,
    status: str = "all",
    category: str = "all",
    priority: str = "all",
    user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    tickets = get_all_tickets(
        db,
        status_filter=status,
        category_filter=category,
        priority_filter=priority
    )

    total = len(tickets)
    open_count = sum(1 for t in tickets if t.status.value == "open")
    in_progress = sum(1 for t in tickets if t.status.value == "in_progress")
    closed = sum(1 for t in tickets if t.status.value == "closed")

    all_tickets = get_all_tickets(db)
    total_all = len(all_tickets)
    open_all = sum(1 for t in all_tickets if t.status.value == "open")
    in_progress_all = sum(1 for t in all_tickets if t.status.value == "in_progress")
    closed_all = sum(1 for t in all_tickets if t.status.value == "closed")

    return templates.TemplateResponse(
        "admin_dashboard.html",
        {
            "request": request,
            "user": user,
            "tickets": tickets,
            "total": total_all,
            "open_count": open_all,
            "in_progress_count": in_progress_all,
            "closed_count": closed_all,
            "current_status": status,
            "current_category": category,
            "current_priority": priority
        }
    )


@router.post("/admin/tickets/{ticket_id}/status")
async def change_ticket_status(
    ticket_id: int,
    status: str = Form(...),
    user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    update_ticket_status(db, ticket_id, status)
    return RedirectResponse(url=f"/tickets/{ticket_id}", status_code=303)


@router.post("/admin/tickets/{ticket_id}/priority")
async def change_ticket_priority(
    ticket_id: int,
    priority: str = Form(...),
    user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    update_ticket_priority(db, ticket_id, priority)
    return RedirectResponse(url=f"/tickets/{ticket_id}", status_code=303)


@router.get("/analytics", response_class=HTMLResponse)
async def analytics_page(
    request: Request,
    user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    data = get_analytics_data(db)

    return templates.TemplateResponse(
        "analytics.html",
        {
            "request": request,
            "user": user,
            "analytics": data
        }
    )


@router.get("/api/analytics")
async def analytics_api(
    user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    data = get_analytics_data(db)
    data["recent_tickets"] = [
        {
            "id": t.id,
            "subject": t.subject,
            "category": t.category.value,
            "priority": t.priority.value,
            "status": t.status.value,
            "created_at": t.created_at.isoformat()
        }
        for t in data["recent_tickets"]
    ]
    return JSONResponse(data)