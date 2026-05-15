"""
IntelliDesk Chat Routes
AI Chatbot interface and processing — with Groq streaming
"""

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from typing import Optional
import httpx
import json

from config import get_db, settings
from models import User, ChatLog
from auth import get_current_user
from services.nlp_service import (
    detect_intent, get_ai_response, is_it_related,
    generate_off_topic_response, get_image_query,
    GROQ_LOADED, groq_client, SYSTEM_PROMPT,
    INTENT_TO_CATEGORY, INTENT_PRIORITY
)
from services.ticket_service import create_ticket

router = APIRouter()
templates = Jinja2Templates(directory="templates")


async def fetch_unsplash_image(query: str) -> Optional[str]:
    """Fetch a relevant image URL from Unsplash."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://api.unsplash.com/search/photos",
                params={"query": query, "per_page": 1, "orientation": "landscape"},
                headers={"Authorization": f"Client-ID {settings.UNSPLASH_ACCESS_KEY}"},
                timeout=5.0
            )
            data = resp.json()
            results = data.get("results", [])
            if results:
                return results[0]["urls"]["small"]
    except Exception as e:
        print(f"[Unsplash] Error: {e}")
    return None


@router.get("/chat", response_class=HTMLResponse)
async def chat_page(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    chat_history = (
        db.query(ChatLog)
        .filter(ChatLog.user_id == user.id)
        .order_by(ChatLog.created_at.desc())
        .limit(50)
        .all()
    )
    chat_history.reverse()
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "user": user,
        "chat_history": chat_history
    })


@router.post("/api/chat")
async def process_chat(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    body = await request.json()
    user_message = body.get("message", "").strip()
    if not user_message:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    intent, confidence = detect_intent(user_message)

    # Fetch recent history
    recent_logs = (
        db.query(ChatLog)
        .filter(ChatLog.user_id == user.id)
        .order_by(ChatLog.created_at.desc())
        .limit(6)
        .all()
    )
    history = [
        {"user": log.user_input, "assistant": log.ai_response}
        for log in reversed(recent_logs)
        if log.ai_response
    ]

    result = get_ai_response(user_message, intent, confidence, history)

    # Fetch image if suggested
    image_url = None
    if result.get("image_query"):
        image_url = await fetch_unsplash_image(result["image_query"])

    ticket_id = None
    if result["should_create_ticket"]:
        ticket = create_ticket(
            db=db,
            user_id=user.id,
            subject=f"[AI Chat] {intent.title()}",
            description=f"User message: {user_message}\n\nDetected intent: {intent}\nConfidence: {confidence}",
            category=result["category"],
            priority=result["priority"]
        )
        ticket_id = ticket.id
        if result["confident"]:
            result["response"] += f"\n\nI've also created ticket #{ticket.id} for our IT team to follow up."
        else:
            result["response"] += f"\n\nTicket #{ticket.id} has been created."

    # Save to chat log
    chat_log = ChatLog(
        user_id=user.id,
        user_input=user_message,
        detected_intent=intent,
        confidence_score=confidence,
        ai_response=result["response"],
        ticket_created=ticket_id
    )
    db.add(chat_log)
    db.commit()

    return JSONResponse({
        "response": result["response"],
        "intent": intent,
        "confidence": round(confidence * 100, 1),
        "ticket_id": ticket_id,
        "confident": result["confident"],
        "image_url": image_url
    })


@router.post("/api/chat/stream")
async def stream_chat(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    body = await request.json()
    user_message = body.get("message", "").strip()
    if not user_message:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    intent, confidence = detect_intent(user_message)

    # Fetch recent history
    recent_logs = (
        db.query(ChatLog)
        .filter(ChatLog.user_id == user.id)
        .order_by(ChatLog.created_at.desc())
        .limit(6)
        .all()
    )
    history = [
        {"user": log.user_input, "assistant": log.ai_response}
        for log in reversed(recent_logs)
        if log.ai_response
    ]

    # Follow-up detection
    followup_signals = ["how to", "how do", "what is", "what about", "narrow", "explain",
                        "in windows", "on mac", "more detail", "step by step", "tell me more",
                        "what if", "why", "which", "can you", "show me", "what does"]
    is_followup = history and any(s in user_message.lower() for s in followup_signals)

    # Only use non-streaming path for greetings
    auto_ticket_intents = ["access permission request"]
    no_ticket_intents = ["greeting or casual conversation"]

    if intent in no_ticket_intents:
        result = get_ai_response(user_message, intent, confidence, history)
        ticket_id = None
        if result["should_create_ticket"]:
            ticket = create_ticket(
                db=db, user_id=user.id,
                subject=f"[AI Chat] {intent.title()}",
                description=f"User message: {user_message}\n\nDetected intent: {intent}\nConfidence: {confidence}",
                category=result["category"], priority=result["priority"]
            )
            ticket_id = ticket.id
            result["response"] += f"\n\nTicket #{ticket.id} has been created."

        chat_log = ChatLog(
            user_id=user.id, user_input=user_message,
            detected_intent=intent, confidence_score=confidence,
            ai_response=result["response"], ticket_created=ticket_id
        )
        db.add(chat_log)
        db.commit()

        async def non_stream():
            yield f"data: {json.dumps({'type': 'meta', 'intent': intent, 'confidence': round(confidence*100,1), 'ticket_id': ticket_id})}\n\n"
            yield f"data: {json.dumps({'type': 'text', 'content': result['response']})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return StreamingResponse(non_stream(), media_type="text/event-stream")

    # ── Streaming path ──
    user_prompt = (
        f"IT issue category: {intent}\n\n"
        f"User's exact message: \"{user_message}\"\n\n"
        f"Give a specific, tailored response. Use markdown formatting: "
        f"**bold** for important terms, numbered lists for steps, and clear paragraphs."
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history[-6:]:
        messages.append({"role": "user", "content": h["user"]})
        messages.append({"role": "assistant", "content": h["assistant"]})
    messages.append({"role": "user", "content": user_prompt})

    full_response = []

    async def event_stream():
        nonlocal full_response
        try:
            # Send meta first
            yield f"data: {json.dumps({'type': 'meta', 'intent': intent, 'confidence': round(confidence*100,1), 'ticket_id': None})}\n\n"

            stream = groq_client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=800,
                stream=True
            )

            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_response.append(delta)
                    yield f"data: {json.dumps({'type': 'chunk', 'content': delta})}\n\n"

            # After streaming, handle ticket + image + save
            complete_response = "".join(full_response)
            ticket_id = None

            if intent in auto_ticket_intents:
                ticket = create_ticket(
                    db=db, user_id=user.id,
                    subject=f"[AI Chat] {intent.title()}",
                    description=f"User message: {user_message}\n\nDetected intent: {intent}\nConfidence: {confidence}",
                    category=INTENT_TO_CATEGORY.get(intent, "general"),
                    priority=INTENT_PRIORITY.get(intent, "low")
                )
                ticket_id = ticket.id

            # Save log
            chat_log = ChatLog(
                user_id=user.id, user_input=user_message,
                detected_intent=intent, confidence_score=confidence,
                ai_response=complete_response, ticket_created=ticket_id
            )
            db.add(chat_log)
            db.commit()

            # Image
            img_query_result = get_image_query(user_message, intent, is_it=True)
            image_url = None
            if img_query_result:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        "https://api.unsplash.com/search/photos",
                        params={"query": img_query_result, "per_page": 1, "orientation": "landscape"},
                        headers={"Authorization": f"Client-ID {settings.UNSPLASH_ACCESS_KEY}"},
                        timeout=5.0
                    )
                    results = resp.json().get("results", [])
                    if results:
                        image_url = results[0]["urls"]["small"]

            yield f"data: {json.dumps({'type': 'done', 'ticket_id': ticket_id, 'image_url': image_url})}\n\n"

        except Exception as e:
            print(f"[STREAM] Error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")