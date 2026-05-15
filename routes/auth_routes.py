"""
IntelliDesk Auth Routes
Login, Register, Logout + Email Verification
"""

import secrets
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from fastapi import APIRouter, Depends, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from config import get_db, settings
from models import User, UserRole
from auth import hash_password, verify_password, create_access_token, get_current_user_optional

router = APIRouter()
templates = Jinja2Templates(directory="templates")


# ── Email sender ───────────────────────────────────────────────────────────────
def send_verification_email(to_email: str, name: str, token: str):
    activation_link = f"{settings.BASE_URL}/verify-email?token={token}"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Activate your IntelliDesk account"
    msg["From"]    = settings.GMAIL_USER
    msg["To"]      = to_email

    html = f"""
    <html><body style="font-family: Arial, sans-serif; background: #f4f4f4; padding: 30px;">
      <div style="max-width: 520px; margin: auto; background: white; border-radius: 12px; padding: 32px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
        <div style="text-align:center; margin-bottom: 24px;">
          <span style="font-size: 2rem;">🤖</span>
          <h2 style="color: #185FA5; margin: 8px 0 0;">IntelliDesk</h2>
        </div>
        <h3 style="color: #1a1a1a;">Hi {name}, welcome aboard!</h3>
        <p style="color: #555; line-height: 1.6;">
          Thanks for signing up. Please click the button below to activate your account.
          This link will expire in <strong>24 hours</strong>.
        </p>
        <div style="text-align: center; margin: 32px 0;">
          <a href="{activation_link}"
             style="background: #185FA5; color: white; padding: 14px 32px; border-radius: 8px;
                    text-decoration: none; font-weight: 600; font-size: 1rem;">
            Activate My Account
          </a>
        </div>
        <p style="color: #999; font-size: 0.8rem; text-align: center;">
          If you didn't create an account, you can ignore this email.
        </p>
      </div>
    </body></html>
    """

    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(settings.GMAIL_USER, settings.GMAIL_APP_PASSWORD)
            server.sendmail(settings.GMAIL_USER, to_email, msg.as_string())
        print(f"[EMAIL] Verification email sent to {to_email}")
    except Exception as e:
        print(f"[EMAIL] Failed to send email: {e}")


# ── Login ──────────────────────────────────────────────────────────────────────
@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, user: User = Depends(get_current_user_optional)):
    if user:
        if user.role.value == "admin":
            return RedirectResponse(url="/admin", status_code=303)
        return RedirectResponse(url="/dashboard", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@router.post("/login")
async def login_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.email == email).first()

    if not user or not verify_password(password, user.password_hash):
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Invalid email or password"}
        )

    # Block unverified accounts
    if not user.is_verified:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Please verify your email before logging in. Check your inbox for the activation link."}
        )

    token = create_access_token(data={
        "user_id": user.id,
        "email": user.email,
        "role": user.role.value
    })

    response = RedirectResponse(
        url="/admin" if user.role.value == "admin" else "/dashboard",
        status_code=303
    )
    response.set_cookie(key="access_token", value=token, httponly=True, max_age=3600, samesite="lax")
    return response


# ── Register ───────────────────────────────────────────────────────────────────
@router.get("/register", response_class=HTMLResponse)
async def register_page(request: Request, user: User = Depends(get_current_user_optional)):
    if user:
        return RedirectResponse(url="/dashboard", status_code=303)
    return templates.TemplateResponse("register.html", {"request": request, "error": None})


@router.post("/register")
async def register_submit(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    db: Session = Depends(get_db)
):
    if password != confirm_password:
        return templates.TemplateResponse(
            "register.html",
            {"request": request, "error": "Passwords do not match"}
        )

    if len(password) < 6:
        return templates.TemplateResponse(
            "register.html",
            {"request": request, "error": "Password must be at least 6 characters"}
        )

    existing = db.query(User).filter(User.email == email).first()
    if existing:
        return templates.TemplateResponse(
            "register.html",
            {"request": request, "error": "An account with this email already exists"}
        )

    # Generate verification token
    token = secrets.token_urlsafe(32)

    new_user = User(
        name=name,
        email=email,
        password_hash=hash_password(password),
        role=UserRole.USER,
        is_verified=False,
        verification_token=token
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # Send activation email
    send_verification_email(email, name, token)

    return templates.TemplateResponse(
        "register.html",
        {
            "request": request,
            "error": None,
            "success": f"Account created! We've sent an activation link to {email}. Please check your inbox."
        }
    )


# ── Email verification ─────────────────────────────────────────────────────────
@router.get("/verify-email", response_class=HTMLResponse)
async def verify_email(request: Request, token: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.verification_token == token).first()

    if not user:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Invalid or expired activation link."}
        )

    user.is_verified = True
    user.verification_token = None
    db.commit()

    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": None, "success": "Account activated! You can now log in."}
    )


# ── Logout ─────────────────────────────────────────────────────────────────────
@router.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("access_token")
    return response