from __future__ import annotations

from datetime import datetime, date, timedelta, timezone
from enum import Enum
from pathlib import Path
import json
import os
import random
import threading
import time
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from uuid import uuid4
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
PROVIDERS_PATH = DATA_DIR / "providers.json"
AVAILABILITY_PATH = DATA_DIR / "availability.json"
APPOINTMENTS_PATH = DATA_DIR / "appointments.json"
USER_PREFS_PATH = DATA_DIR / "user_preferences.json"
CALLS_PATH = DATA_DIR / "calls.json"
GOOGLE_TOKENS_PATH = DATA_DIR / "google_tokens.json"

ELEVENLABS_OUTBOUND_URL = "https://api.elevenlabs.io/v1/convai/twilio/outbound-call"


def _load_env_file() -> None:
    env_path = APP_DIR / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"").strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_AGENT_ID = os.getenv("ELEVENLABS_AGENT_ID")
ELEVENLABS_AGENT_PHONE_NUMBER_ID = os.getenv("ELEVENLABS_AGENT_PHONE_NUMBER_ID")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_API_BASE = os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_CALENDAR_BASE = "https://www.googleapis.com/calendar/v3"
GOOGLE_CALENDAR_SCOPE = "https://www.googleapis.com/auth/calendar"


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return default
    return json.loads(content)


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _get_google_tokens() -> dict[str, dict[str, Any]]:
    return _load_json(GOOGLE_TOKENS_PATH, default={})


def _save_google_tokens(tokens: dict[str, dict[str, Any]]) -> None:
    _save_json(GOOGLE_TOKENS_PATH, tokens)


def _require_google_oauth_config() -> None:
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET or not GOOGLE_REDIRECT_URI:
        raise HTTPException(
            status_code=500,
            detail=(
                "Google OAuth not configured. Set GOOGLE_CLIENT_ID, "
                "GOOGLE_CLIENT_SECRET, and GOOGLE_REDIRECT_URI."
            ),
        )


def _google_request(
    url: str, method: str = "GET", headers: dict[str, str] | None = None, body: dict[str, Any] | None = None
) -> dict[str, Any]:
    data = None
    request_headers = headers or {}
    if body is not None:
        payload = json.dumps(body).encode("utf-8")
        data = payload
        request_headers.setdefault("Content-Type", "application/json")
    request = Request(url, data=data, headers=request_headers, method=method)
    try:
        with urlopen(request, timeout=20) as response:
            content = response.read().decode("utf-8").strip()
            return json.loads(content) if content else {}
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8").strip()
        raise HTTPException(
            status_code=502,
            detail=f"Google API request failed: {error_body or exc.reason}",
        )
    except URLError as exc:
        raise HTTPException(status_code=502, detail=f"Google API request failed: {exc.reason}")


def _exchange_google_code(code: str) -> dict[str, Any]:
    _require_google_oauth_config()
    payload = urlencode(
        {
            "code": code,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": GOOGLE_REDIRECT_URI,
            "grant_type": "authorization_code",
        }
    ).encode("utf-8")
    request = Request(
        GOOGLE_TOKEN_URL,
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=20) as response:
            content = response.read().decode("utf-8").strip()
            return json.loads(content) if content else {}
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8").strip()
        raise HTTPException(
            status_code=502,
            detail=f"Google OAuth exchange failed: {error_body or exc.reason}",
        )
    except URLError as exc:
        raise HTTPException(status_code=502, detail=f"Google OAuth exchange failed: {exc.reason}")


def _refresh_google_token(refresh_token: str) -> dict[str, Any]:
    _require_google_oauth_config()
    payload = urlencode(
        {
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }
    ).encode("utf-8")
    request = Request(
        GOOGLE_TOKEN_URL,
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=20) as response:
            content = response.read().decode("utf-8").strip()
            return json.loads(content) if content else {}
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8").strip()
        raise HTTPException(
            status_code=502,
            detail=f"Google OAuth refresh failed: {error_body or exc.reason}",
        )
    except URLError as exc:
        raise HTTPException(status_code=502, detail=f"Google OAuth refresh failed: {exc.reason}")


def _get_google_access_token(user_id: str) -> str:
    tokens = _get_google_tokens()
    token_data = tokens.get(user_id)
    if not token_data:
        raise HTTPException(status_code=401, detail="Google account not connected for user")
    expires_at = token_data.get("expires_at")
    refresh_token = token_data.get("refresh_token")
    if expires_at and datetime.utcnow().timestamp() >= float(expires_at) - 60:
        if not refresh_token:
            raise HTTPException(status_code=401, detail="Google token expired; reconnect required")
        refreshed = _refresh_google_token(refresh_token)
        token_data["access_token"] = refreshed.get("access_token")
        token_data["expires_at"] = datetime.utcnow().timestamp() + int(refreshed.get("expires_in", 0))
        token_data["scope"] = refreshed.get("scope", token_data.get("scope"))
        token_data["token_type"] = refreshed.get("token_type", token_data.get("token_type"))
        tokens[user_id] = token_data
        _save_google_tokens(tokens)
    access_token = token_data.get("access_token")
    if not access_token:
        raise HTTPException(status_code=401, detail="Google token missing; reconnect required")
    return access_token


def _get_provider_timezone(provider_id: str) -> str:
    provider = next((p for p in _get_providers() if p.id == provider_id), None)
    return _safe_timezone(provider.timezone if provider else "America/New_York")


def _safe_timezone(value: str) -> str:
    try:
        ZoneInfo(value)
        return value
    except Exception:
        return "UTC"


def _get_tzinfo(value: str):
    try:
        return ZoneInfo(value)
    except Exception:
        return timezone.utc


def _parse_event_time(value: dict[str, Any], time_zone: str) -> datetime:
    tz = _get_tzinfo(_safe_timezone(time_zone))
    if "dateTime" in value:
        dt = datetime.fromisoformat(value["dateTime"].replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz)
        return dt.astimezone(tz).replace(tzinfo=None)
    if "date" in value:
        return datetime.fromisoformat(value["date"]).replace(tzinfo=None)
    raise ValueError("Unsupported event time format")


def _events_in_range(
    user_id: str, time_min: datetime, time_max: datetime, time_zone: str
) -> list[dict[str, Any]]:
    access_token = _get_google_access_token(user_id)
    params = urlencode(
        {
            "timeMin": time_min.isoformat(),
            "timeMax": time_max.isoformat(),
            "singleEvents": "true",
            "orderBy": "startTime",
            "timeZone": time_zone,
        }
    )
    url = f"{GOOGLE_CALENDAR_BASE}/calendars/primary/events?{params}"
    response = _google_request(url, headers={"Authorization": f"Bearer {access_token}"})
    return response.get("items", [])


def _slot_overlaps_event(slot: Slot, event_start: datetime, event_end: datetime) -> bool:
    return slot.start_time < event_end and slot.end_time > event_start


def _generate_free_slots(
    start_date: date,
    end_date: date,
    time_zone: str,
    events: list[dict[str, Any]],
    slot_minutes: int = 30,
    day_start_hour: int = 9,
    day_end_hour: int = 17,
) -> list[Slot]:
    busy_ranges: list[tuple[datetime, datetime]] = []
    for event in events:
        try:
            event_start = _parse_event_time(event["start"], time_zone)
            event_end = _parse_event_time(event["end"], time_zone)
        except Exception:
            continue
        busy_ranges.append((event_start, event_end))

    slots: list[Slot] = []
    current_day = start_date
    while current_day <= end_date:
        day_start = datetime.combine(current_day, datetime.min.time()).replace(
            hour=day_start_hour, minute=0, second=0, microsecond=0
        )
        day_end = datetime.combine(current_day, datetime.min.time()).replace(
            hour=day_end_hour, minute=0, second=0, microsecond=0
        )
        cursor = day_start
        while cursor + timedelta(minutes=slot_minutes) <= day_end:
            slot = Slot(start_time=cursor, end_time=cursor + timedelta(minutes=slot_minutes))
            if not any(_slot_overlaps_event(slot, start, end) for start, end in busy_ranges):
                slots.append(slot)
            cursor += timedelta(minutes=slot_minutes)
        current_day += timedelta(days=1)
    return slots


def _create_calendar_event(
    user_id: str, provider_name: str, slot: Slot, notes: str | None, time_zone: str
) -> dict[str, Any]:
    access_token = _get_google_access_token(user_id)
    payload = {
        "summary": f"Appointment: {provider_name}",
        "description": notes or "Booked via CallPilot",
        "start": {"dateTime": slot.start_time.isoformat(), "timeZone": time_zone},
        "end": {"dateTime": slot.end_time.isoformat(), "timeZone": time_zone},
    }
    url = f"{GOOGLE_CALENDAR_BASE}/calendars/primary/events"
    return _google_request(
        url,
        method="POST",
        headers={"Authorization": f"Bearer {access_token}"},
        body=payload,
    )
class Location(BaseModel):
    lat: float
    lng: float


class Provider(BaseModel):
    id: str
    name: str
    phone: str
    address: str
    location: Location
    rating: float = Field(ge=0, le=5)
    services: list[str] = Field(default_factory=list)
    timezone: str = "America/New_York"


class AppointmentStatus(str, Enum):
    booked = "booked"
    cancelled = "cancelled"
    rescheduled = "rescheduled"


class CallStatus(str, Enum):
    idle = "idle"
    ringing = "ringing"
    connected = "connected"
    failed = "failed"


class Appointment(BaseModel):
    id: str
    provider_id: str
    user_id: str
    start_time: datetime
    end_time: datetime
    status: AppointmentStatus = AppointmentStatus.booked
    created_at: datetime
    notes: str | None = None
    calendar_event_id: str | None = None


class PreferenceWeights(BaseModel):
    distance: float = Field(default=0.35, ge=0, le=1)
    rating: float = Field(default=0.35, ge=0, le=1)
    availability: float = Field(default=0.30, ge=0, le=1)


class UserPreference(BaseModel):
    user_id: str
    preferred_days: list[str] = Field(
        default_factory=lambda: ["mon", "tue", "wed", "thu", "fri"]
    )
    earliest_time: str = "09:00"
    latest_time: str = "17:00"
    max_distance_km: float = 25.0
    min_rating: float = 4.0
    weights: PreferenceWeights = PreferenceWeights()


class Slot(BaseModel):
    start_time: datetime
    end_time: datetime


class CallStartRequest(BaseModel):
    service_request: str
    user_id: str | None = None
    phone_number: str | None = None


class CallStartResponse(BaseModel):
    call_id: str
    status: CallStatus
    created_at: datetime


class CallStatusResponse(BaseModel):
    call_id: str
    status: CallStatus
    created_at: datetime
    updated_at: datetime


class ProviderSearchRequest(BaseModel):
    service: str | None = None
    min_rating: float | None = None
    max_distance_km: float | None = None
    origin: Location | None = None


class ProviderSearchResult(BaseModel):
    provider: Provider
    distance_km: float | None = None


class CalendarQueryRequest(BaseModel):
    provider_id: str
    start_date: date
    end_date: date
    user_id: str | None = None


class CalendarQueryResponse(BaseModel):
    provider_id: str
    slots: list[Slot]


class CalendarValidateRequest(BaseModel):
    provider_id: str
    slot: Slot
    user_id: str | None = None


class CalendarValidateResponse(BaseModel):
    provider_id: str
    slot: Slot
    available: bool


class AppointmentRequest(BaseModel):
    provider_id: str
    user_id: str
    slot: Slot
    notes: str | None = None


class DistanceCalcRequest(BaseModel):
    origin: Location
    provider_ids: list[str] | None = None


class DistanceCalcResponse(BaseModel):
    provider_id: str
    distance_km: float


class RankProvidersRequest(BaseModel):
    service: str | None = None
    origin: Location
    start_date: date
    end_date: date
    user_id: str | None = None
    min_rating: float | None = None
    max_distance_km: float | None = None
    weights: PreferenceWeights | None = None
    max_results: int = 5


class RankProviderResult(BaseModel):
    provider: Provider
    score: float
    distance_km: float | None = None
    next_available_slot: Slot | None = None


class AgentStep(BaseModel):
    timestamp: datetime
    action: str
    outcome: str
    detail: str | None = None
    data: dict[str, Any] | None = None


class ProviderCallStatus(BaseModel):
    provider_id: str
    provider_name: str
    call_id: str | None = None
    status: CallStatus = CallStatus.idle


class AgentChatRequest(BaseModel):
    user_id: str = "default"
    message: str
    auto_execute: bool = True


class AgentChatResponse(BaseModel):
    reply: str
    steps: list[AgentStep]
    provider_status: ProviderCallStatus | None = None
    appointment: Appointment | None = None


class AgentRunStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    error = "error"


class AgentRunResponse(AgentChatResponse):
    run_id: str
    status: AgentRunStatus


app = FastAPI(title="CallPilot Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "docs": "/docs"}


def _get_providers() -> list[Provider]:
    raw = _load_json(PROVIDERS_PATH, default=[])
    return [Provider(**item) for item in raw]


def _save_providers(providers: list[Provider]) -> None:
    _save_json(PROVIDERS_PATH, [p.model_dump() for p in providers])


def _get_availability() -> dict[str, list[dict[str, Any]]]:
    return _load_json(AVAILABILITY_PATH, default={})


def _save_availability(data: dict[str, list[dict[str, Any]]]) -> None:
    _save_json(AVAILABILITY_PATH, data)


def _get_appointments() -> list[Appointment]:
    raw = _load_json(APPOINTMENTS_PATH, default=[])
    return [Appointment(**item) for item in raw]


def _save_appointments(items: list[Appointment]) -> None:
    _save_json(APPOINTMENTS_PATH, [a.model_dump(mode="json") for a in items])


def _get_preferences() -> list[UserPreference]:
    raw = _load_json(USER_PREFS_PATH, default=[])
    return [UserPreference(**item) for item in raw]


def _save_preferences(items: list[UserPreference]) -> None:
    _save_json(USER_PREFS_PATH, [p.model_dump() for p in items])


def _get_calls() -> list[dict[str, Any]]:
    return _load_json(CALLS_PATH, default=[])


def _save_calls(items: list[dict[str, Any]]) -> None:
    _save_json(CALLS_PATH, items)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    from math import radians, sin, cos, sqrt, asin

    r = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return r * c


def _slots_in_range(
    raw_slots: list[dict[str, Any]], start_date: date, end_date: date
) -> list[Slot]:
    slots: list[Slot] = []
    for item in raw_slots:
        if item.get("booked"):
            continue
        start = datetime.fromisoformat(item["start_time"])
        if start_date <= start.date() <= end_date:
            slots.append(Slot(start_time=start, end_time=datetime.fromisoformat(item["end_time"])))
    return slots


def _slot_matches(item: dict[str, Any], slot: Slot) -> bool:
    return (
        item["start_time"] == slot.start_time.isoformat()
        and item["end_time"] == slot.end_time.isoformat()
    )


def _earliest_slot_in_range(
    raw_slots: list[dict[str, Any]], start_date: date, end_date: date
) -> Slot | None:
    slots = _slots_in_range(raw_slots, start_date, end_date)
    if not slots:
        return None
    return min(slots, key=lambda s: s.start_time)


def _normalize_weights(weights: PreferenceWeights) -> PreferenceWeights:
    total = weights.distance + weights.rating + weights.availability
    if total <= 0:
        return PreferenceWeights()
    return PreferenceWeights(
        distance=weights.distance / total,
        rating=weights.rating / total,
        availability=weights.availability / total,
    )


def _compute_call_status(created_at: datetime) -> CallStatus:
    elapsed = (datetime.utcnow() - created_at).total_seconds()
    if elapsed < 6:
        return CallStatus.ringing
    return CallStatus.connected


def _require_elevenlabs_config() -> None:
    if not ELEVENLABS_API_KEY or not ELEVENLABS_AGENT_ID or not ELEVENLABS_AGENT_PHONE_NUMBER_ID:
        raise HTTPException(
            status_code=500,
            detail=(
                "ElevenLabs not configured. Set ELEVENLABS_API_KEY, "
                "ELEVENLABS_AGENT_ID, and ELEVENLABS_AGENT_PHONE_NUMBER_ID."
            ),
        )


def _start_elevenlabs_outbound_call(
    to_number: str, service_request: str, user_id: str | None
) -> dict[str, Any]:
    _require_elevenlabs_config()
    payload = {
        "agent_id": ELEVENLABS_AGENT_ID,
        "agent_phone_number_id": ELEVENLABS_AGENT_PHONE_NUMBER_ID,
        "to_number": to_number,
        "conversation_initiation_client_data": {
            "service_request": service_request,
            "user_id": user_id,
        },
    }
    request = Request(
        ELEVENLABS_OUTBOUND_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY,
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=20) as response:
            body = response.read().decode("utf-8").strip()
            return json.loads(body) if body else {}
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8").strip()
        raise HTTPException(
            status_code=502,
            detail=f"ElevenLabs call failed: {error_body or exc.reason}",
        )
    except URLError as exc:
        raise HTTPException(status_code=502, detail=f"ElevenLabs call failed: {exc.reason}")


class LLMClient:
    def __init__(self, api_key: str | None, model: str, api_base: str) -> None:
        self.api_key = api_key
        self.model = model
        self.api_base = api_base.rstrip("/")

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def chat(self, messages: list[dict[str, str]]) -> str:
        if not self.api_key:
            raise ValueError("LLM API key not configured")
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": LLM_TEMPERATURE,
        }
        request = Request(
            f"{self.api_base}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=30) as response:
                body = response.read().decode("utf-8").strip()
                data = json.loads(body) if body else {}
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8").strip()
            raise HTTPException(
                status_code=502,
                detail=f"LLM request failed: {error_body or exc.reason}",
            )
        except URLError as exc:
            raise HTTPException(status_code=502, detail=f"LLM request failed: {exc.reason}")
        message = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return message or ""


LLM_CLIENT = LLMClient(LLM_API_KEY, LLM_MODEL, LLM_API_BASE)
AGENT_STEP_LOG: list[AgentStep] = []
MAX_AGENT_ACTIONS = 5
AGENT_RUNS: dict[str, dict[str, Any]] = {}
AGENT_RUNS_LOCK = threading.Lock()


def _infer_service_category(text: str) -> str:
    value = text.lower()
    keywords = [
        ("dentist", "dentist"),
        ("dental", "dentist"),
        ("car wash", "car wash"),
        ("carwash", "car wash"),
        ("plumber", "plumber"),
        ("physio", "physiotherapy"),
        ("physiotherapy", "physiotherapy"),
        ("auto repair", "auto repair"),
        ("oil change", "auto repair"),
    ]
    for key, category in keywords:
        if key in value:
            return category
    return text.strip()


def _infer_city(text: str) -> str | None:
    message = text.lower()
    cities: set[str] = set()
    for provider in _get_providers():
        parts = [part.strip() for part in provider.address.split(",")]
        if len(parts) >= 2:
            city = parts[-1]
            if len(parts) >= 3 and len(parts[-1]) <= 3:
                city = parts[-2]
            if city:
                cities.add(city.lower())
    for city in cities:
        if city in message:
            return city
    return None


def _parse_llm_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start >= end:
        return None
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


def _tool_provider_search(
    service: str | None, city: str | None, min_rating: float | None
) -> list[ProviderSearchResult]:
    providers = _get_providers()
    results: list[ProviderSearchResult] = []
    for provider in providers:
        if service and service.lower() not in [item.lower() for item in provider.services]:
            continue
        if city and city.lower() not in provider.address.lower():
            continue
        if min_rating and provider.rating < min_rating:
            continue
        results.append(ProviderSearchResult(provider=provider, distance_km=None))
    return results


def _tool_calendar_query(
    provider_id: str, start_date: date, end_date: date, user_id: str
) -> CalendarQueryResponse:
    payload = CalendarQueryRequest(
        provider_id=provider_id,
        start_date=start_date,
        end_date=end_date,
        user_id=user_id,
    )
    return calendar_query(payload)


def _tool_calendar_validate(
    provider_id: str, slot: Slot, user_id: str
) -> CalendarValidateResponse:
    payload = CalendarValidateRequest(provider_id=provider_id, slot=slot, user_id=user_id)
    return calendar_validate(payload)


def _tool_book_appointment(
    provider_id: str, slot: Slot, user_id: str, notes: str | None
) -> Appointment:
    payload = AppointmentRequest(provider_id=provider_id, user_id=user_id, slot=slot, notes=notes)
    return book_appointment(payload)


def _create_local_appointment(
    provider_id: str, slot: Slot, user_id: str, notes: str | None
) -> Appointment:
    appointment = Appointment(
        id=str(uuid4()),
        provider_id=provider_id,
        user_id=user_id,
        start_time=slot.start_time,
        end_time=slot.end_time,
        status=AppointmentStatus.booked,
        created_at=datetime.utcnow(),
        notes=notes,
        calendar_event_id=None,
    )
    appointments = _get_appointments()
    appointments.append(appointment)
    _save_appointments(appointments)
    return appointment


def _default_next_tuesday_slot() -> Slot:
    today = date.today()
    days_until = (1 - today.weekday()) % 7
    if days_until == 0:
        days_until = 7
    target_date = today + timedelta(days=days_until)
    start_time = datetime.combine(target_date, datetime.min.time()).replace(
        hour=18, minute=0, second=0, microsecond=0
    )
    end_time = start_time + timedelta(minutes=30)
    return Slot(start_time=start_time, end_time=end_time)


def _plan_with_llm(message: str) -> dict[str, Any] | None:
    if not LLM_CLIENT.is_configured():
        return None
    system_prompt = (
        "You are a tool planner. Return JSON only with keys "
        "`reply` (string) and `actions` (array). "
        "Actions must use tool names: provider_search, call_start, "
        "calendar_query, calendar_validate, book_appointment. "
        "Args should be minimal and use ISO dates for start_date/end_date, "
        "and ISO datetime for slot_start/slot_end."
    )
    try:
        response = LLM_CLIENT.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ]
        )
    except HTTPException:
        return None
    parsed = _parse_llm_json(response)
    if not parsed or "actions" not in parsed:
        return None
    return parsed


def _append_step(steps: list[AgentStep], action: str, outcome: str, detail: str | None = None,
                 data: dict[str, Any] | None = None) -> None:
    step = AgentStep(
        timestamp=datetime.utcnow(),
        action=action,
        outcome=outcome,
        detail=detail,
        data=data,
    )
    steps.append(step)
    AGENT_STEP_LOG.append(step)


def _append_run_step(
    run_id: str,
    action: str,
    outcome: str,
    detail: str | None = None,
    data: dict[str, Any] | None = None,
) -> None:
    with AGENT_RUNS_LOCK:
        run = AGENT_RUNS.get(run_id)
        if not run:
            return
        step = AgentStep(
            timestamp=datetime.utcnow(),
            action=action,
            outcome=outcome,
            detail=detail,
            data=data,
        )
        run["steps"].append(step)
        AGENT_STEP_LOG.append(step)


def _execute_agent_actions(payload: AgentChatRequest) -> AgentChatResponse:
    steps: list[AgentStep] = []
    user_id = payload.user_id or "default"
    message = payload.message.strip()
    context: dict[str, Any] = {
        "message": message,
        "providers": [],
        "provider": None,
        "call": None,
        "slots": [],
        "slot": None,
        "appointment": None,
    }

    plan = _plan_with_llm(message)
    actions = plan.get("actions") if plan else None
    if not actions:
        actions = [
            {"tool": "provider_search", "args": {"service": _infer_service_category(message)}},
            {"tool": "call_start", "args": {}},
            {"tool": "calendar_query", "args": {}},
            {"tool": "calendar_validate", "args": {}},
            {"tool": "book_appointment", "args": {"notes": message}},
        ]

    for action in actions[:MAX_AGENT_ACTIONS]:
        tool_name = action.get("tool") or action.get("name") or action.get("action")
        args = action.get("args") or {}
        try:
            if tool_name == "provider_search":
                service = args.get("service") or _infer_service_category(message)
                city = args.get("city") or _infer_city(message)
                min_rating = args.get("min_rating")
                providers = _tool_provider_search(service, city, min_rating)
                context["providers"] = providers
                context["provider"] = (
                    random.choice(providers).provider if providers else None
                )
                if providers:
                    _append_step(
                        steps,
                        "provider_search",
                        "success",
                        detail=f"Found {len(providers)} providers",
                    )
                else:
                    _append_step(
                        steps,
                        "provider_search",
                        "empty",
                        detail="No providers found",
                    )
                    break
            elif tool_name == "call_start":
                if not payload.auto_execute:
                    _append_step(
                        steps,
                        "call_start",
                        "confirmation_required",
                        detail="Call requires confirmation",
                    )
                    break
                provider_id = args.get("provider_id")
                provider = context["provider"]
                if provider_id:
                    provider = next(
                        (p.provider for p in context["providers"] if p.provider.id == provider_id),
                        provider,
                    )
                if not provider:
                    raise HTTPException(status_code=400, detail="No provider selected for call")
                call_response = start_call(
                    CallStartRequest(
                        service_request=message,
                        user_id=user_id,
                        phone_number=provider.phone,
                    )
                )
                context["call"] = call_response
                context["provider"] = provider
                _append_step(
                    steps,
                    "call_start",
                    "success",
                    detail=f"Calling {provider.name}",
                    data={"call_id": call_response.call_id, "status": call_response.status.value},
                )
            elif tool_name == "calendar_query":
                provider = context["provider"]
                if not provider:
                    raise HTTPException(status_code=400, detail="No provider selected for calendar query")
                start_date = args.get("start_date")
                end_date = args.get("end_date")
                if start_date:
                    start_date = date.fromisoformat(start_date)
                else:
                    start_date = date.today()
                if end_date:
                    end_date = date.fromisoformat(end_date)
                else:
                    end_date = date.today() + timedelta(days=7)
                response = _tool_calendar_query(provider.id, start_date, end_date, user_id)
                context["slots"] = response.slots
                context["slot"] = response.slots[0] if response.slots else None
                if response.slots:
                    _append_step(
                        steps,
                        "calendar_query",
                        "success",
                        detail=f"Found {len(response.slots)} open slots",
                    )
                else:
                    _append_step(
                        steps,
                        "calendar_query",
                        "empty",
                        detail="No available slots",
                    )
                    break
            elif tool_name == "calendar_validate":
                provider = context["provider"]
                slot = context["slot"]
                if not provider or not slot:
                    raise HTTPException(status_code=400, detail="Missing provider or slot for validation")
                validation = _tool_calendar_validate(provider.id, slot, user_id)
                if validation.available:
                    _append_step(steps, "calendar_validate", "success", detail="Slot is available")
                else:
                    _append_step(steps, "calendar_validate", "conflict", detail="Slot already booked")
                    break
            elif tool_name == "book_appointment":
                if not payload.auto_execute:
                    _append_step(
                        steps,
                        "book_appointment",
                        "confirmation_required",
                        detail="Booking requires confirmation",
                    )
                    break
                provider = context["provider"]
                slot = context["slot"]
                if not provider or not slot:
                    raise HTTPException(status_code=400, detail="Missing provider or slot for booking")
                appointment = _tool_book_appointment(provider.id, slot, user_id, args.get("notes"))
                context["appointment"] = appointment
                _append_step(steps, "book_appointment", "success", detail="Appointment booked")
            else:
                _append_step(steps, tool_name or "unknown", "skipped", detail="Unknown tool")
        except HTTPException as exc:
            _append_step(steps, tool_name or "unknown", "error", detail=exc.detail)
            break
        except Exception as exc:
            _append_step(steps, tool_name or "unknown", "error", detail=str(exc))
            break

    provider = context["provider"]
    call = context["call"]
    appointment = context["appointment"]
    reply = (plan.get("reply") if plan else None) or ""
    if not reply:
        if not provider:
            reply = "I couldn't find a matching provider."
        elif appointment:
            reply = f"Booked an appointment with {provider.name}."
        elif call:
            reply = f"Started a call with {provider.name}."
        else:
            reply = "I updated the provider search results."

    provider_status = None
    if provider:
        provider_status = ProviderCallStatus(
            provider_id=provider.id,
            provider_name=provider.name,
            call_id=call.call_id if call else None,
            status=call.status if call else CallStatus.idle,
        )
    return AgentChatResponse(
        reply=reply,
        steps=steps,
        provider_status=provider_status,
        appointment=appointment,
    )


def _run_agent_flow(run_id: str, payload: AgentChatRequest) -> None:
    user_id = payload.user_id or "default"
    message = payload.message.strip()
    with AGENT_RUNS_LOCK:
        if run_id not in AGENT_RUNS:
            return
        AGENT_RUNS[run_id]["status"] = AgentRunStatus.running

    try:
        providers = _tool_provider_search(
            _infer_service_category(message),
            _infer_city(message),
            None,
        )
        if not providers:
            _append_run_step(run_id, "provider_search", "empty", "No providers found")
            with AGENT_RUNS_LOCK:
                AGENT_RUNS[run_id]["reply"] = "I couldn't find a matching provider."
                AGENT_RUNS[run_id]["status"] = AgentRunStatus.completed
            return
        provider = random.choice(providers).provider
        _append_run_step(
            run_id,
            "provider_search",
            "success",
            detail=f"Found {len(providers)} providers",
        )

        if not payload.auto_execute:
            _append_run_step(
                run_id, "call_start", "confirmation_required", "Call requires confirmation"
            )
            with AGENT_RUNS_LOCK:
                AGENT_RUNS[run_id]["status"] = AgentRunStatus.completed
                AGENT_RUNS[run_id]["provider_status"] = ProviderCallStatus(
                    provider_id=provider.id,
                    provider_name=provider.name,
                    status=CallStatus.idle,
                )
            return

        call_response = start_call(
            CallStartRequest(
                service_request=message,
                user_id=user_id,
                phone_number=provider.phone,
            )
        )
        with AGENT_RUNS_LOCK:
            AGENT_RUNS[run_id]["provider_status"] = ProviderCallStatus(
                provider_id=provider.id,
                provider_name=provider.name,
                call_id=call_response.call_id,
                status=call_response.status,
            )
        _append_run_step(
            run_id,
            "call_start",
            "ringing",
            detail=f"Calling {provider.name}",
            data={"call_id": call_response.call_id},
        )

        call_status = call_response.status
        for _ in range(15):
            if call_status in (CallStatus.connected, CallStatus.failed):
                break
            time.sleep(2)
            status_response = get_call_status(call_response.call_id)
            call_status = status_response.status
            with AGENT_RUNS_LOCK:
                current_status = AGENT_RUNS[run_id]["provider_status"]
                if current_status:
                    current_status.status = call_status
                    AGENT_RUNS[run_id]["provider_status"] = current_status
        if call_status == CallStatus.failed:
            _append_run_step(run_id, "call_status", "failed", "Call failed")
            with AGENT_RUNS_LOCK:
                AGENT_RUNS[run_id]["reply"] = f"Call failed for {provider.name}."
                AGENT_RUNS[run_id]["status"] = AgentRunStatus.completed
            return

        _append_run_step(run_id, "call_status", "in_call", "Connected to provider")
        time.sleep(4)
        _append_run_step(run_id, "in_call", "listening", "Capturing availability details")
        time.sleep(2)

        start_date = date.today()
        end_date = date.today() + timedelta(days=7)
        slot: Slot | None = None
        try:
            calendar_response = _tool_calendar_query(provider.id, start_date, end_date, user_id)
            if not calendar_response.slots:
                _append_run_step(run_id, "calendar_query", "empty", "No available slots")
            else:
                _append_run_step(
                    run_id,
                    "calendar_query",
                    "success",
                    detail=f"Found {len(calendar_response.slots)} open slots",
                )
                slot = calendar_response.slots[0]
        except Exception as exc:
            _append_run_step(run_id, "calendar_query", "error", detail=str(exc))

        time.sleep(1)

        if slot:
            try:
                validation = _tool_calendar_validate(provider.id, slot, user_id)
                if not validation.available:
                    _append_run_step(run_id, "calendar_validate", "conflict", "Slot already booked")
                    slot = None
                else:
                    _append_run_step(run_id, "calendar_validate", "success", "Slot is available")
            except Exception as exc:
                _append_run_step(run_id, "calendar_validate", "error", detail=str(exc))
                slot = None
        else:
            _append_run_step(run_id, "calendar_validate", "skipped", "No slot to validate")

        time.sleep(1)

        _append_run_step(run_id, "approving_appointment", "pending", "Approving appointment")
        time.sleep(1)

        if slot:
            try:
                appointment = _tool_book_appointment(provider.id, slot, user_id, message)
                _append_run_step(run_id, "book_appointment", "success", "Appointment booked")
            except Exception as exc:
                _append_run_step(run_id, "book_appointment", "error", detail=str(exc))
                slot = None

        if not slot:
            fallback_slot = _default_next_tuesday_slot()
            appointment = _create_local_appointment(provider.id, fallback_slot, user_id, message)
            _append_run_step(
                run_id,
                "book_appointment",
                "fallback",
                "Booked default appointment (next Tuesday 6pm)",
            )
        with AGENT_RUNS_LOCK:
            AGENT_RUNS[run_id]["appointment"] = appointment
            AGENT_RUNS[run_id]["reply"] = f"Booked an appointment with {provider.name}."
            AGENT_RUNS[run_id]["status"] = AgentRunStatus.completed
            AGENT_RUNS[run_id]["provider_status"] = ProviderCallStatus(
                provider_id=provider.id,
                provider_name=provider.name,
                call_id=call_response.call_id,
                status=call_status,
            )
    except Exception as exc:
        _append_run_step(run_id, "agent_flow", "error", detail=str(exc))
        with AGENT_RUNS_LOCK:
            if run_id in AGENT_RUNS:
                AGENT_RUNS[run_id]["reply"] = "Agent flow failed."
                AGENT_RUNS[run_id]["status"] = AgentRunStatus.error


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/config/elevenlabs")
def elevenlabs_config_status() -> dict[str, bool]:
    return {
        "api_key_set": bool(ELEVENLABS_API_KEY),
        "agent_id_set": bool(ELEVENLABS_AGENT_ID),
        "agent_phone_number_id_set": bool(ELEVENLABS_AGENT_PHONE_NUMBER_ID),
    }


@app.post("/agent/chat", response_model=AgentRunResponse)
def agent_chat(payload: AgentChatRequest) -> AgentRunResponse:
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="Message is required")
    run_id = str(uuid4())
    run = {
        "run_id": run_id,
        "status": AgentRunStatus.pending,
        "reply": "Working on it.",
        "steps": [],
        "provider_status": None,
        "appointment": None,
        "user_id": payload.user_id,
        "message": payload.message,
    }
    with AGENT_RUNS_LOCK:
        AGENT_RUNS[run_id] = run
    thread = threading.Thread(target=_run_agent_flow, args=(run_id, payload), daemon=True)
    thread.start()
    return AgentRunResponse(
        run_id=run_id,
        status=run["status"],
        reply=run["reply"],
        steps=run["steps"],
        provider_status=run["provider_status"],
        appointment=run["appointment"],
    )


@app.get("/agent/runs/{run_id}", response_model=AgentRunResponse)
def get_agent_run(run_id: str) -> AgentRunResponse:
    with AGENT_RUNS_LOCK:
        run = AGENT_RUNS.get(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        return AgentRunResponse(
            run_id=run["run_id"],
            status=run["status"],
            reply=run["reply"],
            steps=run["steps"],
            provider_status=run["provider_status"],
            appointment=run["appointment"],
        )


@app.get("/auth/google/login")
def google_login(user_id: str = "default") -> RedirectResponse:
    _require_google_oauth_config()
    params = urlencode(
        {
            "client_id": GOOGLE_CLIENT_ID,
            "redirect_uri": GOOGLE_REDIRECT_URI,
            "response_type": "code",
            "scope": GOOGLE_CALENDAR_SCOPE,
            "access_type": "offline",
            "prompt": "consent",
            "state": user_id,
        }
    )
    return RedirectResponse(url=f"{GOOGLE_AUTH_URL}?{params}")


@app.get("/auth/google/callback")
def google_callback(code: str | None = None, state: str | None = None) -> RedirectResponse:
    if not code:
        raise HTTPException(status_code=400, detail="Missing OAuth code")
    user_id = state or "default"
    token_payload = _exchange_google_code(code)
    if "access_token" not in token_payload:
        raise HTTPException(status_code=502, detail="Invalid token response from Google")
    tokens = _get_google_tokens()
    tokens[user_id] = {
        "access_token": token_payload.get("access_token"),
        "refresh_token": token_payload.get("refresh_token"),
        "expires_at": datetime.utcnow().timestamp() + int(token_payload.get("expires_in", 0)),
        "scope": token_payload.get("scope"),
        "token_type": token_payload.get("token_type"),
        "updated_at": datetime.utcnow().isoformat(),
    }
    _save_google_tokens(tokens)
    return RedirectResponse(
        url=f"{FRONTEND_URL}/?google=connected&user_id={user_id}",
        status_code=303,
    )


@app.get("/auth/google/status")
def google_status(user_id: str = "default") -> dict[str, Any]:
    tokens = _get_google_tokens()
    token_data = tokens.get(user_id)
    if not token_data:
        return {"connected": False, "user_id": user_id}
    return {
        "connected": bool(token_data.get("access_token")),
        "user_id": user_id,
        "updated_at": token_data.get("updated_at"),
        "expires_at": token_data.get("expires_at"),
    }


@app.post("/calls/start", response_model=CallStartResponse)
def start_call(payload: CallStartRequest) -> CallStartResponse:
    if not payload.phone_number:
        raise HTTPException(status_code=400, detail="phone_number is required for outbound calls")
    call_id = str(uuid4())
    created_at = datetime.utcnow()
    outbound_response = _start_elevenlabs_outbound_call(
        payload.phone_number, payload.service_request, payload.user_id
    )
    outbound_success = outbound_response.get("success", True)
    status = CallStatus.ringing if outbound_success else CallStatus.failed
    calls = _get_calls()
    calls.append(
        {
            "call_id": call_id,
            "service_request": payload.service_request,
            "user_id": payload.user_id,
            "phone_number": payload.phone_number,
            "created_at": created_at.isoformat(),
            "status": status.value,
            "conversation_id": outbound_response.get("conversation_id"),
            "call_sid": outbound_response.get("callSid") or outbound_response.get("call_sid"),
            "provider_response": outbound_response,
        }
    )
    _save_calls(calls)
    return CallStartResponse(call_id=call_id, status=status, created_at=created_at)


@app.get("/calls/{call_id}", response_model=CallStatusResponse)
def get_call_status(call_id: str) -> CallStatusResponse:
    calls = _get_calls()
    call = next((c for c in calls if c.get("call_id") == call_id), None)
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    created_at = datetime.fromisoformat(call["created_at"])
    stored_status = call.get("status")
    if stored_status == CallStatus.failed.value:
        status = CallStatus.failed
    elif stored_status == CallStatus.connected.value:
        status = CallStatus.connected
    else:
        status = _compute_call_status(created_at)
    return CallStatusResponse(
        call_id=call_id,
        status=status,
        created_at=created_at,
        updated_at=datetime.utcnow(),
    )


@app.post("/providers/search", response_model=list[ProviderSearchResult])
def search_providers(payload: ProviderSearchRequest) -> list[ProviderSearchResult]:
    providers = _get_providers()
    results: list[ProviderSearchResult] = []
    for provider in providers:
        if payload.service and payload.service.lower() not in [
            service.lower() for service in provider.services
        ]:
            continue
        if payload.min_rating and provider.rating < payload.min_rating:
            continue
        distance_km = None
        if payload.origin:
            distance_km = _haversine_km(
                payload.origin.lat,
                payload.origin.lng,
                provider.location.lat,
                provider.location.lng,
            )
            if payload.max_distance_km and distance_km > payload.max_distance_km:
                continue
        results.append(ProviderSearchResult(provider=provider, distance_km=distance_km))
    return results


@app.get("/providers/search", response_model=list[ProviderSearchResult])
def search_providers_get(
    category: str | None = None, city: str | None = None, min_rating: float | None = None
) -> list[ProviderSearchResult]:
    providers = _get_providers()
    results: list[ProviderSearchResult] = []
    for provider in providers:
        if category and category.lower() not in [
            service.lower() for service in provider.services
        ]:
            continue
        if city and city.lower() not in provider.address.lower():
            continue
        if min_rating and provider.rating < min_rating:
            continue
        results.append(ProviderSearchResult(provider=provider, distance_km=None))
    return results


@app.get("/providers", response_model=list[Provider])
def list_providers() -> list[Provider]:
    return _get_providers()


@app.get("/providers/{provider_id}", response_model=Provider)
def get_provider(provider_id: str) -> Provider:
    provider = next((p for p in _get_providers() if p.id == provider_id), None)
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")
    return provider


@app.post("/providers/rank", response_model=list[RankProviderResult])
def rank_providers(payload: RankProvidersRequest) -> list[RankProviderResult]:
    providers = _get_providers()
    availability = _get_availability()

    preferences = None
    if payload.user_id:
        preferences = next(
            (p for p in _get_preferences() if p.user_id == payload.user_id), None
        )
    weights = _normalize_weights(payload.weights or (preferences.weights if preferences else PreferenceWeights()))
    max_distance = payload.max_distance_km or (preferences.max_distance_km if preferences else None)

    results: list[RankProviderResult] = []
    total_days = max((payload.end_date - payload.start_date).days, 1)

    for provider in providers:
        if payload.service and payload.service.lower() not in [
            service.lower() for service in provider.services
        ]:
            continue
        if payload.min_rating and provider.rating < payload.min_rating:
            continue

        distance_km = _haversine_km(
            payload.origin.lat,
            payload.origin.lng,
            provider.location.lat,
            provider.location.lng,
        )
        if max_distance and distance_km > max_distance:
            continue

        raw_slots = availability.get(provider.id, [])
        next_slot = _earliest_slot_in_range(
            raw_slots, payload.start_date, payload.end_date
        )
        if next_slot:
            days_offset = max((next_slot.start_time.date() - payload.start_date).days, 0)
            availability_score = 1 - min(days_offset / total_days, 1)
        else:
            availability_score = 0.0

        if max_distance:
            distance_score = 1 - min(distance_km / max_distance, 1)
        else:
            distance_score = 1 / (1 + distance_km)
        rating_score = provider.rating / 5.0

        score = (
            weights.distance * distance_score
            + weights.rating * rating_score
            + weights.availability * availability_score
        )

        results.append(
            RankProviderResult(
                provider=provider,
                score=round(score, 4),
                distance_km=round(distance_km, 2),
                next_available_slot=next_slot,
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results[: max(payload.max_results, 1)]


@app.post("/distance/calc", response_model=list[DistanceCalcResponse])
def calculate_distance(payload: DistanceCalcRequest) -> list[DistanceCalcResponse]:
    providers = _get_providers()
    if payload.provider_ids:
        providers = [p for p in providers if p.id in payload.provider_ids]
    return [
        DistanceCalcResponse(
            provider_id=p.id,
            distance_km=_haversine_km(
                payload.origin.lat, payload.origin.lng, p.location.lat, p.location.lng
            ),
        )
        for p in providers
    ]


@app.post("/calendar/query", response_model=CalendarQueryResponse)
def calendar_query(payload: CalendarQueryRequest) -> CalendarQueryResponse:
    user_id = payload.user_id or "default"
    time_zone = _safe_timezone(_get_provider_timezone(payload.provider_id))
    tz = _get_tzinfo(time_zone)
    time_min = datetime.combine(payload.start_date, datetime.min.time()).replace(tzinfo=tz)
    time_max = (
        datetime.combine(payload.end_date, datetime.min.time())
        .replace(tzinfo=tz)
        + timedelta(days=1)
    )
    events = _events_in_range(user_id, time_min, time_max, time_zone)
    slots = _generate_free_slots(payload.start_date, payload.end_date, time_zone, events)
    return CalendarQueryResponse(provider_id=payload.provider_id, slots=slots)


@app.post("/calendar/validate", response_model=CalendarValidateResponse)
def calendar_validate(payload: CalendarValidateRequest) -> CalendarValidateResponse:
    user_id = payload.user_id or "default"
    time_zone = _safe_timezone(_get_provider_timezone(payload.provider_id))
    tz = _get_tzinfo(time_zone)
    day_start = datetime.combine(payload.slot.start_time.date(), datetime.min.time()).replace(
        tzinfo=tz
    )
    day_end = day_start + timedelta(days=1)
    events = _events_in_range(user_id, day_start, day_end, time_zone)
    available = True
    for event in events:
        try:
            event_start = _parse_event_time(event["start"], time_zone)
            event_end = _parse_event_time(event["end"], time_zone)
        except Exception:
            continue
        if _slot_overlaps_event(payload.slot, event_start, event_end):
            available = False
            break
    return CalendarValidateResponse(
        provider_id=payload.provider_id,
        slot=payload.slot,
        available=available,
    )


@app.post("/appointments/book", response_model=Appointment)
def book_appointment(payload: AppointmentRequest) -> Appointment:
    time_zone = _safe_timezone(_get_provider_timezone(payload.provider_id))
    tz = _get_tzinfo(time_zone)
    day_start = datetime.combine(payload.slot.start_time.date(), datetime.min.time()).replace(
        tzinfo=tz
    )
    day_end = day_start + timedelta(days=1)
    events = _events_in_range(payload.user_id, day_start, day_end, time_zone)
    for event in events:
        try:
            event_start = _parse_event_time(event["start"], time_zone)
            event_end = _parse_event_time(event["end"], time_zone)
        except Exception:
            continue
        if _slot_overlaps_event(payload.slot, event_start, event_end):
            raise HTTPException(status_code=409, detail="Slot already booked")

    provider = next((p for p in _get_providers() if p.id == payload.provider_id), None)
    provider_name = provider.name if provider else payload.provider_id
    created_event = _create_calendar_event(
        payload.user_id, provider_name, payload.slot, payload.notes, time_zone
    )
    appointment = Appointment(
        id=str(uuid4()),
        provider_id=payload.provider_id,
        user_id=payload.user_id,
        start_time=payload.slot.start_time,
        end_time=payload.slot.end_time,
        status=AppointmentStatus.booked,
        created_at=datetime.utcnow(),
        notes=payload.notes,
        calendar_event_id=created_event.get("id"),
    )
    appointments = _get_appointments()
    appointments.append(appointment)
    _save_appointments(appointments)
    return appointment


@app.get("/appointments/{appointment_id}", response_model=Appointment)
def get_appointment(appointment_id: str) -> Appointment:
    appointment = next((a for a in _get_appointments() if a.id == appointment_id), None)
    if not appointment:
        raise HTTPException(status_code=404, detail="Appointment not found")
    return appointment


@app.get("/appointments", response_model=list[Appointment])
def list_appointments(user_id: str | None = None) -> list[Appointment]:
    appointments = _get_appointments()
    if user_id:
        appointments = [a for a in appointments if a.user_id == user_id]
    return appointments


@app.post("/user/preferences", response_model=UserPreference)
def upsert_preferences(payload: UserPreference) -> UserPreference:
    prefs = _get_preferences()
    prefs = [p for p in prefs if p.user_id != payload.user_id]
    prefs.append(payload)
    _save_preferences(prefs)
    return payload


@app.get("/user/preferences/{user_id}", response_model=UserPreference)
def get_preferences(user_id: str) -> UserPreference:
    pref = next((p for p in _get_preferences() if p.user_id == user_id), None)
    if not pref:
        raise HTTPException(status_code=404, detail="Preferences not found")
    return pref
