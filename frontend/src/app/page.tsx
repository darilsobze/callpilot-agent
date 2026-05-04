"use client";

import { useEffect, useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

type AppointmentStatus = "booked" | "cancelled" | "rescheduled";

type Appointment = {
  id: string;
  provider_id: string;
  user_id: string;
  start_time: string;
  end_time: string;
  status: AppointmentStatus;
  notes?: string | null;
};

type AgentStep = {
  timestamp: string;
  action: string;
  outcome: string;
  detail?: string | null;
  data?: Record<string, unknown> | null;
};

type ProviderCallStatus = {
  provider_id: string;
  provider_name: string;
  call_id?: string | null;
  status: "idle" | "ringing" | "connected" | "failed";
};

type Provider = {
  id: string;
  name: string;
};

type AgentChatResponse = {
  reply: string;
  steps: AgentStep[];
  provider_status?: ProviderCallStatus | null;
  appointment?: Appointment | null;
};

type AgentRunResponse = AgentChatResponse & {
  run_id: string;
  status: "pending" | "running" | "completed" | "error";
};

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  steps?: AgentStep[];
  runId?: string;
  appointment?: Appointment | null;
  status?: "pending" | "running" | "completed" | "error";
  stepsOpen?: boolean;
};

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";
const DEMO_USER_ID = "demo-user";

const STEP_DURATIONS: Record<string, number> = {
  provider_search: 5,
  call_start: 5,
  call_status: 15,
  in_call: 10,
  calendar_query: 5,
  calendar_validate: 5,
  approving_appointment: 5,
  book_appointment: 5
};

const getStepDuration = (step: AgentStep) =>
  STEP_DURATIONS[step.action] ?? 5;

const getCurrentStepIndex = (steps: AgentStep[]) => {
  if (steps.length === 0) return 0;
  const start = new Date(steps[0].timestamp).getTime();
  if (!Number.isFinite(start)) return Math.max(steps.length - 1, 0);
  let elapsed = Math.max(0, (Date.now() - start) / 1000);
  for (let i = 0; i < steps.length; i += 1) {
    const duration = getStepDuration(steps[i]);
    if (elapsed <= duration) return i;
    elapsed -= duration;
  }
  return steps.length - 1;
};

export default function Home() {
  const [chatInput, setChatInput] = useState("");
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [providerStatus, setProviderStatus] =
    useState<ProviderCallStatus | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [lastAppointmentId, setLastAppointmentId] = useState<string | null>(
    null
  );
  const [appointmentDisplayIds, setAppointmentDisplayIds] = useState<
    Record<string, boolean>
  >({});
  const [appointments, setAppointments] = useState<Appointment[]>([]);
  const [appointmentsError, setAppointmentsError] = useState<string | null>(
    null
  );
  const [googleConnected, setGoogleConnected] = useState(false);
  const [providerMap, setProviderMap] = useState<Record<string, string>>({});
  const [, setTick] = useState(0);

  const statusColor = useMemo(() => {
    switch (providerStatus?.status ?? "idle") {
      case "connected":
        return "bg-emerald-400";
      case "ringing":
        return "bg-amber-400";
      case "failed":
        return "bg-rose-400";
      default:
        return "bg-zinc-400";
    }
  }, [providerStatus]);

  const loadAppointments = async () => {
    try {
      const res = await fetch(
        `${API_BASE}/appointments?user_id=${DEMO_USER_ID}`
      );
      if (!res.ok) {
        throw new Error(`Request failed (${res.status})`);
      }
      const data = await res.json();
      setAppointments(data ?? []);
      setAppointmentsError(null);
    } catch (err) {
      setAppointmentsError(
        err instanceof Error ? err.message : "Failed to load appointments."
      );
    }
  };

  useEffect(() => {
    loadAppointments();
  }, []);

  useEffect(() => {
    const loadProviders = async () => {
      try {
        const res = await fetch(`${API_BASE}/providers`);
        if (!res.ok) return;
        const data = (await res.json()) as Provider[];
        const map: Record<string, string> = {};
        data.forEach((provider) => {
          map[provider.id] = provider.name;
        });
        setProviderMap(map);
      } catch {
        setProviderMap({});
      }
    };
    loadProviders();
  }, []);

  useEffect(() => {
    if (!activeRunId) return;

    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/agent/runs/${activeRunId}`);
        if (!res.ok) return;
        const data = (await res.json()) as AgentRunResponse;
        setProviderStatus(data.provider_status ?? null);
        setChatMessages((prev) =>
          prev.map((message) =>
            message.runId === data.run_id
              ? {
                  ...message,
                  content: data.reply,
                  steps: data.steps ?? [],
                  appointment: data.appointment ?? null,
                  status: data.status
                }
              : message
          )
        );
        if (data.status === "completed" || data.status === "error") {
          setActiveRunId(null);
          if (data.appointment?.id && data.appointment.id !== lastAppointmentId) {
            setLastAppointmentId(data.appointment.id);
            setTimeout(async () => {
              setAppointmentDisplayIds((prev) => ({
                ...prev,
                [data.appointment!.id]: true
              }));
              await loadAppointments();
            }, 45000);
          }
          clearInterval(interval);
        }
      } catch {
        clearInterval(interval);
      }
    }, 1500);

    return () => clearInterval(interval);
  }, [activeRunId, lastAppointmentId]);

  useEffect(() => {
    const loadGoogleStatus = async () => {
      try {
        const res = await fetch(
          `${API_BASE}/auth/google/status?user_id=${DEMO_USER_ID}`
        );
        if (!res.ok) return;
        const data = await res.json();
        setGoogleConnected(Boolean(data?.connected));
      } catch {
        setGoogleConnected(false);
      }
    };
    loadGoogleStatus();
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      setTick((value) => value + 1);
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const handleSend = async () => {
    if (!chatInput.trim()) return;
    const message = chatInput.trim();
    setChatInput("");
    setIsLoading(true);
    setError(null);
    setChatMessages((prev) => [
      ...prev,
      { id: `user-${Date.now()}`, role: "user", content: message }
    ]);

    try {
      const res = await fetch(`${API_BASE}/agent/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: DEMO_USER_ID,
          message
        })
      });
      if (!res.ok) {
        throw new Error(`Request failed (${res.status})`);
      }
      const data = (await res.json()) as AgentRunResponse;
      setChatMessages((prev) => [
        ...prev,
        {
          id: data.run_id,
          role: "assistant",
          content: data.reply,
          steps: data.steps ?? [],
          runId: data.run_id,
          appointment: data.appointment ?? null,
          status: data.status,
          stepsOpen: false
        }
      ]);
      setProviderStatus(data.provider_status ?? null);
      setActiveRunId(data.run_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to send message.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-[#0b0b0f] via-[#0f1117] to-[#13151c] px-6 py-16 text-foreground">
      <div className="mx-auto w-full max-w-3xl space-y-8">
        <header className="space-y-3">
          <p className="text-sm uppercase tracking-[0.3em] text-white/50">
            CallPilot
          </p>
          <h1 className="text-3xl font-semibold sm:text-4xl">
            Agent scheduling control
          </h1>
          <p className="text-white/60">
            Chat with the agent to search, call, and book appointments.
          </p>
        </header>

        <section className="rounded-2xl border border-white/10 bg-white/5 p-6 shadow-xl shadow-black/20">
          <div className="flex flex-col gap-4">
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex items-center gap-2 text-sm text-white/70">
                <span className={`h-2.5 w-2.5 rounded-full ${statusColor}`} />
                <span>
                  Provider status:{" "}
                  {providerStatus
                    ? `${providerStatus.provider_name} (${providerStatus.status})`
                    : "Idle"}
                </span>
              </div>
              {error ? (
                <span className="text-xs text-rose-300">{error}</span>
              ) : null}
            </div>

            <div className="max-h-[420px] space-y-4 overflow-y-auto rounded-xl border border-white/10 bg-black/20 p-4">
              {chatMessages.length === 0 ? (
                <p className="text-sm text-white/60">
                  No messages yet. Ask the agent to book a service.
                </p>
              ) : (
                chatMessages.map((message) => (
                  <div key={message.id} className="space-y-2">
                    <div
                      className={`rounded-xl px-4 py-3 text-sm ${
                        message.role === "user"
                          ? "bg-white/10 text-white"
                          : "bg-white/5 text-white/80"
                      }`}
                    >
                      <p className="text-xs uppercase tracking-[0.2em] text-white/40">
                        {message.role === "user" ? "You" : "Agent"}
                      </p>
                      <p className="mt-1">{message.content}</p>
                    </div>
                    {message.steps && message.steps.length > 0 ? (
                      <div className="rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-xs text-white/70">
                        <p className="text-xs uppercase tracking-[0.2em] text-white/40">
                          Steps
                        </p>
                        {(() => {
                          const index = getCurrentStepIndex(message.steps ?? []);
                          const step = message.steps?.[index];
                          if (!step) return null;
                          const duration = getStepDuration(step);
                          return (
                            <div>
                              Step {index + 1} working...{" "}
                              <span className="text-white/40">
                                ({duration}s)
                              </span>
                            </div>
                          );
                        })()}
                        {message.status === "completed" ||
                        message.status === "error" ? (
                          <div className="mt-2 space-y-2">
                            <Button
                              size="sm"
                              variant="secondary"
                              onClick={() => {
                                setChatMessages((prev) =>
                                  prev.map((item) =>
                                    item.id === message.id
                                      ? { ...item, stepsOpen: !item.stepsOpen }
                                      : item
                                  )
                                );
                              }}
                            >
                              {message.stepsOpen ? "Hide steps" : "STEPS"}
                            </Button>
                            {message.stepsOpen ? (
                              <div className="space-y-1 rounded-xl border border-white/10 bg-black/20 px-4 py-3 text-xs text-white/70">
                                {message.steps.map((step, index) => (
                                  <div key={`${step.action}-${index}`}>
                                    Step {index + 1}: {step.action}:{" "}
                                    {step.outcome}
                                    {step.detail ? ` — ${step.detail}` : ""}
                                  </div>
                                ))}
                              </div>
                            ) : null}
                          </div>
                        ) : null}
                      </div>
                    ) : null}
                    {message.appointment &&
                    appointmentDisplayIds[message.appointment.id] ? (
                      <div className="rounded-xl border border-emerald-400/30 bg-emerald-400/10 px-4 py-3 text-xs text-emerald-100">
                        Appointment booked:{" "}
                        {new Date(
                          message.appointment.start_time
                        ).toLocaleString()}{" "}
                        -{" "}
                        {new Date(
                          message.appointment.end_time
                        ).toLocaleTimeString()}
                      </div>
                    ) : null}
                  </div>
                ))
              )}
            </div>

            <div className="flex flex-col gap-3 sm:flex-row">
              <Input
                placeholder="Book a dentist next Tuesday afternoon"
                value={chatInput}
                onChange={(event) => setChatInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") handleSend();
                }}
              />
              <Button onClick={handleSend} disabled={isLoading}>
                {isLoading ? "Sending..." : "Send"}
              </Button>
            </div>
          </div>
        </section>

        <section className="rounded-2xl border border-white/10 bg-white/5 p-6 shadow-xl shadow-black/20">
          <div className="flex flex-col gap-4">
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <h2 className="text-lg font-semibold">Appointments</h2>
                <p className="text-sm text-white/60">
                  Connected to Google Calendar for real availability.
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-3">
                <span
                  className={`h-2.5 w-2.5 rounded-full ${
                    googleConnected ? "bg-emerald-400" : "bg-zinc-400"
                  }`}
                />
                <span className="text-xs text-white/60">
                  {googleConnected ? "Connected" : "Not connected"}
                </span>
                <Button
                  variant="secondary"
                  onClick={() => {
                    window.location.href = `${API_BASE}/auth/google/login?user_id=${DEMO_USER_ID}`;
                  }}
                >
                  {googleConnected ? "Reconnect" : "Connect Google Calendar"}
                </Button>
                <Button variant="secondary" onClick={loadAppointments}>
                  Refresh appointments
                </Button>
              </div>
            </div>

            {appointmentsError ? (
              <p className="text-sm text-rose-300">{appointmentsError}</p>
            ) : null}

            {appointments.length === 0 ? (
              <p className="text-sm text-white/60">
                No appointments booked yet.
              </p>
            ) : (
              <div className="space-y-3">
                {appointments.map((appointment) => (
                  <div
                    key={appointment.id}
                    className="rounded-xl border border-white/10 bg-white/5 p-4"
                  >
                    <div className="flex flex-col gap-1 text-sm">
                      <span className="text-white/70">
                        Provider:{" "}
                        {providerMap[appointment.provider_id] ??
                          appointment.provider_id}
                      </span>
                      <span>
                        {new Date(appointment.start_time).toLocaleString()} -{" "}
                        {new Date(appointment.end_time).toLocaleTimeString()}
                      </span>
                      <span className="text-white/60">
                        Status: {appointment.status}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </section>
      </div>
    </main>
  );
}
