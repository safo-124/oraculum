import { useEffect, useState } from "react";
import { ChevronRight, Loader2 } from "lucide-react";
import { api } from "../../api/client";
import type { ActivityDay, AdminActivity } from "../../api/types";

const RANGES = [7, 14, 30] as const;
const CHART_W = 640;
const CHART_H = 150;
const PAD = 8;

function fmtDate(iso: string | null): string {
  if (!iso) return "–";
  return new Date(iso).toLocaleDateString(undefined, { day: "numeric", month: "short" });
}

function fmtWhen(iso: string | null): string {
  if (!iso) return "never";
  const minutes = Math.round((Date.now() - new Date(iso).getTime()) / 60000);
  if (minutes < 60) return `${Math.max(minutes, 1)} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} h ago`;
  return fmtDate(iso);
}

/** One step of the funnel: a big number, a label and a one-line definition. */
function Step({ value, label, hint }: { value: number; label: string; hint: string }) {
  return (
    <div className="min-w-0 flex-1 rounded-2xl border border-line bg-panel px-4 py-3">
      <p className="text-2xl font-semibold tabular-nums">{value.toLocaleString()}</p>
      <p className="text-sm font-medium">{label}</p>
      <p className="mt-0.5 text-xs text-faint">{hint}</p>
    </div>
  );
}

/** Single-series daily bars; the tooltip carries the breakdown. */
function DailyBars({
  title,
  days,
  value,
  tooltip,
}: {
  title: string;
  days: ActivityDay[];
  value: (d: ActivityDay) => number;
  tooltip: (d: ActivityDay) => string;
}) {
  const values = days.map(value);
  const max = Math.max(...values, 1);
  const total = values.reduce((a, b) => a + b, 0);
  const slot = (CHART_W - PAD * 2) / Math.max(days.length, 1);
  const barW = Math.max(slot * 0.6, 3);
  const labelEvery = Math.max(Math.ceil(days.length / 7), 1);
  return (
    <div className="rounded-2xl border border-line bg-panel p-4">
      <div className="flex items-baseline justify-between">
        <h3 className="text-sm font-medium">{title}</h3>
        <span className="text-xs text-faint">{total.toLocaleString()} in total</span>
      </div>
      {total === 0 ? (
        <div className="mt-2 flex h-32 items-center justify-center rounded-xl border border-dashed border-line text-sm text-faint">
          Nothing yet in this period
        </div>
      ) : (
        <svg viewBox={`0 0 ${CHART_W} ${CHART_H}`} className="mt-2 w-full" role="img" aria-label={title}>
          {[0.5, 1].map((f) => (
            <line
              key={f}
              x1={PAD}
              x2={CHART_W - PAD}
              y1={CHART_H - 22 - (CHART_H - 36) * f}
              y2={CHART_H - 22 - (CHART_H - 36) * f}
              className="stroke-line"
              strokeDasharray="3 4"
            />
          ))}
          {days.map((d, i) => {
            const v = values[i];
            const h = Math.round(((CHART_H - 36) * v) / max);
            const x = PAD + slot * i + (slot - barW) / 2;
            return (
              <g key={d.day}>
                {/* wide invisible hit target so small bars are easy to hover */}
                <rect x={PAD + slot * i} y={0} width={slot} height={CHART_H - 22} fill="transparent">
                  <title>{`${d.day}: ${tooltip(d)}`}</title>
                </rect>
                <rect
                  x={x}
                  y={CHART_H - 22 - h}
                  width={barW}
                  height={Math.max(h, v > 0 ? 2 : 0)}
                  rx={2}
                  className="pointer-events-none fill-accent"
                />
                {i % labelEvery === 0 && (
                  <text x={x + barW / 2} y={CHART_H - 6} textAnchor="middle" className="fill-faint" fontSize="10">
                    {d.day.slice(5)}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      )}
    </div>
  );
}

export default function AdminActivityPage() {
  const [days, setDays] = useState<number>(14);
  const [data, setData] = useState<AdminActivity | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setError(null);
    api
      .adminActivity(days)
      .then((result) => !cancelled && setData(result))
      .catch((err: Error) => !cancelled && setError(err.message));
    return () => {
      cancelled = true;
    };
  }, [days]);

  if (error) return <p className="text-sm text-red-500">Could not load activity: {error}</p>;
  if (!data)
    return (
      <div className="flex items-center gap-2 text-sm text-faint">
        <Loader2 size={14} className="animate-spin" /> Loading activity…
      </div>
    );

  const t = data.totals;
  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold">Activity</h2>
          <p className="text-sm text-muted">
            Who visits the demo and who signs up and uses it. Visitors are counted by a salted hash
            of their IP; no addresses or questions are stored.
          </p>
        </div>
        <div className="flex rounded-lg border border-line p-0.5 text-sm">
          {RANGES.map((r) => (
            <button
              key={r}
              onClick={() => setDays(r)}
              className={`rounded-md px-2.5 py-1 ${days === r ? "bg-surface font-medium text-ink" : "text-muted hover:text-ink"}`}
            >
              {r} days
            </button>
          ))}
        </div>
      </div>

      <section>
        <h3 className="mb-2 text-sm font-medium text-muted">Visitors and the demo</h3>
        <div className="flex flex-col items-stretch gap-2 sm:flex-row sm:items-center">
          <Step value={t.visitors} label="Visited" hint="Opened the site" />
          <ChevronRight size={16} className="hidden shrink-0 text-faint sm:block" />
          <Step value={t.guests_tried} label="Tried the demo" hint={`Asked as guests · ${t.guest_questions} questions`} />
          <ChevronRight size={16} className="hidden shrink-0 text-faint sm:block" />
          <Step value={t.signups} label="Signed up" hint={`${t.total_users} accounts in total`} />
          <ChevronRight size={16} className="hidden shrink-0 text-faint sm:block" />
          <Step value={t.active_users} label="Active users" hint={`${t.user_questions} questions · ${t.uploads} uploads`} />
        </div>
      </section>

      <div className="grid gap-4 lg:grid-cols-2">
        <DailyBars
          title="Visitors per day"
          days={data.daily}
          value={(d) => d.visitors}
          tooltip={(d) => `${d.visitors} visitors, ${d.guests_tried} tried the demo`}
        />
        <DailyBars
          title="Questions per day"
          days={data.daily}
          value={(d) => d.guest_questions + d.user_questions}
          tooltip={(d) => `${d.guest_questions} from guests, ${d.user_questions} from signed-in users`}
        />
      </div>

      <section className="rounded-2xl border border-line bg-panel p-4">
        <h3 className="text-sm font-medium">Signed-up users</h3>
        {data.users.length === 0 ? (
          <p className="py-6 text-center text-sm text-faint">No one has signed up yet.</p>
        ) : (
          <div className="mt-2 overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-line text-left text-xs text-faint">
                  <th className="py-1.5 pr-2 font-medium">Account</th>
                  <th className="py-1.5 pr-2 font-medium">Joined</th>
                  <th className="py-1.5 text-right font-medium">Questions ({days} d)</th>
                  <th className="py-1.5 text-right font-medium">Uploads</th>
                  <th className="py-1.5 pl-3 text-right font-medium">Last active</th>
                </tr>
              </thead>
              <tbody>
                {data.users.map((u) => (
                  <tr key={u.email} className="border-b border-line/60 last:border-0">
                    <td className="max-w-[260px] truncate py-1.5 pr-2">
                      <span className="font-medium">{u.display_name || u.email}</span>
                      {u.display_name && <span className="ml-1.5 text-faint">{u.email}</span>}
                    </td>
                    <td className="py-1.5 pr-2 text-muted">{fmtDate(u.joined)}</td>
                    <td className="py-1.5 text-right tabular-nums">{u.questions}</td>
                    <td className="py-1.5 text-right tabular-nums">{u.uploads}</td>
                    <td className="py-1.5 pl-3 text-right text-muted">{fmtWhen(u.last_active)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
