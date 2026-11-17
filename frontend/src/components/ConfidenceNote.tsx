import { useState } from "react";
import { ChevronDown, Gauge, ShieldAlert, ShieldCheck } from "lucide-react";
import type { ChatResponse } from "../api/types";
import { confidenceBand, describeSignal, formatPercent } from "./confidence";

/**
 * How sure the system is, under each answer.
 *
 * - answered + calibrated confidence → a small band chip ("Confidence 87%");
 * - withheld by the confidence gate → why, plus the held-back draft on
 *   request (the demo shows its work rather than hiding it);
 * - the signals the calibrator used, expandable.
 */
export default function ConfidenceNote({ response }: { response: ChatResponse }) {
  const [open, setOpen] = useState(false);
  const { confidence, abstain_reason: reason, withheld_answer: withheld, signals } = response;
  const hasSignals = signals && Object.keys(signals).length > 0;
  if (confidence == null && reason !== "low_confidence" && !hasSignals) return null;

  const band = confidenceBand(confidence ?? null);
  const withheldByGate = reason === "low_confidence";

  return (
    <div className="mt-3 text-[13px]">
      <div className="flex flex-wrap items-center gap-2">
        {withheldByGate ? (
          <span className="inline-flex items-center gap-1.5 rounded-full border border-amber-500/40 bg-amber-500/10 px-2.5 py-1 text-amber-700 dark:text-amber-300">
            <ShieldAlert size={13} />
            Answer withheld · confidence {formatPercent(confidence)}
          </span>
        ) : confidence != null ? (
          <span
            className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 ${band.className}`}
            title="Calibrated probability that this answer is correct"
          >
            <ShieldCheck size={13} />
            Confidence {formatPercent(confidence)} · {band.label}
          </span>
        ) : null}
        {(hasSignals || withheld) && (
          <button
            onClick={() => setOpen((v) => !v)}
            aria-expanded={open}
            className="inline-flex items-center gap-1 rounded-full px-2 py-1 text-faint transition-colors hover:text-ink"
          >
            <Gauge size={13} />
            {withheldByGate ? "What was held back?" : "How was this scored?"}
            <ChevronDown size={13} className={`transition-transform ${open ? "rotate-180" : ""}`} />
          </button>
        )}
      </div>

      {open && (
        <div className="mt-2 rounded-xl border border-line bg-surface px-3.5 py-3">
          {withheldByGate && withheld && (
            <div className="mb-3">
              <p className="text-faint">The model's draft, not shown as an answer:</p>
              <p className="mt-1 whitespace-pre-wrap italic text-muted">“{withheld}”</p>
            </div>
          )}
          {hasSignals && (
            <>
              <p className="mb-1.5 text-faint">
                Signals the calibrator combines into P(correct):
              </p>
              <dl className="grid grid-cols-[1fr_auto] gap-x-4 gap-y-1">
                {Object.entries(signals!)
                  .map(([name, value]) => describeSignal(name, value))
                  .filter((row): row is NonNullable<typeof row> => row !== null)
                  .map((row) => (
                    <div key={row.name} className="contents">
                      <dt className="text-muted" title={row.help}>
                        {row.label}
                      </dt>
                      <dd className="text-right tabular-nums text-ink">{row.value}</dd>
                    </div>
                  ))}
              </dl>
            </>
          )}
        </div>
      )}
    </div>
  );
}
