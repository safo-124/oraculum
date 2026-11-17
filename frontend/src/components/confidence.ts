// Pure helpers for ConfidenceNote, kept DOM-free so vitest can cover them.

export interface Band {
  label: string;
  className: string;
}

export function formatPercent(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "–";
  return `${Math.round(value * 100)}%`;
}

export function confidenceBand(value: number | null): Band {
  if (value == null) {
    return { label: "not calibrated", className: "border-line text-faint" };
  }
  if (value >= 0.85) {
    return {
      label: "high",
      className: "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300",
    };
  }
  if (value >= 0.65) {
    return {
      label: "moderate",
      className: "border-sky-500/40 bg-sky-500/10 text-sky-700 dark:text-sky-300",
    };
  }
  return {
    label: "low",
    className: "border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-300",
  };
}

interface SignalSpec {
  label: string;
  help: string;
  format: (value: number) => string;
}

const percent = (v: number) => formatPercent(v);
const fixed = (v: number) => v.toFixed(2);

// Only the calibrator's inputs are shown; bookkeeping counters are hidden.
const SIGNALS: Record<string, SignalSpec> = {
  rerank_top1: {
    label: "Best passage relevance",
    help: "Cross-encoder score of the most relevant retrieved passage (logit; higher is more relevant).",
    format: fixed,
  },
  rerank_margin: {
    label: "Lead over 2nd passage",
    help: "How far the best passage scores above the runner-up.",
    format: fixed,
  },
  status_p_answered: {
    label: "Model's P(answerable)",
    help: "The language model's own probability of answering rather than declining.",
    format: percent,
  },
  answer_mean_logprob: {
    label: "Mean token log-prob",
    help: "Average log-probability of the answer tokens (closer to 0 is more certain).",
    format: fixed,
  },
  answer_min_logprob: {
    label: "Least certain token",
    help: "Lowest log-probability among the answer tokens.",
    format: fixed,
  },
  nli_entail: {
    label: "Evidence entails answer",
    help: "NLI model: probability that a cited passage entails the answer.",
    format: percent,
  },
  nli_entail_qa: {
    label: "Evidence entails Q + A",
    help: "NLI model: probability that a cited passage entails the question joined with the answer.",
    format: percent,
  },
  nli_contradiction: {
    label: "Evidence contradicts",
    help: "NLI model: probability that a cited passage contradicts the answer.",
    format: percent,
  },
  cited_fraction: {
    label: "Passages cited",
    help: "Share of the retrieved passages the answer cites.",
    format: percent,
  },
};

export function describeSignal(
  name: string,
  value: number,
): { name: string; label: string; help: string; value: string } | null {
  const spec = SIGNALS[name];
  if (!spec || !Number.isFinite(value)) return null;
  return { name, label: spec.label, help: spec.help, value: spec.format(value) };
}
