import { describe, expect, it } from "vitest";
import { confidenceBand, describeSignal, formatPercent } from "./confidence";

describe("confidence helpers", () => {
  it("formats probabilities as whole percents", () => {
    expect(formatPercent(0.874)).toBe("87%");
    expect(formatPercent(null)).toBe("–");
    expect(formatPercent(Number.NaN)).toBe("–");
  });

  it("bands confidence into high / moderate / low", () => {
    expect(confidenceBand(0.9).label).toBe("high");
    expect(confidenceBand(0.7).label).toBe("moderate");
    expect(confidenceBand(0.3).label).toBe("low");
    expect(confidenceBand(null).label).toBe("not calibrated");
  });

  it("describes calibrator inputs and hides bookkeeping counters", () => {
    expect(describeSignal("nli_entail", 0.912)?.value).toBe("91%");
    expect(describeSignal("rerank_top1", 6.4321)?.value).toBe("6.43");
    expect(describeSignal("n_evidence", 5)).toBeNull();
    expect(describeSignal("nli_entail", Number.NaN)).toBeNull();
  });
});
