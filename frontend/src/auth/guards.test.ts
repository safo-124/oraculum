import { describe, expect, it } from "vitest";
import type { AuthUser } from "../api/types";
import { evaluateGuard } from "./guards";

const user: AuthUser = {
  id: "u-1",
  email: "user@example.com",
  display_name: null,
  role: "user",
  created_at: "2026-01-01T00:00:00Z",
  last_login_at: null,
};

const admin: AuthUser = { ...user, id: "u-2", email: "admin@example.com", role: "admin" };

function decide(overrides: Partial<Parameters<typeof evaluateGuard>[0]>) {
  return evaluateGuard({
    loading: false,
    user: null,
    authRequired: false,
    ...overrides,
  });
}

describe("evaluateGuard", () => {
  it("waits during the session bootstrap (no premature redirect)", () => {
    expect(decide({ loading: true })).toBe("wait");
    expect(decide({ loading: true, user, authRequired: true })).toBe("wait");
  });

  it("allows anonymous access in open-local mode (no API_KEY)", () => {
    expect(decide({})).toBe("allow");
  });

  it("forces /login for anonymous access when auth is required", () => {
    expect(decide({ authRequired: true })).toBe("login");
  });

  it("allows a signed-in user anywhere except admin routes", () => {
    expect(decide({ user })).toBe("allow");
  });

  it("always hides /admin from anonymous access, even open-local", () => {
    expect(decide({ adminOnly: true })).toBe("login");
  });

  it("sends non-admin users away from /admin back to the app", () => {
    expect(decide({ user, adminOnly: true })).toBe("chat");
  });

  it("lets admins into /admin", () => {
    expect(decide({ user: admin, adminOnly: true })).toBe("allow");
    expect(decide({ user: admin, adminOnly: true, authRequired: true })).toBe(
      "allow",
    );
  });

  it("does not redirect a signed-in user to login on normal routes", () => {
    // Regression: with authRequired true and a valid session, /chat renders.
    expect(decide({ user, authRequired: true })).toBe("allow");
  });

  it("lets guests into guest routes when the public demo is on", () => {
    expect(decide({ authRequired: true, guestMode: true, guestAllowed: true })).toBe("allow");
  });

  it("still sends guests to /login on routes that need an account", () => {
    expect(decide({ authRequired: true, guestMode: true })).toBe("login");
    expect(decide({ authRequired: true, guestMode: true, guestAllowed: true, adminOnly: true })).toBe("login");
  });

  it("ignores guestAllowed when the demo is off", () => {
    expect(decide({ authRequired: true, guestAllowed: true })).toBe("login");
  });
});
