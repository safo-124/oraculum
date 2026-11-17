import { Link } from "react-router-dom";
import { Cpu, Gauge, MessageSquare } from "lucide-react";
import { useAuth } from "../auth/AuthProvider";

const FEATURES = [
  {
    icon: MessageSquare,
    title: "Grounded answers",
    body: "Every answer cites the exact passages it came from, and a language-inference model checks that they actually support it.",
  },
  {
    icon: Gauge,
    title: "Knows when it doesn't know",
    body: "Retrieval, token-probability and entailment signals are combined into a calibrated confidence. Low-confidence answers are withheld.",
  },
  {
    icon: Cpu,
    title: "Open models only",
    body: "Qwen2.5 via llama.cpp, bge embeddings and cross-encoders from Hugging Face. Runs on one CPU server with no external AI API.",
  },
];

export default function Landing() {
  const { user, authRequired, registrationEnabled, guestMode } = useAuth();

  // Anonymous + locked API → sign in first, unless the public demo is on.
  const primary =
    !user && authRequired && !guestMode
      ? { label: "Sign in", to: "/login" }
      : { label: user ? "Open app" : "Try the demo", to: "/chat" };

  return (
    <div className="min-h-screen bg-canvas text-ink">
      <header className="mx-auto flex h-16 max-w-5xl items-center justify-between px-4">
        <div className="flex items-center gap-2.5">
          <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent text-on-accent">
            <MessageSquare size={16} strokeWidth={2.25} />
          </span>
          <span className="text-sm font-semibold tracking-tight">Oraculum</span>
        </div>
        <nav className="flex items-center gap-2">
          {user ? (
            <Link
              to="/chat"
              className="rounded-lg bg-accent px-3.5 py-1.5 text-sm font-medium text-on-accent hover:bg-accent-hover"
            >
              Open app
            </Link>
          ) : (
            <>
              <Link
                to="/login"
                className="rounded-lg px-3 py-1.5 text-sm text-muted hover:bg-surface hover:text-ink"
              >
                Sign in
              </Link>
              <Link
                to={registrationEnabled ? "/register" : "/login"}
                className="rounded-lg bg-accent px-3.5 py-1.5 text-sm font-medium text-on-accent hover:bg-accent-hover"
              >
                Get started
              </Link>
            </>
          )}
        </nav>
      </header>

      <main className="mx-auto max-w-5xl px-4">
        <section className="py-16 text-center sm:py-24">
          <h1 className="mx-auto max-w-2xl text-4xl font-bold leading-tight tracking-tight sm:text-5xl">
            Ask your documents.
            <br />
            Get answers it can stand behind.
          </h1>
          <p className="mx-auto mt-5 max-w-xl text-base leading-6 text-muted">
            Oraculum answers questions from your documents with open models,
            cites its evidence, and says how likely the answer is to be right.
            When that is too unlikely, it says so instead of guessing.
          </p>
          <div className="mt-8 flex items-center justify-center gap-3">
            <Link
              to={primary.to}
              className="rounded-xl bg-accent px-6 py-2.5 text-sm font-medium text-on-accent shadow-sm hover:bg-accent-hover"
            >
              {primary.label}
            </Link>
            {!user && registrationEnabled && (
              <Link
                to="/register"
                className="rounded-xl border border-line bg-surface px-6 py-2.5 text-sm font-medium text-ink hover:bg-elevated"
              >
                Create an account
              </Link>
            )}
          </div>
          {!user && (
            <p className="mt-3 text-xs text-faint">
              {guestMode
                ? "No account needed to try it: ask about the demo documents. Sign in to upload your own."
                : "Sign in or create an account to start asking questions."}
            </p>
          )}
        </section>

        <section className="grid gap-4 pb-16 sm:grid-cols-3">
          {FEATURES.map(({ icon: Icon, title, body }) => (
            <div
              key={title}
              className="rounded-2xl border border-line bg-panel p-5"
            >
              <span className="mb-3 flex h-9 w-9 items-center justify-center rounded-lg bg-surface text-accent ring-1 ring-line">
                <Icon size={17} strokeWidth={2} />
              </span>
              <h2 className="text-sm font-semibold">{title}</h2>
              <p className="mt-1.5 text-sm leading-5 text-muted">{body}</p>
            </div>
          ))}
        </section>
      </main>

      <footer className="border-t border-line py-6 text-center text-xs text-faint">
        Oraculum: retrieval-augmented answers with calibrated abstention
      </footer>
    </div>
  );
}
