import { useEffect } from "react";
import { Route, Routes, useLocation } from "react-router-dom";
import { api } from "./api/client";
import { RequireAuth } from "./auth/AuthProvider";
import App from "./App";
import Landing from "./pages/Landing";
import Login from "./pages/Login";
import NotFound from "./pages/NotFound";
import Register from "./pages/Register";
import AdminLayout from "./pages/admin/AdminLayout";
import AdminActivityPage from "./pages/admin/Activity";
import AdminAuditPage from "./pages/admin/Audit";
import AdminFeedbackPage from "./pages/admin/Feedback";
import AdminOverviewPage from "./pages/admin/Overview";
import AdminUsagePage from "./pages/admin/Usage";
import AdminUsersPage from "./pages/admin/Users";

/**
 * Public: `/` landing, `/login`, `/register`. Everything else that touches
 * user data sits behind RequireAuth, which waits for the session bootstrap
 * and then applies the guard matrix (see auth/guards.ts). Admin pages add
 * the adminOnly flag: a real admin session is required even in open-local
 * mode. Chat and Documents also open for guests when the server runs the
 * public demo (read-only shared corpus, stateless chat).
 */
/** Reports each page view once per browser session (admin Activity page).
 *  Admin pages are not counted. Never blocks or breaks navigation. */
function VisitTracker() {
  const { pathname } = useLocation();
  useEffect(() => {
    if (pathname.startsWith("/admin")) return;
    const key = `oraculum.visit:${pathname}`;
    try {
      if (sessionStorage.getItem(key)) return;
      sessionStorage.setItem(key, "1");
    } catch {
      // storage unavailable (private mode, iframes): count anyway
    }
    void api.trackVisit(pathname);
  }, [pathname]);
  return null;
}

export default function AppRoutes() {
  return (
    <>
    <VisitTracker />
    <Routes>
      <Route path="/" element={<Landing />} />
      <Route path="/login" element={<Login />} />
      <Route path="/register" element={<Register />} />
      <Route
        path="/chat"
        element={
          <RequireAuth guestAllowed>
            <App />
          </RequireAuth>
        }
      />
      <Route
        path="/documents"
        element={
          <RequireAuth guestAllowed>
            <App />
          </RequireAuth>
        }
      />
      <Route
        path="/admin"
        element={
          <RequireAuth adminOnly>
            <AdminLayout />
          </RequireAuth>
        }
      >
        <Route index element={<AdminOverviewPage />} />
        <Route path="users" element={<AdminUsersPage />} />
        <Route path="activity" element={<AdminActivityPage />} />
        <Route path="usage" element={<AdminUsagePage />} />
        <Route path="feedback" element={<AdminFeedbackPage />} />
        <Route path="audit" element={<AdminAuditPage />} />
      </Route>
      <Route path="*" element={<NotFound />} />
    </Routes>
    </>
  );
}
