"use client";

import { useEffect, useRef, useState } from "react";

type GuestData = {
  storage: {
    profile_status: string;
    research_logging: boolean;
    scope: string;
    database_available: boolean;
  };
  preferences: {
    cross_session_profile: boolean;
    research_logging: boolean;
  };
  consent_version: string;
};

type Props = {
  open: boolean;
  onClose: () => void;
  apiUrl: string;
  onRevoked?: () => void;
};

function parseGuestData(value: unknown): GuestData {
  if (typeof value !== "object" || value === null) throw new Error("The server returned an invalid guest data status.");
  const data = value as Partial<GuestData>;
  if (
    typeof data.storage?.profile_status !== "string" ||
    typeof data.storage.research_logging !== "boolean" ||
    typeof data.storage.scope !== "string" ||
    typeof data.storage.database_available !== "boolean" ||
    typeof data.preferences?.cross_session_profile !== "boolean" ||
    typeof data.preferences.research_logging !== "boolean" ||
    typeof data.consent_version !== "string"
  ) throw new Error("The server returned an incomplete guest data status.");
  return data as GuestData;
}

export default function ProfileMemorySettings({ open, onClose, apiUrl, onRevoked }: Props) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const openRef = useRef(open);
  const requestRef = useRef(0);
  const controllerRef = useRef<AbortController | null>(null);
  const [data, setData] = useState<GuestData | null>(null);
  const [busy, setBusy] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [deleteConfirm, setDeleteConfirm] = useState(false);
  const [notice, setNotice] = useState<{ message: string; error: boolean } | null>(null);
  const endpoint = `${apiUrl.replace(/\/+$/, "")}/guest`;
  const loading = open && !loaded && !notice;

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    openRef.current = open;
    if (open && !dialog.open) dialog.showModal();
    if (!open && dialog.open) dialog.close();
  }, [open]);

  useEffect(() => {
    const request = ++requestRef.current;
    controllerRef.current?.abort();
    controllerRef.current = null;
    if (!open) return () => { requestRef.current += 1; };

    const controller = new AbortController();
    controllerRef.current = controller;
    void fetch(endpoint, { credentials: "include", cache: "no-store", signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(`Guest data settings are unavailable (HTTP ${response.status}).`);
        return parseGuestData(await response.json());
      })
      .then((result) => {
        if (requestRef.current === request && !controller.signal.aborted) {
          setData(result);
          setLoaded(true);
        }
      })
      .catch((cause: unknown) => {
        if (requestRef.current !== request || controller.signal.aborted) return;
        setLoaded(true);
        setNotice({ message: cause instanceof Error ? cause.message : "Guest data settings could not be loaded.", error: true });
      });

    return () => {
      controller.abort();
      if (controllerRef.current === controller) controllerRef.current = null;
      requestRef.current += 1;
    };
  }, [open, endpoint]);

  async function mutate(method: "PUT" | "DELETE", body?: GuestData["preferences"]) {
    if (busy || !data) return;
    const request = ++requestRef.current;
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setBusy(true);
    setNotice(null);
    try {
      const response = await fetch(endpoint, {
        method,
        credentials: "include",
        cache: "no-store",
        signal: controller.signal,
        ...(body ? { headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {}),
      });
      if (!response.ok) throw new Error(`The server could not update guest data (HTTP ${response.status}).`);
      const result = parseGuestData(await response.json());
      if (requestRef.current !== request || controller.signal.aborted) return;
      setData(result);
      setDeleteConfirm(false);
      setNotice({ message: method === "DELETE" ? "The server confirmed deletion of stored guest data and rotated this browser's guest identity." : "The server confirmed your guest data preferences.", error: false });
      if (method === "DELETE") onRevoked?.();
    } catch (cause) {
      if (requestRef.current !== request || controller.signal.aborted) return;
      setNotice({ message: cause instanceof Error ? cause.message : "The server could not update guest data.", error: true });
    } finally {
      if (requestRef.current === request && !controller.signal.aborted) setBusy(false);
    }
  }

  function handleDialogClose() {
    if (openRef.current) {
      setBusy(false);
      setLoaded(false);
      setData(null);
      setDeleteConfirm(false);
      setNotice(null);
      return;
    }
    requestRef.current += 1;
    controllerRef.current?.abort();
    controllerRef.current = null;
    setBusy(false);
    setLoaded(false);
    setData(null);
    setDeleteConfirm(false);
    setNotice(null);
    onClose();
  }

  return (
    <dialog
      ref={dialogRef}
      aria-labelledby="guest-data-title"
      aria-describedby="guest-data-description"
      aria-busy={busy || loading || undefined}
      onCancel={(event) => { event.preventDefault(); onClose(); }}
      onClose={handleDialogClose}
      className="marble-panel m-auto max-h-[92dvh] w-[calc(100%-1.5rem)] max-w-lg overflow-y-auto border border-ctp-surface1 p-0 text-ctp-text backdrop:bg-ctp-crust/85 backdrop:backdrop-blur-sm"
    >
      <div className="p-5 sm:p-6">
        <header className="border-b border-ctp-surface1 pb-4">
          <h2 id="guest-data-title" className="editorial-modal-title text-ctp-text">Guest data</h2>
          <p id="guest-data-description" className="mt-1 text-[13px] leading-5 text-ctp-subtext0">
            Bluff uses a guest identity stored in an HttpOnly browser cookie. No account or player name is required.
          </p>
        </header>

        <section className="mt-4 space-y-2 text-[13px] leading-6 text-ctp-subtext0" aria-label="Storage status">
          <p>When browser storage is available, local history keeps up to 200 public log entries per game for the 20 most recent games. It includes messages and revealed outcomes, never hidden cards or credentials.</p>
          <p>Active games and in-room Flagship memory live on the game server. Server profiles keep behavior counts separately from research logs.</p>
          {loading && <p role="status">Checking guest data status…</p>}
          {data && (
            <div className="border-l border-ctp-peach/50 pl-3" aria-live="polite">
              <p>Server profile status: {data.storage.profile_status}.</p>
              <p>Storage scope: {data.storage.scope}. Database available: {data.storage.database_available ? "yes" : "no"}.</p>
              <p>Server research logging setting: {data.storage.research_logging ? "on" : "off"}.</p>
            </div>
          )}
          {notice && <p className={notice.error ? "text-ctp-red" : "text-ctp-green"} role={notice.error ? "alert" : "status"}>{notice.message}</p>}
        </section>
        <p className="mt-2 text-[12px] leading-5 text-ctp-overlay1">An enabled setting is consent, not confirmation that a particular game log was saved. The game status reports recording, saved, or unavailable.</p>

        <div className="mt-4 space-y-3">
          <label className="flex gap-3 border border-ctp-surface1 bg-ctp-surface0/40 p-4">
            <input
              type="checkbox"
              checked={data?.preferences.cross_session_profile ?? false}
              disabled={!data || busy || loading}
              onChange={(event) => data && void mutate("PUT", { ...data.preferences, cross_session_profile: event.target.checked })}
              className="mt-1 h-4 w-4 shrink-0 accent-ctp-peach"
            />
            <span>
              <span className="block text-[13px] font-semibold">Remember my play style across guest games</span>
              <span className="mt-1 block text-[12px] leading-relaxed text-ctp-subtext0">Enabling applies to the next game. Turning it off deletes stored Flagship profiles. Only behavior counts from public decisions are used, not transcripts, and this does not change Random, Honest, or Math.</span>
            </span>
          </label>

          <label className="flex gap-3 border border-ctp-surface1 bg-ctp-surface0/40 p-4">
            <input
              type="checkbox"
              checked={data?.preferences.research_logging ?? false}
              disabled={!data || busy || loading}
              onChange={(event) => data && void mutate("PUT", { ...data.preferences, research_logging: event.target.checked })}
              className="mt-1 h-4 w-4 shrink-0 accent-ctp-peach"
            />
            <span>
              <span className="block text-[13px] font-semibold">Allow research logging</span>
              <span className="mt-1 block text-[12px] leading-relaxed text-ctp-subtext0">Enabling applies to the next game. Turning it off deletes stored research logs. The server records moves, dealt cards, revealed outcomes, and bot configuration for research and evaluation. These logs are not shown to your opponent or restored as card knowledge. This is separate from Flagship behavior counts.</span>
            </span>
          </label>
        </div>

        {deleteConfirm ? (
          <div className="mt-4 border border-ctp-red/40 bg-ctp-red/5 p-4">
            <p className="text-[13px] leading-5 text-ctp-subtext0">Delete stored guest profiles and research logs for this guest identity? The server rotates this browser&apos;s guest identity after deletion.</p>
            <div className="mt-3 flex flex-wrap justify-end gap-2">
              <button type="button" disabled={busy} onClick={() => setDeleteConfirm(false)} className="min-h-11 rounded-md border border-ctp-surface1 px-4 text-[13px] text-ctp-subtext0">Keep data</button>
              <button type="button" disabled={!data || busy} onClick={() => void mutate("DELETE")} className="min-h-11 rounded-md border border-ctp-red/50 px-4 text-[13px] font-semibold text-ctp-red disabled:opacity-50">{busy ? "Deleting…" : "Delete stored data"}</button>
            </div>
          </div>
        ) : (
          <div className="mt-4 flex justify-end">
            <button type="button" disabled={!data || busy || loading} onClick={() => setDeleteConfirm(true)} className="min-h-11 rounded-md border border-ctp-red/40 px-4 text-[13px] text-ctp-red disabled:opacity-50">Delete stored data</button>
          </div>
        )}

        <footer className="mt-5 flex justify-end border-t border-ctp-surface1 pt-4">
          <button type="button" onClick={onClose} autoFocus className="min-h-11 rounded-md border border-ctp-surface1 px-4 py-2 text-[13px] font-medium text-ctp-subtext0 hover:text-ctp-text">Close</button>
        </footer>
      </div>
    </dialog>
  );
}
