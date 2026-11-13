export const API = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export type Doc = {
  id: string;
  filename: string;
  status: "pending" | "processing" | "done" | "failed";
  progress: number;
  error: string | null;
};

export type Source = { chunk: number; score: number; text: string };
export type Answer = { answer: string; mode: string; sources: Source[] };

async function check(res: Response) {
  if (res.ok) return res;
  let detail = `Hata ${res.status}`;
  try {
    const body = await res.json();
    if (typeof body.detail === "string") detail = body.detail;
  } catch {
    /* keep default */
  }
  throw new Error(detail);
}

export async function listDocs(): Promise<Doc[]> {
  return (await check(await fetch(`${API}/documents`))).json();
}

export async function uploadDoc(file: File): Promise<Doc> {
  const form = new FormData();
  form.append("file", file);
  return (await check(await fetch(`${API}/documents`, { method: "POST", body: form }))).json();
}

export async function deleteDoc(id: string): Promise<void> {
  await check(await fetch(`${API}/documents/${id}`, { method: "DELETE" }));
}

export async function ask(id: string, question: string): Promise<Answer> {
  const res = await fetch(`${API}/documents/${id}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  return (await check(res)).json();
}

/** Subscribes to server-sent progress events; returns an unsubscribe function. */
export function watch(id: string, onUpdate: (u: Pick<Doc, "status" | "progress" | "error">) => void): () => void {
  const source = new EventSource(`${API}/documents/${id}/events`);
  source.onmessage = (e) => {
    const update = JSON.parse(e.data);
    onUpdate(update);
    if (update.status === "done" || update.status === "failed") source.close();
  };
  source.onerror = () => source.close();
  return () => source.close();
}
