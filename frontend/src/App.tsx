import { useCallback, useEffect, useRef, useState } from "react";
import { Answer, Doc, ask, deleteDoc, listDocs, uploadDoc, watch } from "./api";

const STATUS_LABEL: Record<Doc["status"], string> = {
  pending: "Sırada",
  processing: "İşleniyor",
  done: "Hazır",
  failed: "Başarısız",
};

export default function App() {
  const [docs, setDocs] = useState<Doc[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const watchers = useRef(new Map<string, () => void>());

  const patch = useCallback((id: string, update: Partial<Doc>) => {
    setDocs((prev) => prev.map((d) => (d.id === id ? { ...d, ...update } : d)));
  }, []);

  const track = useCallback(
    (doc: Doc) => {
      if (doc.status === "done" || doc.status === "failed" || watchers.current.has(doc.id)) return;
      const stop = watch(doc.id, (u) => {
        patch(doc.id, u);
        if (u.status === "done" || u.status === "failed") watchers.current.delete(doc.id);
      });
      watchers.current.set(doc.id, stop);
    },
    [patch],
  );

  useEffect(() => {
    listDocs()
      .then((list) => {
        setDocs(list);
        list.forEach(track);
      })
      .catch((e: Error) => setError(e.message));
    const active = watchers.current;
    return () => {
      active.forEach((stop) => stop());
      active.clear();
    };
  }, [track]);

  async function onFile(file: File | undefined) {
    if (!file) return;
    setError(null);
    setUploading(true);
    try {
      const doc = await uploadDoc(file);
      setDocs((prev) => [doc, ...prev]);
      setSelected(doc.id);
      track(doc);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setUploading(false);
    }
  }

  async function remove(id: string) {
    try {
      await deleteDoc(id);
      watchers.current.get(id)?.();
      watchers.current.delete(id);
      setDocs((prev) => prev.filter((d) => d.id !== id));
      if (selected === id) setSelected(null);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  const current = docs.find((d) => d.id === selected) ?? null;

  return (
    <div className="app">
      <header>
        <h1>DocRAG</h1>
        <p>Belge yükle, içeriğiyle ilgili soru sor. İşleme arka planda yapılır, ilerleme canlı görünür.</p>
      </header>

      {error && (
        <div className="error" role="alert">
          {error}
        </div>
      )}

      <main>
        <section className="panel">
          <label className="upload">
            <input
              type="file"
              accept=".pdf,.txt,.md"
              disabled={uploading}
              onChange={(e) => {
                void onFile(e.target.files?.[0]);
                e.target.value = "";
              }}
            />
            {uploading ? "Yükleniyor…" : "PDF, TXT veya MD seç (en fazla 10 MB)"}
          </label>

          <ul className="docs">
            {docs.length === 0 && <li className="muted">Henüz belge yok.</li>}
            {docs.map((d) => (
              <li key={d.id} className={d.id === selected ? "active" : ""}>
                <button className="doc" onClick={() => setSelected(d.id)}>
                  <span className="name">{d.filename}</span>
                  <span className={`badge ${d.status}`}>{STATUS_LABEL[d.status]}</span>
                  {(d.status === "pending" || d.status === "processing") && (
                    <progress max={100} value={d.progress} aria-label="İşleme ilerlemesi" />
                  )}
                  {d.error && <span className="muted">{d.error}</span>}
                </button>
                <button className="link" onClick={() => void remove(d.id)} aria-label={`${d.filename} sil`}>
                  Sil
                </button>
              </li>
            ))}
          </ul>
        </section>

        <section className="panel">{current ? <Ask key={current.id} doc={current} /> : <p className="muted">Soru sormak için bir belge seç.</p>}</section>
      </main>
    </div>
  );
}

function Ask({ doc }: { doc: Doc }) {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      setAnswer(await ask(doc.id, question.trim()));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const ready = doc.status === "done";
  return (
    <div>
      <h2>{doc.filename}</h2>
      {!ready && <p className="muted">Belge hazır olunca soru sorabilirsin ({STATUS_LABEL[doc.status]}).</p>}
      <form onSubmit={submit}>
        <input
          value={question}
          maxLength={500}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Örn: İade süresi kaç gün?"
          disabled={!ready || busy}
        />
        <button type="submit" disabled={!ready || busy || question.trim().length < 3}>
          {busy ? "…" : "Sor"}
        </button>
      </form>
      {error && (
        <div className="error" role="alert">
          {error}
        </div>
      )}
      {answer && (
        <div className="answer">
          {/* Rendered as plain text: model output is never injected as HTML. */}
          <p>{answer.answer}</p>
          <small className="muted">Mod: {answer.mode}</small>
          <details>
            <summary>Kaynak bölümler ({answer.sources.length})</summary>
            {answer.sources.map((s) => (
              <blockquote key={s.chunk}>
                <small className="muted">
                  #{s.chunk} · benzerlik {s.score}
                </small>
                <p>{s.text}</p>
              </blockquote>
            ))}
          </details>
        </div>
      )}
    </div>
  );
}
