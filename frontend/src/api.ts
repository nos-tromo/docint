import axios from "axios";

export type Source = {
  filename?: string;
  page?: number;
  row?: number;
  text?: string;
  preview_text?: string;
  preview_url?: string;
  document_url?: string;
  filetype?: string;
  file_hash?: string;
};

const API = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://localhost:8000",
});

export const listCollections = async (): Promise<string[]> => {
  const { data } = await API.get("/collections/list");
  return data;
};

export const selectCollection = async (name: string) => {
  const { data } = await API.post("/collections/select", { name });
  return data;
};

export const askQuery = async (
  question: string,
  sessionId?: string,
): Promise<{ answer: string; sources: Source[]; session_id: string }> => {
  const { data } = await API.post("/query", { question, session_id: sessionId });
  return data;
};

export type SummaryResponse = {
  summary: string;
  sources: Source[];
};

export const summarizeCollection = async (): Promise<SummaryResponse> => {
  const { data } = await API.post("/summarize");
  return data;
};

export type IngestionResponse = {
  ok: boolean;
  collection: string;
  data_dir: string;
  hybrid: boolean;
};

export type IngestionOptions = {
  tableRowLimit?: number | null;
  tableRowFilter?: string | null;
  hybrid?: boolean | null;
};

export const ingestCollection = async (
  collection: string,
  options?: IngestionOptions,
): Promise<IngestionResponse> => {
  const payload: Record<string, unknown> = { collection };
  if (options?.tableRowLimit !== undefined && options.tableRowLimit !== null) {
    payload.table_row_limit = options.tableRowLimit;
  }
  if (options?.tableRowFilter && options.tableRowFilter.trim().length > 0) {
    payload.table_row_filter = options.tableRowFilter.trim();
  }

  const { data } = await API.post("/ingest", payload);
  return data;
};

export type UploadEvent = {
  type:
    | "start"
    | "upload_progress"
    | "file_saved"
    | "ingestion_started"
    | "ingestion_complete"
    | "error";
  payload: Record<string, unknown>;
};

export const uploadAndIngest = async (
  collection: string,
  files: File[],
  options: IngestionOptions | undefined,
  onEvent?: (event: UploadEvent) => void,
) => {
  const formData = new FormData();
  formData.append("collection", collection);
  formData.append("hybrid", String(options?.hybrid ?? true));
  if (options?.tableRowLimit !== undefined && options.tableRowLimit !== null) {
    formData.append("table_row_limit", String(options.tableRowLimit));
  }
  if (options?.tableRowFilter) {
    formData.append("table_row_filter", options.tableRowFilter);
  }
  files.forEach((file) => formData.append("files", file));

  const endpoint = `${API.defaults.baseURL}/ingest/upload`;
  const response = await fetch(endpoint, { method: "POST", body: formData });
  if (!response.body) {
    throw new Error("Streaming response not supported by the browser");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const flushEvent = (raw: string) => {
    const lines = raw.split("\n").filter(Boolean);
    let eventType = "message";
    let dataPayload = "";
    lines.forEach((line) => {
      if (line.startsWith("event:")) {
        eventType = line.replace("event:", "").trim();
      } else if (line.startsWith("data:")) {
        dataPayload += line.replace("data:", "").trim();
      }
    });

    const parsed = dataPayload ? JSON.parse(dataPayload) : {};
    const allowed: UploadEvent["type"][] = [
      "start",
      "upload_progress",
      "file_saved",
      "ingestion_started",
      "ingestion_complete",
      "error",
    ];
    const typedEvent: UploadEvent["type"] = allowed.includes(
      eventType as UploadEvent["type"],
    )
      ? (eventType as UploadEvent["type"])
      : "error";
    onEvent?.({ type: typedEvent, payload: parsed });
    if (typedEvent === "error") {
      throw new Error((parsed.message as string) || "Upload failed");
    }
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";
    for (const part of parts) {
      if (part.trim().length === 0) continue;
      flushEvent(part);
    }
  }

  if (buffer.trim()) {
    flushEvent(buffer.trim());
  }
};
