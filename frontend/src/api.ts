import axios from "axios";

export type Source = {
  filename?: string;
  page?: number;
  row?: number;
  text?: string;
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
