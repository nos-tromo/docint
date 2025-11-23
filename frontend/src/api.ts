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

export const ingestCollection = async (
  collection: string,
): Promise<IngestionResponse> => {
  const { data } = await API.post("/ingest", { collection });
  return data;
};
