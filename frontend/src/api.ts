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

export type IngestResult = { ok: boolean; message: string };

export const ingestCollection = async (
  name: string,
  files: File[],
  hybrid: boolean,
): Promise<IngestResult> => {
  const formData = new FormData();
  formData.append("name", name);
  formData.append("hybrid", hybrid ? "true" : "false");

  files.forEach((file) => {
    const relPath =
      (file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name;
    formData.append("files", file, relPath);
  });

  const { data } = await API.post("/collections/ingest", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data as IngestResult;
};
