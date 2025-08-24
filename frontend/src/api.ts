import axios from "axios";

export type Source = {
  filename?: string;
  page?: number;
  row?: number;
  text?: string;
};

const API = axios.create({ baseURL: "http://localhost:8001" });

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
): Promise<{ answer: string; sources: Source[] }> => {
  const { data } = await API.post("/query", { question });
  return data;
};
