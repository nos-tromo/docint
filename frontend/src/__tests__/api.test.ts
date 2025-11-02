import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";

const getMock = vi.fn();
const postMock = vi.fn();

vi.mock("axios", () => {
  return {
    default: {
      create: vi.fn(() => ({ get: getMock, post: postMock })),
    },
  };
});

const api = await import("../api");

const { listCollections, selectCollection, askQuery, ingestCollection } = api;

describe("api", () => {
  beforeAll(() => {
    getMock.mockReset();
    postMock.mockReset();
  });

  afterEach(() => {
    getMock.mockReset();
    postMock.mockReset();
  });

  it("lists collections", async () => {
    getMock.mockResolvedValue({ data: ["a"] });
    const names = await listCollections();
    expect(getMock).toHaveBeenCalledWith("/collections/list");
    expect(names).toEqual(["a"]);
  });

  it("selects collection", async () => {
    postMock.mockResolvedValue({ data: { ok: true } });
    const result = await selectCollection("demo");
    expect(postMock).toHaveBeenCalledWith("/collections/select", { name: "demo" });
    expect(result).toEqual({ ok: true });
  });

  it("asks query", async () => {
    postMock.mockResolvedValue({
      data: { answer: "42", sources: [], session_id: "id" },
    });
    const result = await askQuery("Why?", "session");
    expect(postMock).toHaveBeenCalledWith("/query", {
      question: "Why?",
      session_id: "session",
    });
    expect(result).toEqual({ answer: "42", sources: [], session_id: "id" });
  });

  it("ingests collection", async () => {
    postMock.mockResolvedValue({ data: { ok: true, collection: "demo", data_dir: "/tmp", hybrid: true } });
    const result = await ingestCollection("demo");
    expect(postMock).toHaveBeenCalledWith("/ingest", { collection: "demo" });
    expect(result).toEqual({ ok: true, collection: "demo", data_dir: "/tmp", hybrid: true });
  });
});
