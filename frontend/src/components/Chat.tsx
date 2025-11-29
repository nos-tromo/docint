import { useEffect, useRef, useState } from "react";
import {
  Box,
  Button,
  HStack,
  IconButton,
  Image,
  Input,
  Spinner,
  Text,
  VStack,
} from "@chakra-ui/react";
import { askQuery } from "../api";
import type { Source } from "../api";

type Msg = { role: "user" | "assistant"; text: string; sources?: Source[] };

type Props = { collection: string | null };

const SourceCard = ({ source }: { source: Source }) => {
  const [isVisible, setIsVisible] = useState(false);
  const ref = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          observer.disconnect();
        }
      });
    });
    if (ref.current) {
      observer.observe(ref.current);
    }
    return () => observer.disconnect();
  }, []);

  const location = source.page
    ? `Page ${source.page}`
    : source.row
      ? `Row ${source.row}`
      : null;

  const previewText = source.preview_text || source.text || "";
  const previewUrl = source.preview_url || source.document_url;
  const showImage =
    isVisible &&
    previewUrl &&
    typeof source.filetype === "string" &&
    source.filetype.toLowerCase().startsWith("image");

  return (
    <Box
      ref={ref}
      borderWidth="1px"
      borderColor="border.muted"
      borderRadius="md"
      p={3}
      bg="bg.panel"
    >
      <HStack justify="space-between" align="flex-start">
        <Text fontWeight="semibold">{source.filename || "Unknown source"}</Text>
        {location && (
          <Text fontSize="xs" color="fg.muted">
            {location}
          </Text>
        )}
      </HStack>

      {previewText && (
        <Text mt={2} fontSize="sm" color="fg.muted" lineClamp={4} whiteSpace="pre-wrap">
          {previewText}
        </Text>
      )}

      {showImage && previewUrl && (
        <Box mt={2} overflow="hidden" borderRadius="sm">
          <Image
            src={previewUrl}
            alt={`Preview of ${source.filename || "source"}`}
            loading="lazy"
            width="100%"
            maxHeight={200}
            objectFit="cover"
          />
        </Box>
      )}

      {previewUrl && (
        <Button
          size="sm"
          variant="outline"
          colorScheme="teal"
          mt={3}
          onClick={() => window.open(previewUrl, "_blank", "noopener,noreferrer")}
        >
          Open document
        </Button>
      )}
    </Box>
  );
};

export default function Chat({ collection }: Props) {
  const [q, setQ] = useState("");
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [openRefs, setOpenRefs] = useState<Record<number, boolean>>({});
  const [copiedIdx, setCopiedIdx] = useState<number | null>(null);
  const copyTimeout = useRef<number | null>(null);

  const toggleRefs = (idx: number) =>
    setOpenRefs((prev) => ({ ...prev, [idx]: !prev[idx] }));

  const formatMessageForCopy = (msg: Msg) => {
    let payload = msg.text;
    if (msg.sources && msg.sources.length > 0) {
      const srcs = msg.sources
        .map((s, index) => {
          const location = s.page
            ? ` (page ${s.page})`
            : s.row
            ? ` (row ${s.row})`
            : "";
          const text = s.text ? `\n    "${s.text}"` : "";
          return `${index + 1}. ${s.filename || "unknown"}${location}${text}`;
        })
        .join("\n");
      payload += `\n\nSources:\n${srcs}`;
    }
    return payload;
  };

  const handleCopy = (idx: number, msg: Msg) => {
    const payload = formatMessageForCopy(msg);
    void navigator.clipboard.writeText(payload).catch(() => undefined);
    setCopiedIdx(idx);
    if (copyTimeout.current) {
      window.clearTimeout(copyTimeout.current);
    }
    copyTimeout.current = window.setTimeout(() => {
      setCopiedIdx((current) => (current === idx ? null : current));
      copyTimeout.current = null;
    }, 2000);
  };

  const msgsKey = collection ? `chat_msgs_${collection}` : "chat_msgs";
  const sessionKey = collection ? `sessionId_${collection}` : "sessionId";

  useEffect(() => {
    try {
      const stored = localStorage.getItem(msgsKey);
      setMsgs(stored ? (JSON.parse(stored) as Msg[]) : []);
    } catch {
      setMsgs([]);
    }
    try {
      setSessionId(localStorage.getItem(sessionKey));
    } catch {
      setSessionId(null);
    }
  }, [msgsKey, sessionKey]);

  useEffect(() => {
    localStorage.setItem(msgsKey, JSON.stringify(msgs));
  }, [msgs, msgsKey]);

  useEffect(() => () => {
    if (copyTimeout.current) {
      window.clearTimeout(copyTimeout.current);
    }
  }, []);

  const ask = async () => {
    const question = q.trim();
    if (!question || loading) return;
    setMsgs((m) => [...m, { role: "user", text: question }]);
    setQ("");
    setLoading(true);
    try {
      const { answer, sources, session_id } = await askQuery(
        question,
        sessionId || undefined,
      );
      setSessionId(session_id);
      localStorage.setItem(sessionKey, session_id);
      setMsgs((m) => [...m, { role: "assistant", text: answer || "", sources }]);
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } } };
      const msg = err.response?.data?.detail
        ? `⚠️ ${err.response.data.detail}`
        : `⚠️ ${String(e)}`;
      setMsgs((m) => [...m, { role: "assistant", text: msg }]);
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setMsgs([]);
    setSessionId(null);
    localStorage.removeItem(msgsKey);
    localStorage.removeItem(sessionKey);
  };

  const exportChat = () => {
    if (msgs.length === 0) return;

    const lines = msgs.map((m) => {
      let base = `${m.role === "assistant" ? "Assistant" : "You"}: ${m.text}`;
      if (m.sources && m.sources.length > 0) {
        const srcs = m.sources
          .map(
            (s) =>
              `  • ${s.filename || "unknown"}${
                s.page ? ` (p.${s.page})` : s.row ? ` (row ${s.row})` : ""
              }${s.text ? `\n    "${s.text}"` : ""}`,
          )
          .join("\n");
        base += `\nSources:\n${srcs}`;
      }
      return base;
    });

    const content = lines.join("\n\n");
    const blob = new Blob([content], { type: "text/plain" });
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    const filename = collection
      ? `chat_${collection}.txt`
      : "chat_export.txt";
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <VStack align="stretch" gap={3}>
      <VStack
        align="stretch"
        gap={4}
        p={4}
        borderWidth="1px"
        borderColor="border.muted"
        borderRadius="md"
        bg="bg.panel"
      >
        {msgs.map((m, i) => (
          <Box
            key={i}
            bg={m.role === "assistant" ? "bg.muted" : "transparent"}
            p={3}
            borderRadius="md"
          >
            <HStack justify="space-between" align="center" mb={1}>
              <Text fontWeight="bold">
                {m.role === "assistant" ? "Assistant" : "You"}
              </Text>
              {m.role === "assistant" && (
                <IconButton
                  aria-label={copiedIdx === i ? "Copied" : "Copy answer"}
                  size="sm"
                  variant="ghost"
                  onClick={() => handleCopy(i, m)}
                  title={copiedIdx === i ? "Copied!" : "Copy answer"}
                >
                  <Box as="span" fontSize="md" lineHeight={1}>
                    {copiedIdx === i ? "✅" : "📋"}
                  </Box>
                </IconButton>
              )}
            </HStack>
            <Text whiteSpace="pre-wrap">{m.text}</Text>

            {m.sources && m.sources.length > 0 && (
              <>
                <Box my={2} borderTopWidth="1px" borderColor="border.muted" />
                <HStack
                  gap={2}
                  cursor="pointer"
                  onClick={() => toggleRefs(i)}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) =>
                    (e.key === "Enter" || e.key === " ") && toggleRefs(i)
                  }
                >
                  <Text as="span" color="fg.muted" fontWeight="bold">
                    {openRefs[i] ? "▾" : "▸"}
                  </Text>
                  <Text fontSize="sm" color="fg.muted" fontWeight="medium">
                    Sources ({m.sources.length})
                  </Text>
                </HStack>
                {!!openRefs[i] && (
                  <VStack mt={3} align="stretch" gap={3}>
                    {m.sources.map((s, j) => (
                      <SourceCard key={`${s.file_hash || j}-${j}`} source={s} />
                    ))}
                  </VStack>
                )}
              </>
            )}
          </Box>
        ))}

        {loading && (
          <HStack>
            <Spinner size="sm" color="fg.muted" />
            <Text fontSize="sm" color="fg.muted">
              Searching…
            </Text>
          </HStack>
        )}
      </VStack>

      <HStack>
        <Input
          placeholder="Ask something…"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && ask()}
          disabled={loading}
          variant="outline"
          borderColor="border.muted"
          bg="bg.panel"
        />
        <Button onClick={ask} colorScheme="teal" disabled={loading || !q.trim()}>
          Send
        </Button>
        <Button onClick={reset} variant="outline" disabled={loading}>
          New Session
        </Button>
        <Button onClick={exportChat} variant="outline" disabled={msgs.length === 0}>
          Export Chat
        </Button>
      </HStack>
    </VStack>
  );
}
