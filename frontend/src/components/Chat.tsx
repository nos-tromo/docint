import { useEffect, useState } from "react";
import {
  Box,
  Button,
  HStack,
  Input,
  Text,
  VStack,
  Spinner,
} from "@chakra-ui/react";
import { askQuery } from "../api";
import type { Source } from "../api";

type Msg = { role: "user" | "assistant"; text: string; sources?: Source[] };

type Props = { collection: string | null };

export default function Chat({ collection }: Props) {
  const [q, setQ] = useState("");
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

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
            <Text fontWeight="bold">
              {m.role === "assistant" ? "Assistant" : "You"}
            </Text>
            <Text whiteSpace="pre-wrap">{m.text}</Text>

            {m.sources && m.sources.length > 0 && (
              <>
                <Box my={2} borderTopWidth="1px" borderColor="border.muted" />
                <Text fontSize="sm" color="fg.muted">
                  Sources:
                </Text>
                {m.sources.map((s, j) => (
                  <Box key={j} fontSize="sm" color="fg.muted" mb={1}>
                    <Text>
                      • {s.filename || "unknown"}
                      {s.page ? ` (p.${s.page})` : s.row ? ` (row ${s.row})` : ""}
                    </Text>
                    {s.text && (
                      <Text ml={4} whiteSpace="pre-wrap" fontStyle="italic">
                        {s.text}
                      </Text>
                    )}
                  </Box>
                ))}
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
      </HStack>
    </VStack>
  );
}
