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

export default function Chat() {
  const [q, setQ] = useState("");
  const [msgs, setMsgs] = useState<Msg[]>(() => {
    try {
      const stored = localStorage.getItem("chat_msgs");
      return stored ? (JSON.parse(stored) as Msg[]) : [];
    } catch {
      return [];
    }
  });
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    localStorage.setItem("chat_msgs", JSON.stringify(msgs));
  }, [msgs]);

  const ask = async () => {
    const question = q.trim();
    if (!question || loading) return;
    setMsgs((m) => [...m, { role: "user", text: question }]);
    setQ("");
    setLoading(true);
    try {
      const { answer, sources } = await askQuery(question);
      setMsgs((m) => [...m, { role: "assistant", text: answer || "", sources }]);
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } } };
      const msg = err.response?.data?.detail ? `⚠️ ${err.response.data.detail}` : `⚠️ ${String(e)}`;
      setMsgs((m) => [...m, { role: "assistant", text: msg }]);
    } finally {
      setLoading(false);
    }
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
                  <Text key={j} fontSize="sm" color="fg.muted">
                    • {s.filename || "unknown"}
                    {s.page ? ` (p.${s.page})` : s.row ? ` (row ${s.row})` : ""}
                  </Text>
                ))}
              </>
            )}
          </Box>
        ))}

        {loading && (
          <HStack>
            <Spinner size="sm" color="fg.muted" />
            <Text fontSize="sm" color="fg.muted">
              Thinking…
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
      </HStack>
    </VStack>
  );
}