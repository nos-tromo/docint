import { type ChangeEvent, useRef, useState } from "react";
import {
  Box,
  Button,
  FormControl,
  FormLabel,
  HStack,
  Input,
  List,
  ListItem,
  Stack,
  Switch,
  Text,
  VStack,
} from "@chakra-ui/react";
import { ingestCollection } from "../api";

type Props = {
  onIngested?: (collection: string) => Promise<void> | void;
};

type Status = { type: "success" | "error"; text: string } | null;

export default function Ingest({ onIngested }: Props) {
  const [collection, setCollection] = useState("");
  const [hybrid, setHybrid] = useState(true);
  const [files, setFiles] = useState<File[]>([]);
  const [status, setStatus] = useState<Status>(null);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleFilesChange = (event: ChangeEvent<HTMLInputElement>) => {
    const list = event.target.files;
    setFiles(list ? Array.from(list) : []);
  };

  const resetFileInput = () => {
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const clearFiles = () => {
    setFiles([]);
    resetFileInput();
  };

  const submit = async () => {
    const trimmed = collection.trim();
    if (!trimmed) {
      setStatus({ type: "error", text: "Collection name is required." });
      return;
    }
    if (files.length === 0) {
      setStatus({ type: "error", text: "Select at least one document to ingest." });
      return;
    }

    setStatus(null);
    setLoading(true);
    try {
      const response = await ingestCollection(trimmed, files, hybrid);
      setStatus({ type: "success", text: response.message || "Ingestion complete." });
      setCollection(trimmed);
      clearFiles();
      if (onIngested) {
        await onIngested(trimmed);
      }
    } catch (error: unknown) {
      const err = error as {
        response?: { data?: { detail?: string } };
        message?: string;
      };
      const detail = err.response?.data?.detail || err.message || "Ingestion failed.";
      setStatus({ type: "error", text: detail });
    } finally {
      setLoading(false);
    }
  };

  const triggerFilePicker = () => {
    fileInputRef.current?.click();
  };

  return (
    <VStack align="stretch" gap={4}>
      <Box
        borderWidth="1px"
        borderRadius="md"
        borderColor="border.muted"
        bg="bg.panel"
        p={5}
      >
        <Stack gap={4}>
          <Text color="fg.muted" fontSize="sm">
            Upload the documents you would like to add to a collection. Selecting a
            folder (supported in Chromium-based browsers) preserves the directory
            structure used during ingestion.
          </Text>

          <FormControl>
            <FormLabel>Collection name</FormLabel>
            <Input
              value={collection}
              onChange={(event) => setCollection(event.target.value)}
              placeholder="Enter a collection name"
              variant="outline"
              borderColor="border.muted"
              bg="bg.canvas"
            />
          </FormControl>

          <FormControl display="flex" alignItems="center" justifyContent="space-between">
            <FormLabel mb="0">Hybrid search</FormLabel>
            <Switch
              colorScheme="teal"
              isChecked={hybrid}
              onChange={(event) => setHybrid(event.target.checked)}
            />
          </FormControl>

          <Box>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              style={{ display: "none" }}
              onChange={handleFilesChange}
              // @ts-expect-error - directory upload is supported in Chromium
              webkitdirectory=""
              // @ts-expect-error - directory upload support for Firefox variants
              directory=""
            />
            <HStack gap={3}>
              <Button onClick={triggerFilePicker} variant="outline">
                Choose files or folder
              </Button>
              <Button onClick={clearFiles} variant="ghost" disabled={files.length === 0}>
                Clear selection
              </Button>
            </HStack>
            <Text mt={2} fontSize="sm" color="fg.muted">
              {files.length === 0
                ? "No files selected yet."
                : `${files.length} file${files.length === 1 ? "" : "s"} ready for ingestion.`}
            </Text>
            {files.length > 0 && (
              <List mt={3} spacing={1} maxH="180px" overflowY="auto">
                {files.slice(0, 10).map((file) => {
                  const relPath =
                    (file as File & { webkitRelativePath?: string }).webkitRelativePath ||
                    file.name;
                  return (
                    <ListItem key={`${file.name}-${relPath}`} fontSize="sm">
                      {relPath}
                    </ListItem>
                  );
                })}
                {files.length > 10 && (
                  <ListItem fontSize="sm" color="fg.muted">
                    …and {files.length - 10} more
                  </ListItem>
                )}
              </List>
            )}
          </Box>

          {status && (
            <Box
              borderRadius="md"
              p={3}
              bg={status.type === "success" ? "green.900" : "red.900"}
              color="white"
            >
              {status.text}
            </Box>
          )}

          <Button
            onClick={submit}
            colorScheme="teal"
            alignSelf="flex-start"
            isLoading={loading}
          >
            Start ingestion
          </Button>
        </Stack>
      </Box>
    </VStack>
  );
}
