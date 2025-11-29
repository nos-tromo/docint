import { useEffect, useState } from "react";
import type { AxiosError } from "axios";
import {
  Box,
  Button,
  Flex,
  HStack,
  Heading,
  Input,
  Stack,
  Text,
} from "@chakra-ui/react";
import { ingestCollection, uploadAndIngest, type UploadEvent } from "../api";

type Props = {
  currentCollection: string | null;
  onCollectionAttached: (collection: string) => void;
};

export default function IngestionPanel({
  currentCollection,
  onCollectionAttached,
}: Props) {
  const [collection, setCollection] = useState<string>(
    currentCollection ?? "",
  );
  const [status, setStatus] = useState<
    | { type: "success"; message: string }
    | { type: "error"; message: string }
    | null
  >(null);
  const [isLoading, setIsLoading] = useState(false);
  const [tableRowLimit, setTableRowLimit] = useState<string>("");
  const [tableRowFilter, setTableRowFilter] = useState<string>("");
  const [files, setFiles] = useState<File[]>([]);
  const [progress, setProgress] = useState<UploadEvent[]>([]);
  const [uploadProgress, setUploadProgress] = useState<Record<string, number>>({});

  useEffect(() => {
    setCollection(currentCollection ?? "");
  }, [currentCollection]);

  const submit = async () => {
    const name = collection.trim();
    if (!name) {
      setStatus({ type: "error", message: "Collection name is required." });
      return;
    }

    try {
      setIsLoading(true);
      setStatus(null);
      setProgress([]);
      setUploadProgress({});
      let rowLimit: number | null = null;
      if (tableRowLimit.trim()) {
        const parsed = Number(tableRowLimit.trim());
        if (Number.isNaN(parsed)) {
          setStatus({
            type: "error",
            message: "Table row limit must be a valid number.",
          });
          return;
        }
        if (parsed <= 0) {
          setStatus({
            type: "error",
            message: "Table row limit must be a positive number.",
          });
          return;
        }
        rowLimit = parsed;
      }

      if (files.length > 0) {
        await uploadAndIngest(
          name,
          files,
          {
            tableRowLimit: rowLimit,
            tableRowFilter: tableRowFilter.trim() || null,
          },
          (event) => {
            setProgress((current) => [...current, event]);
            if (event.type === "upload_progress") {
              const filename = String(event.payload.filename ?? "");
              const bytes = Number(event.payload.bytes_written ?? 0);
              setUploadProgress((current) => ({
                ...current,
                [filename]: bytes,
              }));
            }
          },
        );
        setStatus({
          type: "success",
          message: `Ingestion complete for "${name}" from uploaded files.`,
        });
        onCollectionAttached(name);
      } else {
        const response = await ingestCollection(name, {
          tableRowLimit: rowLimit,
          tableRowFilter: tableRowFilter.trim() || null,
        });
        setStatus({
          type: "success",
          message: `Ingestion complete for "${response.collection}". Documents loaded from ${response.data_dir}.`,
        });
        setCollection(response.collection);
        onCollectionAttached(response.collection);
      }
    } catch (error: unknown) {
      let message = "Failed to ingest documents. Please try again.";
      if (typeof error === "object" && error !== null) {
        const axiosError = error as AxiosError<{ detail?: string }>;
        message =
          axiosError.response?.data?.detail ??
          axiosError.message ??
          message;
      } else if (error instanceof Error) {
        message = error.message;
      }
      setStatus({ type: "error", message });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Stack gap={6}>
      <Box>
        <Heading size="md" mb={2}>
          Prepare a collection for querying
        </Heading>
        <Text color="fg.muted">
          Ingestion loads documents from the server’s configured data directory
          into a Qdrant collection so they can be queried. Provide a collection
          name below to start the process.
        </Text>
      </Box>

      {status && (
        <Box
          p={4}
          borderRadius="md"
          bg={status.type === "error" ? "red.50" : "green.50"}
          color={status.type === "error" ? "red.800" : "green.800"}
          borderWidth="1px"
          borderColor={status.type === "error" ? "red.200" : "green.200"}
        >
          <Stack gap={1}>
            <Text fontWeight="bold" textTransform="capitalize">
              {status.type}
            </Text>
            <Text>{status.message}</Text>
          </Stack>
        </Box>
      )}

      <Stack gap={4}>
        <Box>
          <Text fontWeight="semibold" mb={1}>Collection name</Text>
          <Input
            value={collection}
            onChange={(event) => setCollection(event.target.value)}
            placeholder="e.g. invoices-2024"
            bg="bg.panel"
          />
        </Box>

        <Box>
          <Text fontWeight="semibold" mb={1}>Table row limit</Text>
          <Input
            type="number"
            value={tableRowLimit}
            onChange={(event) => setTableRowLimit(event.target.value)}
            placeholder="Optional maximum number of rows"
            bg="bg.panel"
          />
          <Text fontSize="sm" color="fg.muted" mt={1}>
            Applies to CSV, TSV, Excel, and Parquet files during ingestion.
          </Text>
        </Box>

        <Box>
          <Text fontWeight="semibold" mb={1}>Table row filter</Text>
          <Input
            value={tableRowFilter}
            onChange={(event) => setTableRowFilter(event.target.value)}
            placeholder={'e.g. status == "active" and amount > 100'}
            bg="bg.panel"
          />
          <Text fontSize="sm" color="fg.muted" mt={1}>
            Optional pandas-style query applied before ingesting table rows.
          </Text>
        </Box>

        <Box>
          <Text fontWeight="semibold" mb={1}>Upload files</Text>
          <Input
            type="file"
            multiple
            onChange={(event) => {
              const selected = event.target.files;
              setFiles(selected ? Array.from(selected) : []);
            }}
            bg="bg.panel"
            p={1}
          />
          <Text fontSize="sm" color="fg.muted" mt={1}>
            Uploaded files are stored in a temporary collection folder before ingestion.
          </Text>
          {files.length > 0 && (
            <Stack gap={1} mt={2} fontSize="sm" color="fg.muted">
              {files.map((file) => (
                <HStack key={file.name} justify="space-between">
                  <Text>{file.name}</Text>
                  {uploadProgress[file.name] && (
                    <Text>{`${Math.round(uploadProgress[file.name] / 1024)} KB`}</Text>
                  )}
                </HStack>
              ))}
            </Stack>
          )}
        </Box>
      </Stack>

      <Button
        onClick={submit}
        colorScheme="teal"
        loading={isLoading}
        width="full"
      >
        Start
      </Button>

      {progress.length > 0 && (
        <Box borderWidth="1px" borderColor="border.muted" borderRadius="md" p={3}>
          <Text fontWeight="bold" mb={2}>
            Upload progress
          </Text>
          <Stack gap={2}>
            {progress.map((event, idx) => {
              const label = (() => {
                switch (event.type) {
                  case "start":
                    return "Starting upload";
                  case "upload_progress":
                    return `Uploading ${event.payload.filename ?? "file"}`;
                  case "file_saved":
                    return `Saved ${(event.payload.filename as string) || "file"}`;
                  case "ingestion_started":
                    return "Ingestion started";
                  case "ingestion_complete":
                    return "Ingestion complete";
                  case "error":
                    return String(event.payload.message || "Upload failed");
                  default:
                    return "";
                }
              })();
              const percent = (() => {
                if (event.type !== "upload_progress") return undefined;
                const bytes = Number(event.payload.bytes_written ?? 0);
                // Without content-length we display a lightweight spinner-like progress bar.
                return Math.min(100, Math.max(10, Math.round(bytes / 1024)));
              })();
              return (
                <Stack key={`${event.type}-${idx}`} gap={1}>
                  <Flex justify="space-between" align="center">
                    <Text fontSize="sm">{label}</Text>
                    {event.type === "upload_progress" && (
                      <Text fontSize="xs" color="fg.muted">
                        {`${Math.round(Number(event.payload.bytes_written ?? 0) / 1024)} KB`}
                      </Text>
                    )}
                  </Flex>
                  {percent !== undefined && (
                    <Box
                      h="4px"
                      w="full"
                      bg="gray.100"
                      borderRadius="full"
                      overflow="hidden"
                    >
                      <Box
                        h="full"
                        bg="teal.500"
                        w={`${percent}%`}
                        transition="width 0.2s"
                      />
                    </Box>
                  )}
                </Stack>
              );
            })}
          </Stack>
        </Box>
      )}
    </Stack>
  );
}
