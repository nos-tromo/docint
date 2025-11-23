import { useEffect, useState } from "react";
import type { AxiosError } from "axios";
import {
  Alert,
  Box,
  Button,
  Field,
  Heading,
  Input,
  Stack,
  Text,
} from "@chakra-ui/react";
import { ingestCollection } from "../api";

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
        <Alert.Root status={status.type} borderRadius="md">
          <Alert.Indicator />
          <Alert.Content>
            <Alert.Title textTransform="capitalize">{status.type}</Alert.Title>
            <Alert.Description>{status.message}</Alert.Description>
          </Alert.Content>
        </Alert.Root>
      )}

      <Stack gap={4}>
        <Field.Root>
          <Field.Label fontWeight="semibold">Collection name</Field.Label>
          <Input
            value={collection}
            onChange={(event) => setCollection(event.target.value)}
            placeholder="e.g. invoices-2024"
            bg="bg.panel"
          />
        </Field.Root>

        <Field.Root>
          <Field.Label fontWeight="semibold">Table row limit</Field.Label>
          <Input
            type="number"
            value={tableRowLimit}
            onChange={(event) => setTableRowLimit(event.target.value)}
            placeholder="Optional maximum number of rows"
            bg="bg.panel"
          />
          <Field.HelperText color="fg.muted">
            Applies to CSV, TSV, Excel, and Parquet files during ingestion.
          </Field.HelperText>
        </Field.Root>

        <Field.Root>
          <Field.Label fontWeight="semibold">Table row filter</Field.Label>
          <Input
            value={tableRowFilter}
            onChange={(event) => setTableRowFilter(event.target.value)}
            placeholder={"e.g. status == \"active\" and amount > 100"}
            bg="bg.panel"
          />
          <Field.HelperText color="fg.muted">
            Optional pandas-style query applied before ingesting table rows.
          </Field.HelperText>
        </Field.Root>
      </Stack>

      <Button
        onClick={submit}
        colorScheme="teal"
        loading={isLoading}
        width="full"
      >
        Start
      </Button>
    </Stack>
  );
}
