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
      const response = await ingestCollection(name);
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
      </Stack>

      <Button
        onClick={submit}
        colorScheme="teal"
        loading={isLoading}
        width="full"
      >
        Start ingestion
      </Button>
    </Stack>
  );
}
