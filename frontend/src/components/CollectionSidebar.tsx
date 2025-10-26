import { useState } from "react";
import {
  Box,
  Button,
  Heading,
  HStack,
  Spinner,
  Stack,
  Text,
} from "@chakra-ui/react";

type Props = {
  selectedCollection: string | null;
  collections: string[];
  onSelect: (name: string) => Promise<void> | void;
  onRefresh: () => void;
  error: string | null;
  isLoading: boolean;
};

export default function CollectionSidebar({
  selectedCollection,
  collections,
  onSelect,
  onRefresh,
  error,
  isLoading,
}: Props) {
  const [pendingSelection, setPendingSelection] = useState<string | null>(null);

  const handleSelect = async (name: string) => {
    setPendingSelection(name);
    try {
      await onSelect(name);
    } finally {
      setPendingSelection(null);
    }
  };

  const isBusy = pendingSelection !== null;

  return (
    <Box
      bg="bg.surface"
      borderRadius="2xl"
      boxShadow="xl"
      borderWidth="1px"
      borderColor="border.muted"
      p={5}
      w={{ base: "full", lg: "72" }}
      flexShrink={0}
    >
      <Stack gap={4}>
        <HStack justify="space-between" align="center">
          <Heading size="sm">Collections</Heading>
          <Button
            variant="ghost"
            size="sm"
            onClick={onRefresh}
            disabled={isLoading}
          >
            Refresh
          </Button>
        </HStack>

        {isLoading && collections.length === 0 ? (
          <HStack justify="center" py={10}>
            <Spinner color="fg.muted" />
          </HStack>
        ) : error ? (
          <Stack gap={3}>
            <Text color="red.400">{error}</Text>
            <Button onClick={onRefresh} size="sm" variant="outline">
              Try again
            </Button>
          </Stack>
        ) : collections.length === 0 ? (
          <Text color="fg.muted">No collections found.</Text>
        ) : (
          <Stack gap={2}>
            {collections.map((name) => {
              const isSelected = name === selectedCollection;
              const isPending = pendingSelection === name;
              return (
                <Button
                  key={name}
                  justifyContent="flex-start"
                  variant={isSelected ? "solid" : "ghost"}
                  colorScheme="teal"
                  onClick={() => handleSelect(name)}
                  disabled={isBusy}
                >
                  {isPending ? "Attaching…" : name}
                </Button>
              );
            })}
          </Stack>
        )}
      </Stack>
    </Box>
  );
}
