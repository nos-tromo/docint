import {
  Container,
  Heading,
  Box,
  Button,
  Text,
  HStack,
  Spacer,
  Tabs,
  Stack,
  Alert,
} from "@chakra-ui/react";
import { useCallback, useEffect, useState } from "react";
import CollectionSidebar from "./components/CollectionSidebar";
import Chat from "./components/Chat";
import { listCollections, selectCollection } from "./api";
import IngestionPanel from "./components/IngestionPanel";

export default function App() {
  const [collection, setCollection] = useState<string | null>(null);
  const [chatKey, setChatKey] = useState(0);
  const [selectionError, setSelectionError] = useState<string | null>(null);

  const [availableCollections, setAvailableCollections] = useState<string[]>([]);
  const [collectionsError, setCollectionsError] = useState<string | null>(null);
  const [loadingCollections, setLoadingCollections] = useState(false);

  const loadCollections = useCallback(async () => {
    setCollectionsError(null);
    setLoadingCollections(true);
    try {
      const names = await listCollections();
      setAvailableCollections(names);
    } catch (err: unknown) {
      setCollectionsError(
        err instanceof Error
          ? err.message
          : "Unable to load collections. Please try again.",
      );
    } finally {
      setLoadingCollections(false);
    }
  }, []);

  useEffect(() => {
    const stored = localStorage.getItem("collection");
    if (stored) {
      setCollection(stored);
      selectCollection(stored).catch(() => {});
    }
  }, []);

  const refreshCollections = useCallback(() => {
    void loadCollections();
  }, [loadCollections]);

  useEffect(() => {
    refreshCollections();
  }, [refreshCollections]);

  const attachCollection = (name: string) => {
    setCollection(name);
    localStorage.setItem("collection", name);
  };

  const handleSelectCollection = async (name: string) => {
    const trimmed = name.trim();
    if (!trimmed) return;
    setSelectionError(null);
    try {
      await selectCollection(trimmed);
      attachCollection(trimmed);
    } catch (err: unknown) {
      setSelectionError(
        err instanceof Error
          ? err.message
          : "Failed to switch collection. Please try again.",
      );
    }
  };

  const quitSession = () => {
    if (collection) {
      localStorage.removeItem(`chat_msgs_${collection}`);
      localStorage.removeItem(`sessionId_${collection}`);
    }
    localStorage.removeItem("collection");
    setCollection(null);
    setChatKey((k) => k + 1);
    setSelectionError(null);
  };

  const handleCollectionAttached = async (name: string) => {
    await handleSelectCollection(name);
    refreshCollections();
  };

  const collectionsToShow = collection
    ? availableCollections.includes(collection)
      ? availableCollections
      : [collection, ...availableCollections]
    : availableCollections;

  return (
    <Box bg="bg.canvas" color="fg.default" minH="100vh" fontFamily="body">
      <Container maxW="7xl" py={16}>
        <Stack gap={8}>
          <HStack>
            <Box>
              <Heading size="lg" fontFamily="heading">
                Document Intelligence
              </Heading>
              <Text color="fg.muted">Manage and explore your data.</Text>
            </Box>
            <Spacer />
            <Button onClick={quitSession} variant="outline">
              Quit session
            </Button>
          </HStack>

          {selectionError && (
            <Alert.Root status="error" borderRadius="md">
              <Alert.Indicator />
              <Alert.Content>
                <Alert.Title>Error</Alert.Title>
                <Alert.Description>{selectionError}</Alert.Description>
              </Alert.Content>
            </Alert.Root>
          )}

          <Stack
            direction={{ base: "column", lg: "row" }}
            align="stretch"
            gap={6}
          >
            <CollectionSidebar
              selectedCollection={collection}
              collections={collectionsToShow}
              onRefresh={refreshCollections}
              onSelect={handleSelectCollection}
              error={collectionsError}
              isLoading={loadingCollections}
            />

            <Box flex="1">
              <Box
                bg="bg.surface"
                borderRadius="2xl"
                boxShadow="xl"
                borderWidth="1px"
                borderColor="border.muted"
                overflow="hidden"
              >
                <Tabs.Root
                  defaultValue="query"
                  colorPalette="teal"
                  variant="subtle"
                >
                  <Tabs.List
                    display="grid"
                    gridTemplateColumns="repeat(2, minmax(0, 1fr))"
                    gap={0}
                    bg="bg.subtle"
                    p={1.5}
                  >
                    <Tabs.Trigger
                      value="query"
                      fontWeight="semibold"
                      justifyContent="center"
                      py={3}
                    >
                      Query
                    </Tabs.Trigger>
                    <Tabs.Trigger
                      value="ingest"
                      fontWeight="semibold"
                      justifyContent="center"
                      py={3}
                    >
                      Ingest
                    </Tabs.Trigger>
                  </Tabs.List>
                  <Tabs.Content value="query" px={{ base: 4, md: 6 }} py={6}>
                    <Stack gap={6}>
                      <Box>
                        <Text fontWeight="semibold" color="fg.muted">
                          Current collection
                        </Text>
                        <Text>
                          {collection
                            ? collection
                            : "Select a collection from the sidebar to begin."}
                        </Text>
                      </Box>

                      {collection ? (
                        <Chat key={chatKey} collection={collection} />
                      ) : (
                        <Box
                          borderWidth="1px"
                          borderColor="border.muted"
                          borderRadius="lg"
                          p={6}
                          textAlign="center"
                          bg="bg.panel"
                        >
                          <Text color="fg.muted">
                            Choose a collection from the sidebar to enable
                            querying.
                          </Text>
                        </Box>
                      )}
                    </Stack>
                  </Tabs.Content>
                  <Tabs.Content value="ingest" px={{ base: 4, md: 6 }} py={6}>
                    <IngestionPanel
                      currentCollection={collection}
                      onCollectionAttached={handleCollectionAttached}
                    />
                  </Tabs.Content>
                </Tabs.Root>
              </Box>
            </Box>
          </Stack>
        </Stack>
      </Container>
    </Box>
  );
}

