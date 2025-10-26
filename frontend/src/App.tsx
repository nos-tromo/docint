import {
  Container,
  Heading,
  Box,
  Button,
  useDisclosure,
  Text,
  HStack,
  Spacer,
  Tabs,
  Stack,
} from "@chakra-ui/react";
import { useEffect, useState } from "react";
import CollectionPicker from "./components/CollectionPicker";
import Chat from "./components/Chat";
import { selectCollection } from "./api";
import IngestionPanel from "./components/IngestionPanel";

export default function App() {
  const { open, onOpen, onClose } = useDisclosure({ defaultOpen: true });
  const [collection, setCollection] = useState<string | null>(null);
  const [chatKey, setChatKey] = useState(0);

  useEffect(() => {
    const stored = localStorage.getItem("collection");
    if (stored) {
      setCollection(stored);
      selectCollection(stored).catch(() => {});
      onClose();
    }
  }, [onClose]);

  const attachCollection = (name: string) => {
    setCollection(name);
    localStorage.setItem("collection", name);
  };

  const quitSession = () => {
    if (collection) {
      localStorage.removeItem(`chat_msgs_${collection}`);
      localStorage.removeItem(`sessionId_${collection}`);
    }
    localStorage.removeItem("collection");
    setCollection(null);
    setChatKey((k) => k + 1);
    onOpen();
  };

  return (
    <Box bg="bg.canvas" color="fg.default" minH="100vh" fontFamily="body">
      <Container maxW="5xl" py={16}>
        <Stack gap={8}>
          <HStack>
            <Box>
              <Heading size="lg" fontFamily="heading">
                Document Intelligence
              </Heading>
              <Text color="fg.muted">Manage and explore your collections.</Text>
            </Box>
            <Spacer />
            <Button onClick={quitSession} variant="outline">
              Quit session
            </Button>
          </HStack>

          <CollectionPicker
            isOpen={open}
            onClose={onClose}
            onAttached={attachCollection}
          />

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
                  Querying
                </Tabs.Trigger>
                <Tabs.Trigger
                  value="ingest"
                  fontWeight="semibold"
                  justifyContent="center"
                  py={3}
                >
                  Ingestion
                </Tabs.Trigger>
              </Tabs.List>
              <Tabs.Content value="query" px={{ base: 4, md: 6 }} py={6}>
                <Stack gap={6}>
                  <Box display="flex" gap={3} alignItems="center">
                    <Button onClick={onOpen} variant="outline">
                      {collection ? `Collection: ${collection}` : "Select collection"}
                    </Button>
                    {collection && (
                      <Text fontSize="sm" color="fg.muted">
                        Attached
                      </Text>
                    )}
                  </Box>

                  <Chat key={chatKey} collection={collection} />
                </Stack>
              </Tabs.Content>
              <Tabs.Content value="ingest" px={{ base: 4, md: 6 }} py={6}>
                <IngestionPanel
                  currentCollection={collection}
                  onCollectionAttached={(name) => {
                    attachCollection(name);
                    selectCollection(name).catch(() => {});
                  }}
                />
              </Tabs.Content>
            </Tabs.Root>
          </Box>
        </Stack>
      </Container>
    </Box>
  );
}

