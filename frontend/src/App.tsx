import {
  Container,
  Heading,
  Box,
  Button,
  useDisclosure,
  Text,
  HStack,
  Spacer,
} from "@chakra-ui/react";
import { useEffect, useState } from "react";
import CollectionPicker from "./components/CollectionPicker";
import Chat from "./components/Chat";
import { selectCollection } from "./api";

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
    localStorage.removeItem("chat_msgs");
    localStorage.removeItem("collection");
    setCollection(null);
    setChatKey((k) => k + 1);
    onOpen();
  };

  return (
    <Box bg="bg.canvas" color="fg.default" minH="100vh" fontFamily="body">
      <Container maxW="6xl" py={10}>
        <HStack mb={6}>
          <Heading size="lg" fontFamily="heading">
            Wizard RAG
          </Heading>
          <Spacer />
          <Button onClick={quitSession} variant="outline">
            Quit session
          </Button>
        </HStack>

        <Box mb={6} display="flex" gap={3} alignItems="center">
          <Button onClick={onOpen} variant="outline">
            {collection ? `Collection: ${collection}` : "Select collection"}
          </Button>
          {collection && (
            <Text fontSize="sm" color="fg.muted">
              Attached
            </Text>
          )}
        </Box>

        <CollectionPicker
          isOpen={open}
          onClose={onClose}
          onAttached={attachCollection}
        />

        <Chat key={chatKey} />
      </Container>
    </Box>
  );
}