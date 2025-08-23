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
import { useState } from "react";
import CollectionPicker from "./components/CollectionPicker";
import Chat from "./components/Chat";

export default function App() {
  const { open, onOpen, onClose } = useDisclosure({ defaultOpen: true });
  const [collection, setCollection] = useState<string | null>(null);

  return (
    <Box bg="bg.canvas" color="fg.default" minH="100vh" fontFamily="body">
      <Container maxW="6xl" py={10}>
        <HStack mb={6}>
          <Heading size="lg" fontFamily="heading">
            Wizard RAG
          </Heading>
          <Spacer />
          {/* no toggle */}
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
          onAttached={(name) => setCollection(name)}
        />

        <Chat />
      </Container>
    </Box>
  );
}