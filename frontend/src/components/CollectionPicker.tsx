import { useEffect, useState } from "react";
import {
  Button,
  DialogRoot,
  DialogBackdrop,
  DialogContent,
  DialogHeader,
  DialogBody,
  DialogFooter,
  CloseButton,
  Stack,
  Input,
  Text,
} from "@chakra-ui/react";
import { listCollections, selectCollection } from "../api";

type Props = {
  isOpen: boolean;
  onClose: () => void;
  onAttached: (name: string) => void;
};

export default function CollectionPicker({ isOpen, onClose, onAttached }: Props) {
  const [collections, setCollections] = useState<string[]>([]);
  const [mode, setMode] = useState<"pick" | "create">("pick");
  const [selected, setSelected] = useState<string>("");
  const [newName, setNewName] = useState("");

  useEffect(() => {
    if (isOpen) {
      listCollections().then(setCollections).catch(() => setCollections([]));
    }
  }, [isOpen]);

  const attach = async () => {
    const name = mode === "create" ? newName.trim() : selected || "default";
    if (!name) return;
    await selectCollection(name);
    onAttached(name);
    onClose();
  };

  return (
    <DialogRoot
      open={isOpen}
      onOpenChange={(e) => {
        if (!e.open) onClose();
      }}
    >
      <DialogBackdrop />
      <DialogContent>
        <DialogHeader display="flex" justifyContent="space-between" alignItems="center">
          <Text fontWeight="bold">Select or create a collection</Text>
          <CloseButton onClick={onClose} />
        </DialogHeader>
        <DialogBody>
          <Stack gap={3}>
            <label>
              <Text mb="1">Mode</Text>
              <select
                value={mode}
                onChange={(e: React.ChangeEvent<HTMLSelectElement>) =>
                  setMode(e.target.value as "pick" | "create")
                }
                style={{
                  width: "100%",
                  padding: "8px",
                  background: "transparent",
                  color: "inherit",
                  borderRadius: "6px",
                  border:
                    "1px solid var(--chakra-colors-border-muted, rgba(255,255,255,0.16))",
                }}
              >
                <option value="pick">Pick existing</option>
                <option value="create">Create new</option>
              </select>
            </label>

            {mode === "pick" ? (
              <label>
                <Text mb="1">Existing collections</Text>
                <select
                  value={selected}
                  onChange={(e: React.ChangeEvent<HTMLSelectElement>) =>
                    setSelected(e.target.value)
                  }
                  style={{
                    width: "100%",
                    padding: "8px",
                    background: "transparent",
                    color: "inherit",
                    borderRadius: "6px",
                    border:
                      "1px solid var(--chakra-colors-border-muted, rgba(255,255,255,0.16))",
                  }}
                >
                  <option value="" disabled>
                    Choose collection
                  </option>
                  {collections.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </label>
            ) : (
              <label>
                <Text mb="1">New collection name</Text>
                <Input
                  placeholder="my-collection"
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  bg="bg.panel"
                />
              </label>
            )}
          </Stack>
        </DialogBody>
        <DialogFooter>
          <Button mr={3} onClick={onClose} variant="ghost">
            Cancel
          </Button>
          <Button onClick={attach} colorScheme="teal">
            Use collection
          </Button>
        </DialogFooter>
      </DialogContent>
    </DialogRoot>
  );
}