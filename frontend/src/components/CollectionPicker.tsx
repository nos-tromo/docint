import { useEffect, useState } from "react";
import type React from "react";
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
  const [selected, setSelected] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      setError(null);
      listCollections()
        .then((cols) => {
          setCollections(cols);
        })
        .catch((err: unknown) => {
          setError(
            err instanceof Error ? err.message : "Failed to load collections",
          );
          setCollections([]);
        });
    }
  }, [isOpen]);

  const attach = async () => {
    const name = selected.trim();
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
      <DialogContent bg="bg.panel" color="fg.default">
        <DialogHeader display="flex" justifyContent="space-between" alignItems="center">
          <Text fontWeight="bold">Select a collection</Text>
          <CloseButton onClick={onClose} color="fg.default" />
        </DialogHeader>
        <DialogBody>
          <Stack gap={3}>
            {error && <Text color="red.400">{error}</Text>}

            <label>
              <Text mb="1">Available collections</Text>
              <select
                value={selected}
                onChange={(e: React.ChangeEvent<HTMLSelectElement>) =>
                  setSelected(e.target.value)
                }
                style={{
                  width: "100%",
                  padding: "8px",
                  background: "var(--chakra-colors-bg-panel)",
                  color: "var(--chakra-colors-fg-default)",
                  borderRadius: "6px",
                  fontFamily: "system-ui, sans-serif", 
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
          </Stack>
        </DialogBody>
        <DialogFooter>
          <Button mr={3} onClick={onClose} variant="ghost">
            Cancel
          </Button>
          <Button onClick={attach} colorScheme="teal" disabled={!selected}>
            Use collection
          </Button>
        </DialogFooter>
      </DialogContent>
    </DialogRoot>
  );
}

