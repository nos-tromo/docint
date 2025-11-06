import React from "react";
import ReactDOM from "react-dom/client";
import { ChakraProvider, Theme } from "@chakra-ui/react";
import App from "./App";
import system from "./theme";

const getPreferredAppearance = () => {
  if (typeof window === "undefined") {
    return "light" as const;
  }

  try {
    const stored = window.localStorage.getItem("chakra-ui-color-mode");
    if (stored === "light" || stored === "dark") {
      return stored;
    }
  } catch {
    // Access to localStorage can throw in private browsing or restricted contexts.
  }

  try {
    if (
      typeof window.matchMedia === "function" &&
      window.matchMedia("(prefers-color-scheme: dark)").matches
    ) {
      return "dark";
    }
  } catch {
    // matchMedia may throw in rare environments; ignore and fall back to light.
  }

  return "light" as const;
};

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ChakraProvider value={system}>
      <Theme appearance={getPreferredAppearance()}>
        <App />
      </Theme>
    </ChakraProvider>
  </React.StrictMode>,
);
