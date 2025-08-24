import React from "react";
import ReactDOM from "react-dom/client";
import { ChakraProvider, Theme } from "@chakra-ui/react";
import App from "./App";
import system from "./theme";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ChakraProvider value={system}>
      <Theme appearance="dark">
        <App />
      </Theme>
    </ChakraProvider>
  </React.StrictMode>,
);
