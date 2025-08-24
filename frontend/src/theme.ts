import { createSystem, defaultConfig } from "@chakra-ui/react";

const system = createSystem(defaultConfig, {
  theme: {
    tokens: {
      fonts: {
        body: {
          value:
            'Inter, system-ui, -apple-system, "Segoe UI", Roboto, Arial, "Noto Sans", "Apple Color Emoji", "Segoe UI Emoji"',
        },
        heading: {
          value:
            'Inter, system-ui, -apple-system, "Segoe UI", Roboto, Arial, "Noto Sans"',
        },
        mono: {
          value:
            '"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
        },
      },
    },
    semanticTokens: {
      colors: {
        // Surfaces
        "bg.canvas": { value: { base: "{colors.gray.50}", _dark: "{colors.gray.900}" } },
        "bg.panel": { value: { base: "{colors.white}", _dark: "{colors.gray.800}" } },
        "bg.muted": { value: { base: "{colors.gray.50}", _dark: "{colors.gray.700}" } },

        // Text
        "fg.default": { value: { base: "{colors.gray.900}", _dark: "{colors.gray.50}" } },
        "fg.muted": { value: { base: "{colors.gray.600}", _dark: "{colors.gray.300}" } },

        // Borders
        "border.muted": { value: { base: "{colors.gray.200}", _dark: "{colors.gray.700}" } },

        // Accent
        "accent.solid": { value: { base: "{colors.teal.600}", _dark: "{colors.teal.400}" } },
      },
    },
  },
});

export default system;
