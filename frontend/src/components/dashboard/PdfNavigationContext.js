import { createContext } from 'react';

// Provides navigation from extracted fields -> PDF page viewer.
// value: { openPage: (pageNumber: number) => void }
export const PdfNavigationContext = createContext({
  openPage: () => {},
});
