import mermaid from "https://unpkg.com/mermaid@10/dist/mermaid.esm.min.mjs"

mermaid.initialize({
  startOnLoad: true,
  theme: 'default',
  themeVariables: {
    primaryColor: '#9c27b0',
    primaryTextColor: '#fff',
    primaryBorderColor: '#7b1fa2',
    lineColor: '#f50057',
    sectionBkgColor: '#f3e5f5',
    altSectionBkgColor: '#e1bee7',
    gridColor: '#e0e0e0',
    secondaryColor: '#e8eaf6',
    tertiaryColor: '#f5f5f5'
  }
});

document$.subscribe(() => {
  mermaid.contentLoaded();
});