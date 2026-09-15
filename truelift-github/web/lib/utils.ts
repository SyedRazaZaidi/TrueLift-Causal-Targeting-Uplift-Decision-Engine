export function fmt(n: number | null | undefined, d = 3): string {
  if (n == null || Number.isNaN(n)) return "—";
  return Number(n).toLocaleString(undefined, { maximumFractionDigits: d });
}

export function pct(x: number): string {
  return `${(100 * x).toFixed(0)}%`;
}

export function colorKind(k: string): string {
  const m: Record<string, string> = {
    persuadable: "#5eead4",
    sure_thing: "#e8c468",
    lost_cause: "#64748b",
    sleeping_dog: "#f472b6",
  };
  return m[k] || "#60a5fa";
}

export const ROOMS = [
  { id: "stage", label: "Campaign" },
  { id: "lab", label: "Models" },
  { id: "identify", label: "Identify" },
  { id: "allocate", label: "Allocate" },
  { id: "desk", label: "People" },
  { id: "data", label: "Data" },
  { id: "risk", label: "Governance" },
] as const;

export type RoomId = (typeof ROOMS)[number]["id"];
