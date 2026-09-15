export async function api<T>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${r.status} ${path}`);
  return r.json() as Promise<T>;
}
